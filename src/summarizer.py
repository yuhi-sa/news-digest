"""Article summarization (pluggable strategy)."""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from .parser import Article

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 3000


def _fetch_page_text(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return plain text extracted from HTML.

    Returns empty string on any failure.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NewsDigestBot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to fetch %s", url)
        return ""

    # Remove script/style blocks, then strip all tags
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_BODY_CHARS]


def _fetch_pages_parallel(
    urls: list[str], max_workers: int = 6,
) -> dict[str, str]:
    """Fetch multiple URLs in parallel. Returns {url: text}."""
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_fetch_page_text, url): url for url in urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = ""
    return results

_PROMPT_TEMPLATE = (
    "以下のニュース記事のタイトルと概要を読んで、日本語で1〜2文の簡潔な要約を書いてください。"
    "要約のみを返してください。\n\n"
    "タイトル: {title}\n"
    "概要: {summary}"
)

_BATCH_PROMPT_TEMPLATE = (
    "以下の複数のニュース記事について、それぞれ日本語で1〜2文の簡潔な要約を書いてください。\n"
    "各要約は番号付きで返してください（例: 1. 要約文）。\n"
    "要約のみを返してください。\n\n"
    "{articles}"
)


class Summarizer(ABC):
    """Base class for article summarizers."""

    @abstractmethod
    def summarize(self, articles: list[Article]) -> list[Article]:
        """Return articles with potentially updated summaries."""


class PassthroughSummarizer(Summarizer):
    """Uses RSS description as-is (no external API calls)."""

    def summarize(self, articles: list[Article]) -> list[Article]:
        logger.info("PassthroughSummarizer: keeping original summaries for %d articles", len(articles))
        return articles


class GeminiSummarizer(Summarizer):
    """Summarizes articles in Japanese using Google Gemini API (free tier)."""

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call_gemini(self, prompt: str) -> str | None:
        """Call Gemini API and return the generated text."""
        url = f"{self.ENDPOINT}?key={self.api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            logger.exception("Gemini API call failed")
            return None

    def _summarize_single(self, article: Article) -> Article:
        """Summarize a single article via Gemini API."""
        prompt = _PROMPT_TEMPLATE.format(title=article.title, summary=article.summary)
        ja_summary = self._call_gemini(prompt)
        if ja_summary:
            return replace(article, summary=ja_summary)
        logger.warning("Fallback to original summary for: %s", article.title)
        return article

    def _summarize_batch(self, batch: list[Article]) -> list[Article]:
        """Summarize a batch of articles in a single API call.

        Falls back to individual calls if the batch call fails.
        """
        articles_text = "\n".join(
            f"{i + 1}. タイトル: {a.title}\n   概要: {a.summary}"
            for i, a in enumerate(batch)
        )
        prompt = _BATCH_PROMPT_TEMPLATE.format(articles=articles_text)
        response = self._call_gemini(prompt)

        if response:
            summaries = self._parse_batch_response(response, len(batch))
            if summaries:
                results: list[Article] = []
                for article, summary in zip(batch, summaries):
                    results.append(replace(article, summary=summary))
                return results

        # Fallback: summarize individually
        logger.warning("Batch summarization failed, falling back to individual calls for %d articles", len(batch))
        return [self._summarize_single(a) for a in batch]

    @staticmethod
    def _parse_batch_response(response: str, expected_count: int) -> list[str] | None:
        """Parse numbered summaries from a batch response.

        Returns None if parsing fails or count doesn't match.
        """
        import re
        lines = response.strip().split("\n")
        summaries: list[str] = []
        current = ""
        for line in lines:
            match = re.match(r"^\d+[\.\)]\s*", line)
            if match:
                if current:
                    summaries.append(current.strip())
                current = line[match.end():]
            else:
                if current:
                    current += " " + line.strip()
        if current:
            summaries.append(current.strip())

        if len(summaries) == expected_count:
            return summaries
        logger.warning(
            "Batch response parse mismatch: expected %d, got %d",
            expected_count,
            len(summaries),
        )
        return None

    def summarize(self, articles: list[Article], batch_size: int = 5) -> list[Article]:
        logger.info("GeminiSummarizer: summarizing %d articles in Japanese (batch_size=%d)", len(articles), batch_size)
        results: list[Article] = []
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            results.extend(self._summarize_batch(batch))
        return results

    # ------------------------------------------------------------------
    # Two-stage briefing
    # ------------------------------------------------------------------

    def _select_articles(self, articles: list[Article]) -> list[int]:
        """Stage 1: Ask Gemini to pick the most important article indices."""
        article_list = "\n".join(
            f"{i}. [{a.category}] {a.title}: {a.summary}"
            for i, a in enumerate(articles)
        )
        prompt = (
            "あなたはデータエンジニア・セキュリティエンジニア兼日本株・米国株の個人投資家向けの"
            "シニアニュースアナリストです。\n"
            "以下の記事一覧から、読者にとって本当に重要な記事を**10〜15件**選んでください。\n\n"
            "## 読者の技術スタック\n"
            "読者は以下の技術を日常的に使うデータエンジニア・セキュリティエンジニアです。"
            "これらに関連する記事は優先的に選んでください:\n"
            "- 言語: TypeScript/Next.js, Python, Go, Spark\n"
            "- インフラ: Kubernetes, Kafka, MySQL, Cassandra, Redis, Hadoop, Athenz\n"
            "- データ基盤: dbt, Airflow, Databricks, BigQuery, Athena\n\n"
            "## 選定基準\n"
            "- 上記スタックに関連する重要アップデート・脆弱性・ベストプラクティス\n"
            "- 技術的に重要（AI/ML、データ基盤、セキュリティの新動向・脆弱性）\n"
            "- 投資判断に直結（マクロ指標、決算、セクター動向）\n"
            "- 些末なニュース、宣伝的な記事、既知の繰り返しは除外\n"
            "- 技術情報を優先、投資情報はサブ\n\n"
            "## 出力形式\n"
            "選んだ記事の番号をJSON配列で返してください。それ以外のテキストは不要です。\n"
            "例: [0, 3, 7, 12, 15]\n\n"
            f"## 記事一覧（{len(articles)}件）\n\n"
            f"{article_list}"
        )
        logger.info("Stage 1: selecting important articles from %d candidates", len(articles))
        response = self._call_gemini(prompt)
        if not response:
            return []

        # Extract JSON array from response
        try:
            match = re.search(r"\[[\d\s,]+\]", response)
            if match:
                indices = json.loads(match.group())
                valid = [i for i in indices if 0 <= i < len(articles)]
                logger.info("Stage 1: selected %d articles", len(valid))
                return valid
        except (json.JSONDecodeError, ValueError):
            pass
        logger.warning("Stage 1: failed to parse selection response")
        return []

    def generate_briefing(self, articles: list[Article]) -> str | None:
        """Generate a curated daily briefing using two-stage approach.

        Stage 1: Select important articles from RSS summaries.
        Stage 2: Fetch full text of selected articles, then generate deep briefing.
        """
        # Stage 1: Select
        selected_indices = self._select_articles(articles)
        if not selected_indices:
            logger.warning("Stage 1 returned no articles, falling back to summary-only briefing")
            selected = articles[:15]
        else:
            selected = [articles[i] for i in selected_indices]

        # Fetch full text of selected articles
        urls = [a.link for a in selected if a.link]
        logger.info("Stage 2: fetching full text for %d selected articles", len(urls))
        page_texts = _fetch_pages_parallel(urls)
        fetched = sum(1 for t in page_texts.values() if t)
        logger.info("Stage 2: successfully fetched %d/%d pages", fetched, len(urls))

        # Build enriched article list
        enriched_parts: list[str] = []
        for a in selected:
            body = page_texts.get(a.link, "")
            entry = (
                f"### [{a.category}] {a.title}\n"
                f"- URL: {a.link}\n"
                f"- RSS概要: {a.summary}\n"
            )
            if body:
                entry += f"- 記事本文（抜粋）: {body}\n"
            enriched_parts.append(entry)
        enriched_text = "\n".join(enriched_parts)

        # Stage 2: Generate briefing with full context
        prompt = (
            "あなたは、データエンジニア・セキュリティエンジニア兼日本株・米国株の個人投資家向けの"
            "ニュースレターライターです。\n"
            "以下の厳選されたニュース記事（本文付き）を分析し、日本語で**デイリーブリーフィング**を作成してください。\n\n"
            "## 読者プロフィール\n"
            "読者は以下の技術スタックを日常的に使うエンジニアです。"
            "記事の解説では、これらの技術との関連があれば積極的に言及してください:\n"
            "- 言語: TypeScript/Next.js, Python, Go, Spark\n"
            "- インフラ: Kubernetes, Kafka, MySQL, Cassandra, Redis, Hadoop, Athenz\n"
            "- データ基盤: dbt, Airflow, Databricks, BigQuery, Athena\n\n"
            "## 文章スタイル\n\n"
            "- **ニュースレターのように自然で読みやすい文体**で書く。テンプレートの穴埋めのような機械的な文章は避ける。\n"
            "- 1つのトピックは**3〜4行で完結**させる。長々と書かない。\n"
            "- 「→ それが意味すること:」「→ 実務への影響:」のような定型ラベルは使わない。"
            "代わりに、事実の説明から自然に「つまり〜」「ポイントは〜」「注目すべきは〜」と繋げる。\n"
            "- 1文を短く保つ（40字以内目安）。読点で繋げすぎない。\n"
            "- 同じ記事をハイライトと各セクションで重複させない。ハイライトで触れたものは各セクションでは省略する。\n"
            "- 関連する複数の記事は1つのトピックにまとめて論じてよい。\n\n"
            "## セクション構成（Markdown）\n\n"
            "### `## 🔥 本日のハイライト`\n"
            "最重要の3件。各項目は:\n"
            "- **太字の見出し**（1行）\n"
            "- 何が起きて、なぜ重要か（2〜3文を自然に繋げる）\n"
            "- 📎 [記事タイトル](URL)\n\n"
            "### `## 🛠️ テクノロジー`\n"
            "エンジニアとして押さえるべきトピックを2〜4件。\n"
            "記事本文を踏まえ、「何が新しいのか・なぜ重要か」を簡潔に。\n"
            "📎 リンクを各項目末尾に。\n\n"
            "### `## 📊 データエンジニアリング`\n"
            "データ基盤・パイプライン関連。該当なしなら省略。\n\n"
            "### `## 🔒 セキュリティ`\n"
            "脆弱性・攻撃動向・対策。該当なしなら省略。\n"
            "CVEは影響範囲と緊急度を明記。\n\n"
            "### `## 📈 マーケット`\n"
            "投資家向け。具体的な数字（金利・指数・為替等）を含める。\n\n"
            "### `## 🔮 今後の注目`\n"
            "直近のイベント・予測を2〜3点。\n\n"
            "## ルール\n"
            "- 記事本文の内容を踏まえて書く（RSS概要だけに頼らない）\n"
            "- 事実の羅列ではなく「だから何？」を常に意識する\n"
            "- 複数記事を横断的に結びつけてトレンドを抽出する\n"
            "- 煽りや感情的な表現は避ける\n"
            "- 冒頭の挨拶文や末尾の締め文は不要。セクションだけを出力する\n"
            "- 技術セクションを先に、投資セクションは後に配置する\n\n"
            f"## 厳選記事（{len(selected)}件・本文付き）\n\n"
            f"{enriched_text}"
        )
        logger.info("Stage 2: generating briefing with enriched content")
        return self._call_gemini(prompt)


def generate_briefing(articles: list[Article], api_key: str | None = None) -> str:
    """Generate a curated briefing. Returns empty string if no API key."""
    if not api_key:
        return ""
    summarizer = GeminiSummarizer(api_key=api_key)
    result = summarizer.generate_briefing(articles)
    return result or ""


def get_summarizer(api_key: str | None = None) -> Summarizer:
    """Factory: returns GeminiSummarizer if API key is available, else Passthrough."""
    if api_key:
        return GeminiSummarizer(api_key=api_key)
    return PassthroughSummarizer()
