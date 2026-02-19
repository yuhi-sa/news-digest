"""Article summarization (pluggable strategy)."""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from .parser import Article

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 10000


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


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def call_gemini(prompt: str, api_key: str, max_retries: int = 2) -> str | None:
    """Call Gemini API with retry logic and return the generated text.

    Retries up to max_retries times on failure with backoff.
    """
    url = f"{GEMINI_ENDPOINT}?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
    }).encode("utf-8")

    for attempt in range(max_retries + 1):
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
            if attempt < max_retries:
                delay = 5 * (attempt + 1)
                logger.warning(
                    "Gemini API call failed (attempt %d/%d), retrying in %ds",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
            else:
                logger.exception(
                    "Gemini API call failed after %d attempts", max_retries + 1,
                )
    return None


class GeminiSummarizer(Summarizer):
    """Summarizes articles in Japanese using Google Gemini API (free tier)."""

    ENDPOINT = GEMINI_ENDPOINT

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _call_gemini(self, prompt: str) -> str | None:
        """Call Gemini API and return the generated text."""
        return call_gemini(prompt, self.api_key)

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
            "以下の記事一覧から、読者にとって本当に重要な記事を**8〜10件**選んでください。\n\n"
            "## 読者の技術スタック\n"
            "読者は以下の技術を日常的に使うデータエンジニア・セキュリティエンジニアです。"
            "これらに関連する記事は優先的に選んでください:\n"
            "- 言語: TypeScript/Next.js, Python, Go, Spark\n"
            "- インフラ: Kubernetes, Kafka, MySQL, Cassandra, Redis, Hadoop, Athenz\n"
            "- データ基盤: dbt, Airflow, Databricks, BigQuery, Athena\n\n"
            "## 必須の選定配分\n"
            "以下のカテゴリごとに最低限の記事数を確保すること:\n"
            "- セキュリティ: 3〜5件（実際に悪用されているCVE、重大な脆弱性、攻撃キャンペーンのみ。"
            "一般論や啓蒙記事は除外）\n"
            "- マーケット/投資: 2〜3件（具体的数値・指標・決算を含む記事を優先。"
            "数字のない一般的な経済論評は除外）\n"
            "- データエンジニアリング: 1〜3件（dbt/Airflow/Spark/BigQuery等の具体的ツール更新・"
            "アーキテクチャ変更を含む記事）\n"
            "- テクノロジー全般: 3〜5件（読者スタックに直結する記事を優先）\n\n"
            "## 選定基準（優先順）\n"
            "1. 具体的な数値・メトリクス・CVE番号を含む記事を最優先\n"
            "2. 上記スタックに関連する重要アップデート・脆弱性・ベストプラクティス\n"
            "3. 投資判断に直結（マクロ指標の具体数値、決算、セクター動向）\n"
            "4. 些末なニュース、宣伝的な記事、既知の繰り返しは除外\n"
            "5. 量より質: 似たテーマの記事は最も情報量の多い1件だけ選ぶ\n\n"
            "## 出力形式\n"
            "選んだ記事の番号をJSON配列で返してください。それ以外のテキストは不要です。\n"
            "例: [0, 3, 5, 7, 9, 12, 15, 18]\n\n"
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

    _BRIEFING_MIN_CHARS = 200

    def generate_briefing(self, articles: list[Article]) -> str | None:
        """Generate a curated daily briefing using two-stage approach.

        Stage 1: Select important articles from RSS summaries.
        Stage 2: Fetch full text of selected articles, then generate deep briefing.
        Includes retry logic for empty or too-short results.
        """
        # Stage 1: Select
        selected_indices = self._select_articles(articles)
        if not selected_indices:
            logger.warning("Stage 1 returned no articles, falling back to summary-only briefing")
            selected = articles[:10]
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
            "あなたはベテランのテックジャーナリストです。データエンジニア・セキュリティエンジニア兼"
            "個人投資家（日米株）向けのデイリーブリーフィングを日本語で作成してください。\n\n"
            "## 読者\n"
            "- 技術スタック: TypeScript/Next.js, Python, Go, Spark, "
            "Kubernetes, Kafka, MySQL, Cassandra, Redis, Hadoop, Athenz, "
            "dbt, Airflow, Databricks, BigQuery, Athena\n"
            "- 読者のスタックに直結する話題は技術名を挙げて影響を具体的に述べる\n"
            "- 読者は日米の個別株・ETFに投資している。ニュースの投資インパクトを知りたい\n\n"
            "## 禁止表現（これらを使ったら書き直す）\n"
            "- 「〜に注目が集まっています」「〜が重要です」「〜が求められています」\n"
            "- 「〜の可能性があります」で終わる文\n"
            "- 「エンジニアは注意が必要です」「対策が急務です」\n"
            "- 「〜が進んでいます」「〜が加速しています」\n"
            "- 「今後の動向に注目」「引き続き注視」\n"
            "- 「〜が期待されます」「〜が見込まれます」（根拠なしの場合）\n"
            "- 同じ語尾の3連続（「〜した。〜した。〜した。」は不可）\n\n"
            "## 文体\n"
            "- 1トピック5〜8行。事実・背景・読者への影響を踏み込んで書く\n"
            "- 1文は40字以内。長い文は分割する\n"
            "- 基本構成: 事実(1〜2文) ＋ 技術的背景(1〜2文) ＋ 読者の業務への影響(1〜2文)\n"
            "- 全トピックの末尾に 📎 [記事タイトル](URL) 必須。例外なし\n"
            "- 複数の関連記事は1トピックにまとめてよい\n"
            "- 各バレットポイントには必ず1つ以上の具体的事実（数値、固有名詞、バージョン番号、"
            "CVE番号など）を含める。具体性のないバレットは書かない\n\n"
            "## セクション構成\n\n"
            "### `## 🔥 本日のハイライト`\n"
            "最重要の3件のみ。各セクションと重複しないこと。\n"
            "- **太字見出し**（10字前後）\n"
            "- 事実1文 + 意味1文\n"
            "- 📎 リンク\n\n"
            "### `## 🛠️ テクノロジー`\n"
            "読者の技術スタック（TypeScript, Python, Go, K8s, Kafka等）に直結するトピックのみ。\n"
            "ハイライトと重複しない別のトピック。最大3件。\n"
            "具体的なバージョン番号、API変更点、マイグレーション手順があれば明記。\n"
            "📎 リンク必須。\n\n"
            "### `## 📊 データエンジニアリング`\n"
            "データ基盤・パイプライン関連。該当なしなら省略。最大3件。\n"
            "dbt/Airflow/Spark/BigQuery/Databricks等の具体名で影響を述べる。\n"
            "ツールのバージョン、設定変更点、パフォーマンス改善の具体数値を含める。\n"
            "📎 リンク必須。\n\n"
            "### `## 🔒 セキュリティ`\n"
            "脆弱性・攻撃動向。該当なしなら省略。**最大5件、影響度順**。\n"
            "各項目に必須: (1)CVE番号（あれば）, (2)影響を受けるソフトウェア・バージョン, "
            "(3)深刻度（Critical/High/Medium）, (4)具体的対応策（パッチ適用、設定変更等）\n"
            "類似の脆弱性は1トピックにまとめる。\n"
            "📎 リンク必須。\n\n"
            "### `## 📈 マーケット`\n"
            "**記事本文から抽出した具体的数値のみ記載**。以下を可能な限り含む:\n"
            "- 株価指数（S&P500, NASDAQ, 日経225, TOPIX）の数値と前日比%\n"
            "- 為替（USD/JPY）の水準\n"
            "- 米国債利回り（10年）の水準\n"
            "- 個別銘柄の決算・株価変動（ティッカーシンボル付き）\n"
            "**記事に数値がない場合は「データ不足：該当記事に具体的数値の記載なし」と正直に書く。**\n"
            "数値を捏造・推測しないこと。\n"
            "📎 リンク必須。\n\n"
            "### `## 🔮 今後の注目`\n"
            "1〜2週間以内のイベント・予測を2〜3点。**具体的な日付を必ず明記**。\n"
            "漠然とした予測は書かない。\n\n"
            "## ルール\n"
            "- 記事本文を踏まえて書く（RSS概要だけに頼らない）\n"
            "- 「だから何？」を常に意識。事実の羅列は不可\n"
            "- 複数記事を横断的に結びつけてトレンドを抽出\n"
            "- ハイライトの記事は他セクションに書かない（重複厳禁）\n"
            "- 冒頭挨拶・末尾締め不要。セクションだけ出力\n"
            "- 記事に書かれていない数値や事実を捏造しない\n\n"
            f"## 厳選記事（{len(selected)}件・本文付き）\n\n"
            f"{enriched_text}"
        )
        logger.info("Stage 2: generating briefing with enriched content")
        draft = self._call_gemini(prompt)
        if not draft:
            logger.error("Stage 2: Gemini returned no content")
            return None
        if len(draft) < self._BRIEFING_MIN_CHARS:
            logger.warning(
                "Stage 2: briefing unusually short (%d chars < %d minimum)",
                len(draft), self._BRIEFING_MIN_CHARS,
            )

        # Stage 3: LLM-based refinement then deterministic post-processing
        refined = self._refine_briefing(draft)
        return self._post_process_briefing(refined)

    def _refine_briefing(self, draft: str) -> str:
        """Stage 3: Self-critique and refine the briefing for quality."""
        prompt = (
            "以下のデイリーブリーフィングの原稿を校正・改善してください。\n\n"
            "## 品質チェック項目（不合格なら修正）\n"
            "1. 📎リンクのないトピックがあれば、そのトピックを削除する\n"
            "2. 以下の定型表現があれば具体的な表現に書き換える:\n"
            "   - 「〜に注目が集まっています」→ 具体的に誰が何に注目しているか\n"
            "   - 「〜が重要です」→ なぜ重要かを具体的に\n"
            "   - 「注意が必要です」→ 具体的に何をすべきか\n"
            "   - 「〜の可能性があります」→ 根拠を示して断定するか削除\n"
            "   - 「今後の動向に注目」「引き続き注視」→ 削除するか具体的な日付・イベントに置換\n"
            "   - 「〜が期待されます」→ 誰がなぜ期待しているか具体的に\n"
            "3. 同じ語尾が3回以上連続していたら語尾を変える\n"
            "4. 1トピックが9行以上なら8行以内に削る\n"
            "5. **マーケットセクション**: 具体的な数値（指数、%、ティッカー）が1つもなければ、\n"
            "   セクション冒頭に「データ不足：該当記事に具体的数値の記載なし」と追記\n"
            "6. **セキュリティセクション**: 各項目にCVE番号または具体的なソフトウェア名がなければ、\n"
            "   その項目を削除するか具体化する\n"
            "7. ハイライトと他セクションで同じ記事（同じURL）を扱っていたら他セクション側を削除\n"
            "8. 具体的事実（数値、固有名詞、バージョン等）を1つも含まないバレットポイントは削除\n\n"
            "## ルール\n"
            "- Markdownのセクション構造はそのまま維持する\n"
            "- 情報を追加・捏造しない。原稿にある情報だけで改善する\n"
            "- 改善後のブリーフィング全文のみを出力する。説明やコメントは不要\n\n"
            "## 原稿\n\n"
            f"{draft}"
        )
        logger.info("Stage 3: refining briefing quality")
        refined = self._call_gemini(prompt)
        return refined or draft

    # ------------------------------------------------------------------
    # Deterministic post-processing (no LLM calls)
    # ------------------------------------------------------------------

    _BANNED_PHRASES = [
        "注目が集まっています",
        "注目が集まって",
        "が重要です",
        "が求められています",
        "の可能性があります",
        "注意が必要です",
        "対策が急務です",
        "が進んでいます",
        "が加速しています",
        "今後の動向に注目",
        "引き続き注視",
        "が期待されます",
        "が見込まれます",
    ]

    @staticmethod
    def _section_has_link(section_text: str) -> bool:
        """Check if a section contains at least one 📎 markdown link."""
        return bool(re.search(r"📎\s*\[.*?\]\(https?://.*?\)", section_text))

    @staticmethod
    def _market_section_has_numbers(section_text: str) -> bool:
        """Check if market section contains actual numeric data."""
        return bool(re.search(
            r"\d+[,.]?\d*\s*%|"           # percentages like 3.5%
            r"(?:S&P|NASDAQ|日経|TOPIX|USD/JPY|ドル円)\s*[\d,]+|"  # index values
            r"\$\s*[\d,]+|"                # dollar amounts
            r"[\d,]+\s*(?:円|ドル|bps)",   # yen/dollar/bps amounts
            section_text,
        ))

    def _post_process_briefing(self, text: str) -> str:
        """Deterministic quality checks applied after LLM refinement."""
        sections = re.split(r"(^## .+$)", text, flags=re.MULTILINE)
        result_parts: list[str] = []
        banned_found: list[str] = []

        i = 0
        while i < len(sections):
            part = sections[i]
            # Check if this is a section header
            if part.startswith("## "):
                header = part
                body = sections[i + 1] if i + 1 < len(sections) else ""
                combined = header + body

                # Drop sections without links (except 今後の注目 which may not need them)
                if "🔮" not in header and not self._section_has_link(combined):
                    logger.warning(
                        "Post-process: dropping section without links: %s",
                        header.strip(),
                    )
                    i += 2
                    continue

                # Market section: inject data-insufficient notice if no numbers
                if "マーケット" in header and not self._market_section_has_numbers(body):
                    body = "\nデータ不足：該当記事に具体的数値の記載なし\n" + body
                    logger.info("Post-process: added data-insufficient notice to market section")

                result_parts.append(header)
                result_parts.append(body)
                i += 2
            else:
                result_parts.append(part)
                i += 1

        processed = "".join(result_parts)

        # Log banned phrases still present (for monitoring, not removal --
        # removing mid-sentence could break readability)
        for phrase in self._BANNED_PHRASES:
            count = processed.count(phrase)
            if count > 0:
                banned_found.append(f"'{phrase}' x{count}")
        if banned_found:
            logger.warning(
                "Post-process: banned phrases still present: %s",
                ", ".join(banned_found),
            )

        # Check for duplicate URLs across highlight and other sections
        highlight_urls: set[str] = set()
        in_highlight = False
        for line in processed.split("\n"):
            if "🔥" in line and line.startswith("## "):
                in_highlight = True
            elif line.startswith("## "):
                in_highlight = False
            if in_highlight:
                for url_match in re.finditer(r"\(https?://[^\s)]+\)", line):
                    highlight_urls.add(url_match.group())

        if highlight_urls:
            dup_count = 0
            for url in highlight_urls:
                # Count occurrences outside highlight section
                all_occurrences = processed.count(url)
                if all_occurrences > 1:
                    dup_count += 1
            if dup_count:
                logger.warning(
                    "Post-process: %d URL(s) appear in both highlights and other sections",
                    dup_count,
                )

        return processed


def generate_briefing(articles: list[Article], api_key: str | None = None) -> str:
    """Generate a curated briefing. Returns empty string if no API key."""
    if not api_key:
        logger.warning("No API key provided, skipping briefing generation")
        return ""
    if not articles:
        logger.warning("No articles provided, skipping briefing generation")
        return ""
    summarizer = GeminiSummarizer(api_key=api_key)
    result = summarizer.generate_briefing(articles)
    if not result:
        logger.error(
            "Briefing generation failed after all retries for %d articles",
            len(articles),
        )
    return result or ""


def get_summarizer(api_key: str | None = None) -> Summarizer:
    """Factory: returns GeminiSummarizer if API key is available, else Passthrough."""
    if api_key:
        return GeminiSummarizer(api_key=api_key)
    return PassthroughSummarizer()
