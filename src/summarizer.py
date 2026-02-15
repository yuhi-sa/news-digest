"""Article summarization (pluggable strategy)."""

from __future__ import annotations

import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import replace

from .parser import Article

logger = logging.getLogger(__name__)

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

    def generate_briefing(self, articles: list[Article]) -> str | None:
        """Generate a curated daily briefing for data/security engineers and JP/US stock investors."""
        article_list = "\n".join(
            f"- [{a.category}] {a.title}: {a.summary} (link: {a.link})"
            for a in articles
        )
        prompt = (
            "あなたは、データエンジニア・セキュリティエンジニア兼日本株・米国株の個人投資家向けの"
            "シニアニュースアナリストです。\n"
            "以下の本日のニュース記事一覧を分析し、日本語で**デイリーブリーフィング**を作成してください。\n\n"
            "## 最重要方針\n\n"
            "1. **厳選**: 記事一覧の全てを載せるのではなく、読者にとって本当に価値のある情報だけを選ぶ。"
            "ノイズ・重複・些末な話題は捨てる。全体で10〜15トピック程度に絞る。\n"
            "2. **解説**: 「何が起きたか」だけでなく「それが何を意味するのか」「読者は何をすべきか」を必ず書く。\n"
            "3. **ソースリンク**: 各トピックの末尾に関連する元記事のリンクを貼る。\n"
            "4. **技術情報がメイン、投資情報はサブ**という優先度で構成する。\n\n"
            "## フォーマット（Markdown・絵文字活用）\n\n"
            "### `## 🔥 本日のハイライト`\n"
            "本日最も重要な3件を厳選。各項目に:\n"
            "- 何が起きたか（1〜2文）\n"
            "- **→ So What?**: なぜ読者に関係あるか、何を意味するか（2〜3文で深掘り）\n"
            "- 📎 [記事タイトル](URL)\n\n"
            "### `## 🛠️ エンジニアリング・テクノロジー`\n"
            "**最も重要なセクション。** エンジニアとして知っておくべき情報を厳選して深掘り:\n"
            "- 各項目は「事実 → それが意味すること → 実務への影響」の3段構成で書く\n"
            "- AI/ML、新ツール・OSS、注目論文、インフラ・クラウドなどの中から重要なものだけ\n"
            "- 各項目の末尾に 📎 [記事タイトル](URL) を付ける\n\n"
            "### `## 📊 データエンジニアリング`\n"
            "データエンジニアが実務で使える情報のみ:\n"
            "- dbt, Airflow, Spark, Snowflake, Databricks, BigQuery等の重要アップデート\n"
            "- データ品質・オブザーバビリティ・ガバナンスの実践的な話題\n"
            "- 各項目の末尾に 📎 [記事タイトル](URL) を付ける\n"
            "- 該当記事がない場合はセクション省略\n\n"
            "### `## 🔒 セキュリティ`\n"
            "セキュリティエンジニアがアクションを取るべき情報:\n"
            "- 重大な脆弱性・CVE → 影響範囲と対応の緊急度を明記\n"
            "- 攻撃手法のトレンド → 防御側として具体的に何をすべきか\n"
            "- 各項目の末尾に 📎 [記事タイトル](URL) を付ける\n"
            "- 該当記事がない場合はセクション省略\n\n"
            "### `## 📈 投資・マーケット`\n"
            "日米株の個人投資家向け。**アクショナブルな情報**を厳選:\n"
            "- 📌 **注目セクター・銘柄**: ニュースから導かれる投資機会とその根拠\n"
            "- マクロ動向（FRB/日銀、金利、為替）→ ポジションへの影響\n"
            "- 具体的な数字（金利、指数、為替、PER等）を必ず含める\n"
            "- 各項目の末尾に 📎 [記事タイトル](URL) を付ける\n\n"
            "### `## 🔮 明日以降の注目ポイント`\n"
            "直近に控えるイベント・予測を2〜3点:\n"
            "- 経済指標発表、企業決算、カンファレンス等\n"
            "- 本日の流れから今後起こりそうなこと\n\n"
            "## ルール\n"
            "- **取捨選択が最重要**: 記事一覧は大量にあるが、読者の時間を節約するため本当に重要なものだけ選ぶ\n"
            "- 些末なニュース、宣伝的な記事、既知の情報の繰り返しは除外する\n"
            "- 表面的な要約ではなく「**それが何を意味するのか**」を必ず解説する\n"
            "- 複数記事を横断的に結びつけ、大きなトレンドやテーマを抽出する\n"
            "- 各項目に必ず元記事のリンクを 📎 Markdownリンク形式で付ける\n"
            "- 煽りや感情的な表現は避け、事実と分析に基づく\n"
            "- 該当トピックがないセクションは省略する\n"
            "- **技術セクション（🛠️📊🔒）を先に、投資セクション（📈）は後に**配置する\n\n"
            f"## 本日の記事一覧（{len(articles)}件）\n\n"
            f"{article_list}"
        )
        logger.info("Generating daily investor/engineer briefing")
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
