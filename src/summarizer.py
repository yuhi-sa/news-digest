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

    def summarize(self, articles: list[Article]) -> list[Article]:
        logger.info("GeminiSummarizer: summarizing %d articles in Japanese", len(articles))
        results: list[Article] = []
        for article in articles:
            prompt = _PROMPT_TEMPLATE.format(title=article.title, summary=article.summary)
            ja_summary = self._call_gemini(prompt)
            if ja_summary:
                results.append(replace(article, summary=ja_summary))
            else:
                logger.warning("Fallback to original summary for: %s", article.title)
                results.append(article)
        return results

    def generate_briefing(self, articles: list[Article]) -> str | None:
        """Generate a curated weekly briefing for engineers and JP/US stock investors."""
        article_list = "\n".join(
            f"- [{a.category}] {a.title}: {a.summary}" for a in articles
        )
        prompt = (
            "あなたは、ソフトウェアエンジニア兼日本株・米国株の個人投資家向けの"
            "ニュースアナリストです。\n"
            "以下の今週のニュース記事一覧を分析し、日本語で**週次ブリーフィング**を作成してください。\n"
            "単なる記事の羅列ではなく、**なぜ重要なのか、どう影響するのか**を深掘りしてください。\n\n"
            "## フォーマット（Markdown・絵文字活用）\n\n"
            "```\n"
            "## 🔥 今週のハイライト\n"
            "今週最も重要な3〜5件を、**なぜ重要か**の一言解説付きで箇条書き。\n"
            "影響の大きさを直感的に示す。\n\n"
            "## 📈 米国株・マーケット\n"
            "米国株投資家が注目すべきポイントを深掘り。\n"
            "- FRB動向・金利見通し → ポートフォリオへの影響\n"
            "- セクター別の注目点（テック、エネルギー、金融など）\n"
            "- 具体的な数字（金利、指数水準、バリュエーション）を含める\n"
            "- 来週以降の注目イベント・決算があれば触れる\n\n"
            "## 🏯 日本株・アジア市場\n"
            "日本株投資家が注目すべきポイントを深掘り。\n"
            "- 日銀政策、為替動向（ドル円）、アジア市場の連動\n"
            "- 日本企業・産業に波及しうるグローバルトレンド\n"
            "- 該当ニュースが無い場合はセクションごと省略可\n\n"
            "## 🛠️ エンジニアリング・テクノロジー\n"
            "エンジニアとして押さえておくべき内容を深掘り。\n"
            "- 新しいツール・フレームワーク → 何が嬉しいのか、既存技術との比較\n"
            "- AI/ML の進展 → 実務への影響、使い所\n"
            "- 注目論文・OSS → 技術的に何が新しいのか\n"
            "- セキュリティ動向があれば含める\n\n"
            "## 🔮 来週の注目ポイント\n"
            "来週に控えるイベント・発表・トレンドの予測を2〜3点。\n"
            "記事の内容から推測できる範囲で。\n"
            "```\n\n"
            "## ルール\n"
            "- 各セクションの見出しには上記の絵文字を必ず使う\n"
            "- 表面的な要約に留まらず**「So What?（だから何？）」**を常に意識する\n"
            "- 複数の記事を横断的に結びつけ、トレンドやテーマを抽出する\n"
            "- 投資判断に関わる具体的な数字（金利、指数、為替、時価総額など）は積極的に含める\n"
            "- 煽りや感情的な表現は避け、事実と分析に基づいて書く\n"
            "- 記事一覧に該当トピックがないセクションは省略する\n"
            "- 項目が多い場合は重要度で絞り、各セクション3〜5項目を目安にする\n\n"
            f"## 今週の記事一覧（{len(articles)}件）\n\n"
            f"{article_list}"
        )
        logger.info("Generating weekly investor/engineer briefing")
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
