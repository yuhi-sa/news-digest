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
        """Generate a curated briefing for engineers and JP/US stock investors."""
        article_list = "\n".join(
            f"- [{a.category}] {a.title}: {a.summary}" for a in articles
        )
        prompt = (
            "あなたは、ソフトウェアエンジニア兼日本株・米国株の個人投資家向けの"
            "ニュースキュレーターです。\n"
            "以下の本日のニュース記事一覧を分析し、日本語でブリーフィングを作成してください。\n\n"
            "## フォーマット（Markdown）\n\n"
            "### 本日のハイライト\n"
            "最も重要な3〜5件のニュースを箇条書きで簡潔にまとめる。\n\n"
            "### 米国株・マーケット\n"
            "米国株投資家が注目すべきポイントを整理する（FRB動向、決算、セクター別の動きなど）。\n\n"
            "### 日本株・アジア市場\n"
            "日本株投資家が注目すべきポイントを整理する（日銀政策、為替、アジア市場動向など）。\n"
            "該当ニュースが無い場合はセクションごと省略してよい。\n\n"
            "### エンジニアリング・テクノロジー\n"
            "エンジニアとして押さえておくべき技術トレンド・ツール・論文を整理する。\n\n"
            "## ルール\n"
            "- 各セクション3〜5項目に絞り、冗長にならないこと\n"
            "- 投資判断に直接関わる数字（指数、金利、為替）があれば含める\n"
            "- 煽りや感情的な表現は避け、事実ベースで淡々と書く\n"
            "- 記事一覧に該当トピックがない場合、そのセクションは省略する\n\n"
            f"## 本日の記事一覧（{len(articles)}件）\n\n"
            f"{article_list}"
        )
        logger.info("Generating investor/engineer briefing")
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
