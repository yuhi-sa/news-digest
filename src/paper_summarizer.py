"""Structured paper summarization via Gemini API (two-stage)."""

from __future__ import annotations

import logging

from .paper_fetcher import Paper
from .summarizer import call_gemini

logger = logging.getLogger(__name__)

# Stage 1: Prerequisites, background, and method
_STAGE1_PROMPT = """\
あなたはコンピュータサイエンスの研究論文を、実務経験のあるエンジニア向けにわかりやすく解説する専門家です。
以下の論文について、日本語で前半部分の解説を作成してください。

## 論文情報
- タイトル: {title}
- 著者: {authors}
- 発表年: {year}
- 被引用数: {citation_count}
- 分野: {category_ja}

## アブストラクト
{abstract}

## 出力形式
以下の3セクションをMarkdownで出力してください。

### 🎓 前提知識
この論文を理解するために必要な基礎概念を2〜3個、それぞれ2〜3文で説明する。
読者はソフトウェアエンジニアだが、この論文の専門分野には詳しくない前提で書く。
例: 分散合意アルゴリズムの論文なら「分散システムにおける合意問題」「CAP定理」など。

### 📖 背景と動機
この論文が発表された当時の技術的状況と、解決しようとした課題。5〜8文で具体的に。
既存手法の限界は何だったのか、なぜ新しいアプローチが必要だったのかを明確にする。

### 🔬 手法・アプローチ
提案された手法やシステムの核心的なアイデアを詳細に説明する。5〜8文。
具体的なアルゴリズムやアーキテクチャの特徴を、技術的に正確かつ平易に記述する。

## ルール
- 技術的に正確であること。不明な点は推測せず「詳細は原論文を参照」と記す
- 冒頭の挨拶や末尾の締め文は不要。セクションのみを出力する
"""

# Stage 2: Architecture diagram, contributions, impact, keywords
_STAGE2_PROMPT = """\
あなたはコンピュータサイエンスの研究論文を解説する専門家です。
以下の論文の前半解説を踏まえて、後半部分を作成してください。

## 論文情報
- タイトル: {title}
- 著者: {authors}
- 発表年: {year}
- 被引用数: {citation_count}
- 分野: {category_ja}

## アブストラクト
{abstract}

## 前半の解説（参考）
{stage1_summary}

## 出力形式
以下の4セクションをMarkdownで出力してください。

### 🏗️ アーキテクチャ図
論文の提案手法やシステム構成を表すMermaid図を1つ作成する。
図の種類は内容に応じて最適なものを選ぶ（flowchart, sequence diagram, block-beta等）。
```mermaid
（ここに図）
```
図の直後に、図の読み方を2〜3文で補足する。

### 💡 主要な貢献
この論文が分野にもたらした具体的な成果や新規性。3〜5項目を箇条書きで。
各項目は1〜2文で、何が新しく、なぜ重要かを明確にする。

### 🌍 影響と意義
この研究が後続の研究や実務に与えた影響。5〜8文。
- 被引用数{citation_count}件の背景にある理由
- この論文から派生した技術やプロダクト（具体名を挙げる）
- 現在の実務でどのように活用されているか

### 📚 関連キーワード
この論文に関連する技術用語やキーワードを5〜8個、箇条書きで列挙する。
各キーワードに1文の簡潔な説明を付ける。

## ルール
- 技術的に正確であること
- Mermaid図は必ず含める。GitHubで正しく表示されるMermaid記法を使うこと
- 冒頭の挨拶や末尾の締め文は不要。セクションのみを出力する
"""

_FALLBACK_TEMPLATE = """\
### 🎓 前提知識
この論文は{category_ja}分野に関するものです。詳細な前提知識についてはアブストラクトおよび原論文を参照してください。

### 📖 背景と動機
{abstract_short}

### 🔬 手法・アプローチ
詳細は原論文を参照してください。

### 💡 主要な貢献
- 被引用数 {citation_count} 件の高インパクト論文です

### 🌍 影響と意義
{category_ja}分野における重要な研究です。

### 📚 関連キーワード
- 詳細は原論文を参照してください
"""


def _format_authors(authors: list[str]) -> str:
    """Format author list, truncating if more than 5."""
    authors_str = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_str += f" 他{len(authors) - 5}名"
    return authors_str


def summarize_paper(paper: Paper, api_key: str | None) -> str:
    """Generate a structured summary using two Gemini API calls.

    Stage 1: Prerequisites + background + method
    Stage 2: Architecture diagram + contributions + impact + keywords

    Falls back to a basic summary if no API key or on failure.
    """
    if not api_key:
        logger.info("No API key, using fallback summary")
        return _fallback_summary(paper)

    authors_str = _format_authors(paper.authors)
    abstract = paper.abstract or f"(アブストラクト未登録。タイトル「{paper.title}」から内容を推測してください)"

    fmt_args = {
        "title": paper.title,
        "authors": authors_str,
        "year": paper.year or "不明",
        "citation_count": paper.citation_count,
        "category_ja": paper.category_ja,
        "abstract": abstract,
    }

    # Stage 1
    logger.info("Stage 1: generating prerequisites, background, and method")
    stage1 = call_gemini(_STAGE1_PROMPT.format(**fmt_args), api_key)
    if not stage1:
        logger.warning("Stage 1 failed, using fallback for: %s", paper.title)
        return _fallback_summary(paper)

    # Stage 2
    logger.info("Stage 2: generating diagram, contributions, and impact")
    stage2 = call_gemini(
        _STAGE2_PROMPT.format(**fmt_args, stage1_summary=stage1),
        api_key,
    )
    if not stage2:
        logger.warning("Stage 2 failed, returning stage 1 only for: %s", paper.title)
        return stage1

    return f"{stage1}\n\n{stage2}"


def _fallback_summary(paper: Paper) -> str:
    """Generate a basic summary without LLM."""
    abstract_short = paper.abstract[:300] if paper.abstract else paper.title
    if len(paper.abstract or "") > 300:
        abstract_short += "..."

    return _FALLBACK_TEMPLATE.format(
        abstract_short=abstract_short,
        citation_count=paper.citation_count,
        category_ja=paper.category_ja,
    )
