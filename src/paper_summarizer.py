"""Structured paper summarization via Gemini API (two-stage)."""

from __future__ import annotations

import logging

from .paper_fetcher import Paper
from .summarizer import _fetch_page_text, call_gemini

logger = logging.getLogger(__name__)

_PDF_TEXT_LIMIT = 6000

# Stage 1: Prerequisites, background, and method
_STAGE1_PROMPT = """\
あなたは「古い論文がなぜ今も重要なのか」を伝えるテックライターです。
実務経験3〜5年のソフトウェアエンジニアが「この論文、読んでみたい」と思えるような解説を日本語で書いてください。

## 論文情報
- タイトル: {title}
- 著者: {authors}
- 発表年: {year}
- 被引用数: {citation_count}
- 分野: {category_ja}

## アブストラクト
{abstract}
{pdf_section}
## 出力形式
以下の3セクションをMarkdownで出力してください。

### 🎓 前提知識
この論文を理解するために必要な基礎概念を2〜3個、それぞれ2〜3文で説明する。
各概念には**現実世界のたとえ**を1つ必ず入れること（例: 「Paxosの合意は、全員が同じレストランを選ぶグループLINEのようなもの」）。
読者はソフトウェアエンジニアだが、この論文の専門分野には詳しくない前提で書く。

### 📖 背景と動機
当時エンジニアが直面していた**具体的な問題**にフォーカスする。5〜8文。
抽象的な理論の説明ではなく、「どんなシステムで何が困っていたのか」を描写すること。
既存手法の限界と、なぜ新しいアプローチが必要だったのかを明確にする。

### 🔬 手法・アプローチ
**まず1文で手法のTL;DRを書く**（例: 「一言でいえば、〇〇を△△で解決するアプローチである」）。
その後、核心的なアイデアを5〜8文で説明する。
最後に**トレードオフを明示する**（何を得て、何を犠牲にしたのか）。

## ルール
- 教科書的な硬い文体は避ける。読み物として面白い文章を心がける
- 文末を変化させること。「〜である」を3回以上連続で使わない。「〜だ」「〜した」「〜になる」「〜といえる」等を織り交ぜる
- 「詳細は原論文を参照」は本当に情報がない場合のみ。安易に使わない
- 冒頭の挨拶や末尾の締め文は不要。セクションのみを出力する
"""

# Stage 2: Architecture diagram, contributions, impact, keywords
_STAGE2_PROMPT = """\
あなたは「古い論文がなぜ今も重要なのか」を伝えるテックライターです。
以下の論文の前半解説を踏まえて、後半部分を日本語で作成してください。

## 論文情報
- タイトル: {title}
- 著者: {authors}
- 発表年: {year}
- 被引用数: {citation_count}
- 分野: {category_ja}

## アブストラクト
{abstract}
{pdf_section}
## 前半の解説（参考）
{stage1_summary}

## 出力形式
以下の4セクションをMarkdownで出力してください。

### 🏗️ アーキテクチャ図
論文の提案手法やシステム構成を表すMermaid図を1つ作成する。
制約:
- **flowchart TD** を優先する（内容上sequence diagramが明らかに適切な場合のみ別の種類を使ってよい）
- ノード数は**最大10個**に抑える。主要な要素だけを含め、細部は省略する
- **ラベルはすべて英語**にする（GitHubでの日本語レンダリング問題を回避するため）
- ノードIDとラベルを分けて書く（例: `A["Input Data"]`）
```mermaid
（ここに図）
```
図の直後に、図の読み方を2〜3文で日本語で補足する。

### 💡 主要な貢献
この論文が分野にもたらした具体的な成果や新規性。3〜5項目を箇条書きで。
各項目は「**結論を太字** — その説明」のフォーマットで書く。
例: **RNNなしで系列変換を実現** — Self-Attentionのみで構成することで、並列計算が可能になり学習速度が大幅に向上した。

### 🌍 影響と意義
この研究が後続の研究や実務に与えた影響。5〜8文。
- 被引用数{citation_count}件の背景にある理由
- この論文から派生した技術やプロダクトについて言及してよいが、**確信がある具体名のみ**記載すること。推測で製品名を挙げない
- 現在の実務でどのように活用されているか

### 📚 関連キーワード
この論文に関連するキーワードを5〜8個、箇条書きで列挙する。
各キーワードに1文の簡潔な説明を付ける。
**この論文の用語だけでなく、関連する現代の技術・概念を含める**こと（例: Raftの論文なら「etcd」「Kubernetes control plane」など）。

## ルール
- 技術的に正確であること
- Mermaid図は必ず含める。GitHubで正しく表示されるMermaid記法を使うこと
- 冒頭の挨拶や末尾の締め文は不要。セクションのみを出力する
"""

_FALLBACK_TEMPLATE = """\
### 🎓 前提知識
この論文は**{category_ja}**分野の研究です。以下のアブストラクトから主要な概念を把握できます。

### 📖 背景と動機
{abstract_short}

### 🔬 手法・アプローチ
この論文の手法の詳細は原論文を参照してください。アブストラクトに記載された内容から、{category_ja}分野における既存の課題に対する新しいアプローチを提案しています。

### 💡 主要な貢献
- **被引用数 {citation_count} 件**の高インパクト論文であり、{category_ja}分野で広く参照されている
- 原論文の詳細な貢献についてはアブストラクトおよび本文を参照

### 🌍 影響と意義
被引用数 {citation_count} 件は、この論文が{category_ja}分野で大きな影響力を持っていることを示している。後続の研究や実務に与えた具体的な影響については原論文を参照してください。

### 📚 関連キーワード
- **{category_ja}**: この論文の主要な研究分野
"""


def _format_authors(authors: list[str]) -> str:
    """Format author list, truncating if more than 5."""
    authors_str = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_str += f" 他{len(authors) - 5}名"
    return authors_str


def _build_pdf_section(pdf_text: str) -> str:
    """Build the optional PDF text section for prompts."""
    if not pdf_text:
        return "\n"
    return f"\n## 論文本文（抜粋）\n{pdf_text}\n\n"


def summarize_paper(paper: Paper, api_key: str | None) -> str:
    """Generate a structured summary using two Gemini API calls.

    Stage 1: Prerequisites + background + method
    Stage 2: Architecture diagram + contributions + impact + keywords

    If pdf_url is available, fetches the PDF text to enrich the prompts.
    Falls back to a basic summary if no API key or on failure.
    """
    if not api_key:
        logger.info("No API key, using fallback summary")
        return _fallback_summary(paper)

    authors_str = _format_authors(paper.authors)
    abstract = paper.abstract or f"(アブストラクト未登録。タイトル「{paper.title}」から内容を推測してください)"

    # Fetch PDF text if available
    pdf_text = ""
    if paper.pdf_url:
        logger.info("Fetching PDF text from: %s", paper.pdf_url)
        pdf_text = _fetch_page_text(paper.pdf_url)
        if pdf_text:
            pdf_text = pdf_text[:_PDF_TEXT_LIMIT]
            logger.info("Fetched %d chars of PDF text", len(pdf_text))
        else:
            logger.info("No text extracted from PDF URL")

    pdf_section = _build_pdf_section(pdf_text)

    fmt_args = {
        "title": paper.title,
        "authors": authors_str,
        "year": paper.year or "不明",
        "citation_count": paper.citation_count,
        "category_ja": paper.category_ja,
        "abstract": abstract,
        "pdf_section": pdf_section,
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
