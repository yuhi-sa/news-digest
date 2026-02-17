"""Tests for paper_summarizer module."""

from unittest.mock import call, patch

from src.paper_fetcher import Paper
from src.paper_summarizer import summarize_paper


def _make_paper(**kwargs) -> Paper:
    defaults = {
        "paper_id": "abc123",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017,
        "citation_count": 100000,
        "url": "https://www.semanticscholar.org/paper/abc123",
        "pdf_url": "https://example.com/paper.pdf",
        "category": "ai",
        "category_ja": "AI",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


class TestSummarizePaper:
    def test_two_stage_gemini_calls(self):
        """Should call Gemini twice (stage 1 + stage 2) and concatenate."""
        paper = _make_paper()
        with patch("src.paper_summarizer.call_gemini", side_effect=["Stage1結果", "Stage2結果"]) as mock:
            result = summarize_paper(paper, "test-api-key")

        assert mock.call_count == 2
        assert "Stage1結果" in result
        assert "Stage2結果" in result

    def test_stage1_prompt_contains_paper_info(self):
        paper = _make_paper()
        with patch("src.paper_summarizer.call_gemini", side_effect=["s1", "s2"]) as mock:
            summarize_paper(paper, "test-api-key")

        stage1_prompt = mock.call_args_list[0][0][0]
        assert "Attention Is All You Need" in stage1_prompt
        assert "Ashish Vaswani" in stage1_prompt
        assert "2017" in stage1_prompt
        assert "前提知識" in stage1_prompt

    def test_stage2_prompt_contains_stage1_and_mermaid(self):
        paper = _make_paper()
        with patch("src.paper_summarizer.call_gemini", side_effect=["Stage1の解説", "s2"]) as mock:
            summarize_paper(paper, "test-api-key")

        stage2_prompt = mock.call_args_list[1][0][0]
        assert "Stage1の解説" in stage2_prompt
        assert "Mermaid" in stage2_prompt or "mermaid" in stage2_prompt

    def test_fallback_when_stage1_fails(self):
        paper = _make_paper()
        with patch("src.paper_summarizer.call_gemini", return_value=None):
            result = summarize_paper(paper, "test-api-key")

        assert "📖 背景と動機" in result
        assert "dominant sequence transduction" in result

    def test_returns_stage1_only_when_stage2_fails(self):
        paper = _make_paper()
        with patch("src.paper_summarizer.call_gemini", side_effect=["Stage1のみ", None]):
            result = summarize_paper(paper, "test-api-key")

        assert "Stage1のみ" in result

    def test_fallback_without_api_key(self):
        paper = _make_paper()
        result = summarize_paper(paper, None)

        assert "🎓 前提知識" in result
        assert "📖 背景と動機" in result
        assert "💡 主要な貢献" in result
        assert "100,000" in result or "100000" in result

    def test_truncates_long_author_list(self):
        paper = _make_paper(authors=[f"Author {i}" for i in range(10)])
        with patch("src.paper_summarizer.call_gemini", side_effect=["s1", "s2"]) as mock:
            summarize_paper(paper, "test-api-key")

        stage1_prompt = mock.call_args_list[0][0][0]
        assert "他5名" in stage1_prompt
