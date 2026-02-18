"""Tests for paper_formatter module."""

from src.paper_fetcher import Paper
from src.paper_formatter import format_paper_pr_body


def _make_paper(**kwargs) -> Paper:
    defaults = {
        "paper_id": "2401.12345v1",
        "title": "MapReduce: Simplified Data Processing on Large Clusters",
        "abstract": "MapReduce is a programming model...",
        "authors": ["Jeffrey Dean", "Sanjay Ghemawat"],
        "year": 2024,
        "citation_count": 0,
        "url": "http://arxiv.org/abs/2401.12345v1",
        "pdf_url": "http://arxiv.org/pdf/2401.12345v1",
        "category": "distributed_systems",
        "category_ja": "大規模分散処理",
        "published": "2024-01-15T00:00:00Z",
        "categories": ["cs.DC"],
    }
    defaults.update(kwargs)
    return Paper(**defaults)


class TestFormatPaperPrBody:
    def test_contains_title_and_metadata(self):
        paper = _make_paper()
        result = format_paper_pr_body(paper, "要約テキスト", "2026-02-17")

        assert "MapReduce" in result
        assert "Jeffrey Dean" in result
        assert "2024-01-15" in result
        assert "大規模分散処理" in result
        assert "2026-02-17" in result

    def test_contains_summary(self):
        paper = _make_paper()
        result = format_paper_pr_body(paper, "これはテスト要約です", "2026-02-17")
        assert "これはテスト要約です" in result

    def test_includes_pdf_link(self):
        paper = _make_paper(pdf_url="http://arxiv.org/pdf/2401.12345v1")
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "http://arxiv.org/pdf/2401.12345v1" in result
        assert "PDF" in result

    def test_no_pdf_link_when_none(self):
        paper = _make_paper(pdf_url=None)
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "PDF" not in result

    def test_includes_arxiv_link(self):
        paper = _make_paper()
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "http://arxiv.org/abs/2401.12345v1" in result
        assert "arXiv" in result

    def test_no_citation_count_in_output(self):
        paper = _make_paper()
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "被引用数" not in result

    def test_shows_published_date(self):
        paper = _make_paper(published="2024-01-15T00:00:00Z")
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "発表日" in result
        assert "2024-01-15" in result

    def test_truncates_long_author_list(self):
        paper = _make_paper(authors=[f"Author {i}" for i in range(10)])
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "他5名" in result

    def test_unknown_published_date(self):
        paper = _make_paper(published="")
        result = format_paper_pr_body(paper, "要約", "2026-02-17")
        assert "不明" in result
