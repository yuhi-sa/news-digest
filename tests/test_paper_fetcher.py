"""Tests for paper_fetcher module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.paper_fetcher import (
    Paper,
    fetch_papers_for_category,
    get_todays_category,
    search_arxiv,
    select_paper,
)

# Sample arXiv Atom XML response for testing
_SAMPLE_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>  A Novel Distributed
      Consensus Algorithm  </title>
    <summary>  We propose a new consensus algorithm
      that achieves high throughput.  </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <published>2024-01-15T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345v1" type="application/pdf"/>
    <category term="cs.DC"/>
    <category term="cs.CR"/>
  </entry>
</feed>
"""

_EMPTY_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
</feed>
"""


class TestGetTodaysCategory:
    def test_rotation_covers_all_categories(self):
        """Four consecutive days should cover all 4 categories."""
        categories = set()
        for day in range(1, 5):
            dt = datetime(2026, 1, day, tzinfo=timezone.utc)
            categories.add(get_todays_category(dt))
        assert len(categories) == 4

    def test_same_day_returns_same_category(self):
        dt = datetime(2026, 2, 17, tzinfo=timezone.utc)
        cat1 = get_todays_category(dt)
        cat2 = get_todays_category(dt)
        assert cat1 == cat2

    def test_returns_valid_category(self):
        from src.paper_fetcher import CATEGORY_ORDER
        dt = datetime(2026, 6, 15, tzinfo=timezone.utc)
        cat = get_todays_category(dt)
        assert cat in CATEGORY_ORDER


class TestSearchArxiv:
    def test_parses_xml_response(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_ARXIV_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_arxiv("cat:cs.DC AND distributed")

        assert len(papers) == 1
        assert papers[0].paper_id == "2401.12345v1"
        assert papers[0].title == "A Novel Distributed Consensus Algorithm"
        assert papers[0].published == "2024-01-15T00:00:00Z"
        assert papers[0].year == 2024
        assert papers[0].citation_count == 0
        assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.12345v1"
        assert len(papers[0].authors) == 2
        assert "cs.DC" in papers[0].categories
        assert "cs.CR" in papers[0].categories

    def test_normalizes_whitespace_in_title_and_abstract(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_ARXIV_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_arxiv("test")

        assert "\n" not in papers[0].title
        assert "  " not in papers[0].title
        assert "\n" not in papers[0].abstract

    def test_handles_empty_response(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _EMPTY_ARXIV_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_arxiv("test query")

        assert papers == []

    def test_handles_api_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            papers = search_arxiv("test query", max_retries=1)

        assert papers == []


class TestSelectPaper:
    def _make_paper(self, pid, published="2024-01-01T00:00:00Z"):
        return Paper(
            pid, f"Paper {pid}", "abs", ["A"], 2024, 0, "", None,
            "ai", "AI", published=published, categories=["cs.AI"],
        )

    def test_selects_from_unseen(self):
        papers = [
            self._make_paper("p1", "2024-03-01T00:00:00Z"),
            self._make_paper("p2", "2024-02-01T00:00:00Z"),
            self._make_paper("p3", "2024-01-01T00:00:00Z"),
        ]
        seen = {"p1"}
        result = select_paper(papers, seen)
        assert result is not None
        assert result.paper_id in {"p2", "p3"}

    def test_returns_none_when_all_seen(self):
        papers = [
            self._make_paper("p1"),
            self._make_paper("p2"),
        ]
        seen = {"p1", "p2"}
        result = select_paper(papers, seen)
        assert result is None

    def test_selects_from_pool_when_none_seen(self):
        papers = [
            self._make_paper(f"p{i}", f"2024-{12-i:02d}-01T00:00:00Z")
            for i in range(20)
        ]
        result = select_paper(papers, set())
        assert result is not None
        # Should select from top-10 by default
        assert result.paper_id in {f"p{i}" for i in range(10)}

    def test_top_k_limits_pool(self):
        papers = [
            self._make_paper("p1", "2024-03-01T00:00:00Z"),
            self._make_paper("p2", "2024-02-01T00:00:00Z"),
            self._make_paper("p3", "2024-01-01T00:00:00Z"),
        ]
        # top_k=1 forces selecting the first (newest) unseen
        result = select_paper(papers, set(), top_k=1)
        assert result is not None
        assert result.paper_id == "p1"
