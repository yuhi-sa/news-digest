"""Tests for paper_fetcher module."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.paper_fetcher import (
    Paper,
    fetch_papers_for_category,
    get_todays_category,
    search_papers,
    select_paper,
)


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


class TestSearchPapers:
    def test_parses_api_response(self):
        mock_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "MapReduce: Simplified Data Processing",
                    "abstract": "MapReduce is a programming model...",
                    "authors": [{"name": "Jeffrey Dean"}, {"name": "Sanjay Ghemawat"}],
                    "year": 2004,
                    "citationCount": 25000,
                    "url": "https://www.semanticscholar.org/paper/abc123",
                    "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_papers("MapReduce distributed computing")

        assert len(papers) == 1
        assert papers[0].paper_id == "abc123"
        assert papers[0].title == "MapReduce: Simplified Data Processing"
        assert papers[0].citation_count == 25000
        assert papers[0].pdf_url == "https://example.com/paper.pdf"
        assert len(papers[0].authors) == 2

    def test_filters_below_min_citations(self):
        mock_data = {
            "data": [
                {
                    "paperId": "low1",
                    "title": "Low Impact Paper",
                    "abstract": "Some abstract",
                    "authors": [],
                    "year": 2020,
                    "citationCount": 50,
                    "url": "https://example.com",
                    "openAccessPdf": None,
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_papers("test query", min_citations=200)

        assert len(papers) == 0

    def test_includes_papers_without_abstract(self):
        mock_data = {
            "data": [
                {
                    "paperId": "no_abs",
                    "title": "No Abstract Paper",
                    "abstract": None,
                    "authors": [],
                    "year": 2020,
                    "citationCount": 1000,
                    "url": "https://example.com",
                    "openAccessPdf": None,
                },
            ],
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = search_papers("test query")

        assert len(papers) == 1
        assert papers[0].abstract == ""

    def test_handles_api_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            papers = search_papers("test query")

        assert papers == []


class TestSelectPaper:
    def test_selects_from_unseen(self):
        papers = [
            Paper("p1", "Paper 1", "abs", ["A"], 2020, 5000, "", None, "ai", "AI"),
            Paper("p2", "Paper 2", "abs", ["B"], 2019, 3000, "", None, "ai", "AI"),
            Paper("p3", "Paper 3", "abs", ["C"], 2018, 1000, "", None, "ai", "AI"),
        ]
        seen = {"p1"}
        result = select_paper(papers, seen)
        assert result is not None
        assert result.paper_id in {"p2", "p3"}

    def test_returns_none_when_all_seen(self):
        papers = [
            Paper("p1", "Paper 1", "abs", ["A"], 2020, 5000, "", None, "ai", "AI"),
            Paper("p2", "Paper 2", "abs", ["B"], 2019, 3000, "", None, "ai", "AI"),
        ]
        seen = {"p1", "p2"}
        result = select_paper(papers, seen)
        assert result is None

    def test_selects_from_pool_when_none_seen(self):
        papers = [
            Paper(f"p{i}", f"Paper {i}", "abs", ["A"], 2020, 5000 - i, "", None, "ai", "AI")
            for i in range(20)
        ]
        result = select_paper(papers, set())
        assert result is not None
        # Should select from top-10 by default
        assert result.paper_id in {f"p{i}" for i in range(10)}

    def test_top_k_limits_pool(self):
        papers = [
            Paper("p1", "Paper 1", "abs", ["A"], 2020, 5000, "", None, "ai", "AI"),
            Paper("p2", "Paper 2", "abs", ["B"], 2019, 3000, "", None, "ai", "AI"),
            Paper("p3", "Paper 3", "abs", ["C"], 2018, 1000, "", None, "ai", "AI"),
        ]
        # top_k=1 forces selecting the highest-cited unseen
        result = select_paper(papers, set(), top_k=1)
        assert result is not None
        assert result.paper_id == "p1"
