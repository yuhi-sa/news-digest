"""Tests for paper_fetcher module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.paper_fetcher import (
    Paper,
    _extract_abstract_from_description,
    _extract_paper_id_from_link,
    _parse_rss_response,
    enrich_paper,
    fetch_paper_metadata,
    fetch_papers_for_category,
    fetch_rss,
    get_todays_category,
    search_arxiv,
    select_paper,
)

# Sample arXiv Atom XML response for testing (API format)
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

# Sample arXiv RSS 2.0 response for testing
_SAMPLE_RSS_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0">
<channel>
  <title>cs.DC updates on arXiv.org</title>
  <item>
    <title>Distributed Order Recording Techniques</title>
    <link>https://arxiv.org/abs/2602.15995</link>
    <description>arXiv:2602.15995v1 Announce Type: new
Abstract: We propose a new technique for recording distributed order.</description>
    <guid isPermaLink="false">oai:arXiv.org:2602.15995v1</guid>
    <category>cs.DC</category>
    <pubDate>Thu, 19 Feb 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Alice Smith, Bob Jones, Charlie Brown</dc:creator>
  </item>
  <item>
    <title>Updated: Old Techniques Revisited</title>
    <link>https://arxiv.org/abs/2601.00001</link>
    <description>arXiv:2601.00001v2 Announce Type: replace
Abstract: This is an updated version.</description>
    <guid isPermaLink="false">oai:arXiv.org:2601.00001v2</guid>
    <category>cs.DC</category>
    <pubDate>Thu, 19 Feb 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>replace</arxiv:announce_type>
    <dc:creator>Dan Wilson</dc:creator>
  </item>
  <item>
    <title>Another New Paper on Consensus</title>
    <link>https://arxiv.org/abs/2602.16000</link>
    <description>arXiv:2602.16000v1 Announce Type: new
Abstract: A novel consensus protocol for distributed systems.</description>
    <guid isPermaLink="false">oai:arXiv.org:2602.16000v1</guid>
    <category>cs.DC</category>
    <category>cs.CR</category>
    <pubDate>Wed, 18 Feb 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Eve Adams</dc:creator>
  </item>
</channel>
</rss>
"""

_EMPTY_RSS_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<rss version="2.0">
<channel>
  <title>cs.DC updates on arXiv.org</title>
</channel>
</rss>
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


class TestParseRssResponse:
    def test_parses_new_papers(self):
        """Should parse 'new' announce type papers and skip 'replace' ones."""
        papers = _parse_rss_response(_SAMPLE_RSS_XML)
        # Should have 2 papers (the 'replace' one is filtered out)
        assert len(papers) == 2

    def test_paper_fields(self):
        papers = _parse_rss_response(_SAMPLE_RSS_XML)
        p = papers[0]
        assert p.paper_id == "2602.15995"
        assert p.title == "Distributed Order Recording Techniques"
        assert "recording distributed order" in p.abstract
        assert p.authors == ["Alice Smith", "Bob Jones", "Charlie Brown"]
        assert p.url == "https://arxiv.org/abs/2602.15995"
        assert p.pdf_url == "https://arxiv.org/pdf/2602.15995"
        assert "cs.DC" in p.categories
        assert p.year == 2026

    def test_skips_replace_announcements(self):
        papers = _parse_rss_response(_SAMPLE_RSS_XML)
        ids = [p.paper_id for p in papers]
        assert "2601.00001" not in ids  # This was 'replace' type

    def test_empty_rss(self):
        papers = _parse_rss_response(_EMPTY_RSS_XML)
        assert papers == []

    def test_multiple_categories(self):
        papers = _parse_rss_response(_SAMPLE_RSS_XML)
        # The second new paper has both cs.DC and cs.CR
        p = papers[1]
        assert "cs.DC" in p.categories
        assert "cs.CR" in p.categories


class TestExtractHelpers:
    def test_extract_abstract(self):
        desc = "arXiv:2602.15995v1 Announce Type: new\nAbstract: Some abstract text here."
        assert _extract_abstract_from_description(desc) == "Some abstract text here."

    def test_extract_abstract_no_prefix(self):
        desc = "Just a plain description without abstract marker"
        result = _extract_abstract_from_description(desc)
        assert result == desc

    def test_extract_paper_id(self):
        assert _extract_paper_id_from_link("https://arxiv.org/abs/2602.15995") == "2602.15995"

    def test_extract_paper_id_with_version(self):
        assert _extract_paper_id_from_link("https://arxiv.org/abs/2602.15995v1") == "2602.15995v1"


class TestFetchRss:
    def test_fetches_and_parses_rss(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_RSS_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = fetch_rss(["cs.DC"])

        assert len(papers) == 2
        assert papers[0].paper_id == "2602.15995"

    def test_deduplicates_across_feeds(self):
        """When fetching multiple RSS codes, papers appearing in both should be deduped."""
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_RSS_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            # Fetching same feed twice should still deduplicate
            papers = fetch_rss(["cs.DC", "cs.DC"])

        assert len(papers) == 2  # Not 4

    def test_handles_rss_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            papers = fetch_rss(["cs.DC"])

        assert papers == []


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

    def test_default_max_retries_is_3(self):
        """API is now a fallback, so max_retries defaults to 3."""
        import inspect
        sig = inspect.signature(search_arxiv)
        assert sig.parameters["max_retries"].default == 3


class TestFetchPaperMetadata:
    def test_fetches_single_paper(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_ARXIV_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            paper = fetch_paper_metadata("2401.12345v1")

        assert paper is not None
        assert paper.paper_id == "2401.12345v1"
        assert paper.title == "A Novel Distributed Consensus Algorithm"

    def test_returns_none_on_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            paper = fetch_paper_metadata("2401.12345v1")

        assert paper is None

    def test_returns_none_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _EMPTY_ARXIV_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            paper = fetch_paper_metadata("nonexistent")

        assert paper is None


class TestEnrichPaper:
    def _make_paper(self, abstract="Short abstract"):
        return Paper(
            paper_id="2401.12345v1",
            title="Test Paper",
            abstract=abstract,
            authors=["Alice"],
            year=2024,
            citation_count=0,
            url="https://arxiv.org/abs/2401.12345v1",
            pdf_url=None,
            category="ai",
            category_ja="AI",
            published="2024-01-15T00:00:00Z",
            categories=["cs.AI"],
        )

    def test_enriches_with_longer_abstract(self):
        paper = self._make_paper("Short")
        full_paper = self._make_paper("This is a much longer and more detailed abstract text")
        full_paper.authors = ["Alice", "Bob", "Charlie"]
        full_paper.pdf_url = "http://arxiv.org/pdf/2401.12345v1"

        with patch("src.paper_fetcher.fetch_paper_metadata", return_value=full_paper):
            result = enrich_paper(paper)

        assert result.abstract == "This is a much longer and more detailed abstract text"
        assert result.authors == ["Alice", "Bob", "Charlie"]
        assert result.pdf_url == "http://arxiv.org/pdf/2401.12345v1"

    def test_keeps_original_on_failure(self):
        paper = self._make_paper("Original abstract")

        with patch("src.paper_fetcher.fetch_paper_metadata", return_value=None):
            result = enrich_paper(paper)

        assert result.abstract == "Original abstract"

    def test_keeps_original_when_api_abstract_shorter(self):
        paper = self._make_paper("This is already a long and detailed abstract")
        full_paper = self._make_paper("Short")

        with patch("src.paper_fetcher.fetch_paper_metadata", return_value=full_paper):
            result = enrich_paper(paper)

        assert result.abstract == "This is already a long and detailed abstract"


class TestFetchPapersForCategory:
    def test_rss_primary_returns_papers(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_RSS_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = fetch_papers_for_category("distributed_systems")

        assert len(papers) == 2
        for p in papers:
            assert p.category == "distributed_systems"
            assert p.category_ja == "大規模分散処理"

    def test_falls_back_to_api_when_rss_empty(self):
        empty_rss_response = MagicMock()
        empty_rss_response.read.return_value = _EMPTY_RSS_XML.encode("utf-8")
        empty_rss_response.__enter__ = lambda s: s
        empty_rss_response.__exit__ = MagicMock(return_value=False)

        api_response = MagicMock()
        api_response.read.return_value = _SAMPLE_ARXIV_XML.encode("utf-8")
        api_response.__enter__ = lambda s: s
        api_response.__exit__ = MagicMock(return_value=False)

        def urlopen_side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            if "rss.arxiv.org" in url:
                return empty_rss_response
            return api_response

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            papers = fetch_papers_for_category("distributed_systems")

        assert len(papers) == 1
        assert papers[0].paper_id == "2401.12345v1"
        assert papers[0].category == "distributed_systems"
        assert papers[0].category_ja == "大規模分散処理"

    def test_sets_category_fields_on_rss_papers(self):
        mock_response = MagicMock()
        mock_response.read.return_value = _SAMPLE_RSS_XML.encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            papers = fetch_papers_for_category("security")

        for p in papers:
            assert p.category == "security"
            assert p.category_ja == "セキュリティ"

    def test_returns_empty_when_both_fail(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            papers = fetch_papers_for_category("distributed_systems")

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
