"""arXiv paper fetcher with RSS primary source and API fallback."""

from __future__ import annotations

import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RSS_BASE = "https://rss.arxiv.org/rss"

PAPER_CATEGORIES: dict[str, dict] = {
    "distributed_systems": {
        "name_ja": "大規模分散処理",
        "query": "cat:cs.DC",
        "rss_codes": ["cs.DC"],
    },
    "security": {
        "name_ja": "セキュリティ",
        "query": "cat:cs.CR",
        "rss_codes": ["cs.CR"],
    },
    "ai": {
        "name_ja": "AI",
        "query": "cat:cs.AI OR cat:cs.LG",
        "rss_codes": ["cs.AI", "cs.LG"],
    },
    "cloud": {
        "name_ja": "クラウド",
        "query": "cat:cs.NI OR cat:cs.SE",
        "rss_codes": ["cs.NI", "cs.SE"],
    },
}

CATEGORY_ORDER = ["distributed_systems", "security", "ai", "cloud"]

# arXiv Atom XML namespace (for API responses)
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

# Dublin Core namespace (for RSS responses)
_DC_NS = "http://purl.org/dc/elements/1.1/"


@dataclass
class Paper:
    """A research paper from arXiv."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    citation_count: int
    url: str
    pdf_url: str | None
    category: str
    category_ja: str
    published: str = ""
    categories: list[str] | None = None


def get_todays_category(date) -> str:
    """Determine today's paper category based on day of year.

    Rotates through 4 categories: distributed_systems -> security -> ai -> cloud.
    """
    day_of_year = date.timetuple().tm_yday
    index = day_of_year % 4
    return CATEGORY_ORDER[index]


# ---------------------------------------------------------------------------
# RSS feed parsing
# ---------------------------------------------------------------------------

def _extract_abstract_from_description(description: str) -> str:
    """Extract the abstract text from RSS description field.

    RSS descriptions have format: 'arXiv:XXXX.XXXXXvN Announce Type: new\nAbstract: ...'
    """
    match = re.search(r"Abstract:\s*(.+)", description, re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(description.split())


def _extract_paper_id_from_link(link: str) -> str:
    """Extract paper ID from arXiv URL like https://arxiv.org/abs/2602.15995."""
    parts = link.rsplit("/", 1)
    return parts[-1] if len(parts) > 1 else link


def _parse_rss_response(xml_text: str) -> list[Paper]:
    """Parse arXiv RSS 2.0 XML response into Paper objects."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    channel = root.find("channel")
    if channel is None:
        return papers

    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        title = " ".join(title.split())

        link = item.findtext("link", "").strip()
        paper_id = _extract_paper_id_from_link(link)

        description = item.findtext("description", "")
        abstract = _extract_abstract_from_description(description)

        # Authors from dc:creator (comma-separated)
        creator = item.findtext(f"{{{_DC_NS}}}creator", "")
        authors = [a.strip() for a in creator.split(",") if a.strip()] if creator else []

        # Publication date
        pub_date_str = item.findtext("pubDate", "")
        published = ""
        year = None
        if pub_date_str:
            try:
                pub_dt = parsedate_to_datetime(pub_date_str)
                published = pub_dt.isoformat()
                year = pub_dt.year
            except (ValueError, TypeError):
                pass

        # Categories
        categories = [
            cat.text.strip()
            for cat in item.findall("category")
            if cat.text
        ]

        # PDF URL
        pdf_url = link.replace("/abs/", "/pdf/") if "/abs/" in link else None

        # Skip replace/cross-list announcements - we only want new papers
        announce_type = item.findtext(f"{{{_ARXIV_NS}}}announce_type", "")
        if announce_type and announce_type not in ("new", ""):
            continue

        if not title or not paper_id:
            continue

        papers.append(Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            citation_count=0,
            url=link,
            pdf_url=pdf_url,
            category="",
            category_ja="",
            published=published,
            categories=categories,
        ))

    return papers


def fetch_rss(rss_codes: list[str]) -> list[Paper]:
    """Fetch papers from arXiv RSS feeds for the given category codes.

    RSS feeds are not rate-limited, unlike the arXiv API.
    """
    all_papers: list[Paper] = []
    seen_ids: set[str] = set()

    for code in rss_codes:
        url = f"{ARXIV_RSS_BASE}/{code}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NewsDigestBot/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_text = resp.read().decode("utf-8")
            papers = _parse_rss_response(xml_text)
            for p in papers:
                if p.paper_id not in seen_ids:
                    seen_ids.add(p.paper_id)
                    all_papers.append(p)
            logger.info("RSS feed %s returned %d papers", code, len(papers))
        except Exception as e:
            logger.warning("RSS feed %s failed: %s", code, e)

    return all_papers


# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------

def _parse_arxiv_response(xml_text: str) -> list[Paper]:
    """Parse arXiv Atom XML response into Paper objects."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        # paper_id from <id> tag (e.g. http://arxiv.org/abs/2401.12345v1)
        id_text = entry.findtext(f"{{{_ATOM_NS}}}id", "")
        paper_id = id_text.rsplit("/", 1)[-1] if id_text else ""

        title = entry.findtext(f"{{{_ATOM_NS}}}title", "").strip()
        # Normalize whitespace in title
        title = " ".join(title.split())

        abstract = entry.findtext(f"{{{_ATOM_NS}}}summary", "").strip()
        abstract = " ".join(abstract.split())

        authors = [
            name.text.strip()
            for author in entry.findall(f"{{{_ATOM_NS}}}author")
            if (name := author.find(f"{{{_ATOM_NS}}}name")) is not None and name.text
        ]

        published = entry.findtext(f"{{{_ATOM_NS}}}published", "")
        year = int(published[:4]) if published and len(published) >= 4 else None

        # Categories
        categories = [
            cat.get("term", "")
            for cat in entry.findall(f"{{{_ATOM_NS}}}category")
            if cat.get("term")
        ]

        # Links
        url = ""
        pdf_url = None
        for link in entry.findall(f"{{{_ATOM_NS}}}link"):
            href = link.get("href", "")
            link_type = link.get("type", "")
            rel = link.get("rel", "")
            if link_type == "application/pdf" or (rel == "related" and href.endswith(".pdf")):
                pdf_url = href
            elif rel == "alternate" or (not rel and "abs" in href):
                url = href

        if not url and id_text:
            url = id_text

        papers.append(Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            citation_count=0,
            url=url,
            pdf_url=pdf_url,
            category="",
            category_ja="",
            published=published,
            categories=categories,
        ))

    return papers


def search_arxiv(query: str, max_results: int = 20, max_retries: int = 3) -> list[Paper]:
    """Search arXiv API for papers matching the query.

    Used as fallback when RSS feeds return insufficient results.
    Retries with exponential backoff + jitter on failure.
    """
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NewsDigestBot/1.0"},
    )

    # Initial wait to respect arXiv rate limits (3s minimum between requests)
    time.sleep(3)

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_text = resp.read().decode("utf-8")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter: base * 2^attempt + random(0, base)
                base = 5
                wait = min(base * (2 ** attempt) + random.uniform(0, base), 120)
                logger.info("arXiv API failed, retrying in %.0fs (attempt %d/%d): %s", wait, attempt + 1, max_retries, e)
                time.sleep(wait)
                continue
            logger.warning("arXiv API failed for query: %s (%s)", query, e)
            return []

    papers = _parse_arxiv_response(xml_text)
    logger.info("arXiv API returned %d papers for query: %s", len(papers), query[:60])
    return papers


def fetch_paper_metadata(paper_id: str) -> Paper | None:
    """Fetch full metadata for a single paper via the arXiv API.

    Used to enrich RSS-sourced papers with full abstracts before summarization.
    This is a single targeted request, unlikely to trigger rate limiting.
    """
    query = f"id_list={paper_id}"
    url = f"{ARXIV_API}?{query}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NewsDigestBot/1.0"},
    )

    time.sleep(3)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_text = resp.read().decode("utf-8")
        papers = _parse_arxiv_response(xml_text)
        if papers:
            logger.info("Fetched full metadata for paper %s", paper_id)
            return papers[0]
        logger.warning("No metadata returned for paper %s", paper_id)
        return None
    except Exception as e:
        logger.warning("Failed to fetch metadata for paper %s: %s", paper_id, e)
        return None


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_papers_for_category(category: str) -> list[Paper]:
    """Fetch candidate papers for a given category.

    Strategy:
    1. Try RSS feeds (primary, not rate-limited)
    2. Fall back to arXiv API if RSS returns no results
    """
    cat_info = PAPER_CATEGORIES[category]
    category_ja = cat_info["name_ja"]
    rss_codes = cat_info["rss_codes"]
    query = cat_info["query"]

    # 1. Try RSS feeds (primary source)
    logger.info("Fetching RSS feeds for category %s: %s", category, rss_codes)
    papers = fetch_rss(rss_codes)

    # 2. Fall back to API if RSS returned nothing
    if not papers:
        logger.info("RSS returned no papers, falling back to arXiv API: %s", query[:80])
        papers = search_arxiv(query)

    # Set category fields on all papers
    for p in papers:
        p.category = category
        p.category_ja = category_ja

    # Sort by published date descending (newest first)
    papers.sort(key=lambda p: p.published, reverse=True)
    logger.info("Found %d papers for category %s", len(papers), category)

    return papers


def select_paper(papers: list[Paper], seen_ids: set[str], top_k: int = 10) -> Paper | None:
    """Select a random unseen paper from the top-k candidates by recency.

    Papers are already sorted by published date descending (newest first).
    Picks randomly from the top-k unseen papers to add variety.
    Returns None if all papers have been seen.
    """
    candidates = [p for p in papers if p.paper_id not in seen_ids]
    if not candidates:
        logger.warning("All candidate papers have been seen")
        return None
    pool = candidates[:top_k]
    selected = random.choice(pool)
    logger.info(
        "Selected from top-%d unseen (pool size: %d, total unseen: %d)",
        top_k, len(pool), len(candidates),
    )
    return selected


def enrich_paper(paper: Paper) -> Paper:
    """Enrich a paper with full metadata from the arXiv API.

    If the paper was sourced from RSS, its abstract may be truncated.
    This fetches the full abstract via a single targeted API call.
    Returns the original paper if enrichment fails.
    """
    full = fetch_paper_metadata(paper.paper_id)
    if full and len(full.abstract) > len(paper.abstract):
        logger.info("Enriched abstract for %s (%d -> %d chars)",
                     paper.paper_id, len(paper.abstract), len(full.abstract))
        paper.abstract = full.abstract
        if full.authors and len(full.authors) > len(paper.authors):
            paper.authors = full.authors
        if full.pdf_url and not paper.pdf_url:
            paper.pdf_url = full.pdf_url
    return paper
