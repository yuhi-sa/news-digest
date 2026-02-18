"""arXiv API client for fetching latest CS papers."""

from __future__ import annotations

import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

PAPER_CATEGORIES: dict[str, dict] = {
    "distributed_systems": {
        "name_ja": "大規模分散処理",
        "query": "cat:cs.DC",
    },
    "security": {
        "name_ja": "セキュリティ",
        "query": "cat:cs.CR",
    },
    "ai": {
        "name_ja": "AI",
        "query": "cat:cs.AI OR cat:cs.LG",
    },
    "cloud": {
        "name_ja": "クラウド",
        "query": "cat:cs.NI OR cat:cs.SE",
    },
}

CATEGORY_ORDER = ["distributed_systems", "security", "ai", "cloud"]

# arXiv Atom XML namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


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


def search_arxiv(query: str, max_results: int = 20, max_retries: int = 5) -> list[Paper]:
    """Search arXiv for papers matching the query.

    Returns papers sorted by submission date (newest first).
    Retries with exponential backoff on failure. arXiv recommends
    waiting at least 3 seconds between requests.
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
                wait = 15 * (attempt + 1)
                logger.info("arXiv API failed, retrying in %ds (attempt %d/%d): %s", wait, attempt + 1, max_retries, e)
                time.sleep(wait)
                continue
            logger.warning("arXiv API failed for query: %s (%s)", query, e)
            return []

    papers = _parse_arxiv_response(xml_text)
    logger.info("arXiv returned %d papers for query: %s", len(papers), query[:60])
    return papers


def fetch_papers_for_category(category: str) -> list[Paper]:
    """Fetch candidate papers for a given category.

    Sends a single query per category and sorts by published date (newest first).
    """
    cat_info = PAPER_CATEGORIES[category]
    category_ja = cat_info["name_ja"]
    query = cat_info["query"]

    logger.info("Searching arXiv: %s", query[:80])
    results = search_arxiv(query)

    papers: list[Paper] = []
    for paper in results:
        papers.append(Paper(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            year=paper.year,
            citation_count=paper.citation_count,
            url=paper.url,
            pdf_url=paper.pdf_url,
            category=category,
            category_ja=category_ja,
            published=paper.published,
            categories=paper.categories,
        ))

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
