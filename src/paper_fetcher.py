"""Semantic Scholar API client for fetching classic CS papers."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

PAPER_CATEGORIES: dict[str, dict] = {
    "distributed_systems": {
        "name_ja": "大規模分散処理",
        "queries": [
            "MapReduce Raft Paxos distributed consensus",
            "Spanner Dynamo distributed database",
            "GFS Spark resilient distributed computing",
        ],
    },
    "security": {
        "name_ja": "セキュリティ",
        "queries": [
            "TLS zero-knowledge proof cryptography",
            "differential privacy secure computation",
            "Byzantine fault tolerance RSA",
        ],
    },
    "ai": {
        "name_ja": "AI",
        "queries": [
            "Transformer attention BERT pre-training",
            "GPT large language model",
            "reinforcement learning generative adversarial network",
        ],
    },
    "cloud": {
        "name_ja": "クラウド",
        "queries": [
            "serverless microservices cloud architecture",
            "container orchestration Kubernetes auto-scaling",
            "service mesh distributed tracing DevOps",
        ],
    },
}

CATEGORY_ORDER = ["distributed_systems", "security", "ai", "cloud"]


@dataclass
class Paper:
    """A research paper from Semantic Scholar."""

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


def get_todays_category(date) -> str:
    """Determine today's paper category based on day of year.

    Rotates through 4 categories: distributed_systems → security → ai → cloud.
    """
    day_of_year = date.timetuple().tm_yday
    index = day_of_year % 4
    return CATEGORY_ORDER[index]


def search_papers(query: str, min_citations: int = 200, max_retries: int = 3) -> list[Paper]:
    """Search Semantic Scholar for papers matching the query.

    Returns papers with at least min_citations citations.
    Retries with exponential backoff on 429 rate limit errors.
    """
    params = urllib.parse.urlencode({
        "query": query,
        "limit": 20,
        "fields": "paperId,title,abstract,authors,year,citationCount,url,openAccessPdf",
    })
    url = f"{SEMANTIC_SCHOLAR_API}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NewsDigestBot/1.0"},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                logger.info("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            logger.warning("Semantic Scholar API failed for query: %s (%s)", query, e)
            return []
        except Exception:
            logger.warning("Semantic Scholar API failed for query: %s", query)
            return []

    papers: list[Paper] = []
    for item in data.get("data", []):
        citation_count = item.get("citationCount") or 0
        if citation_count < min_citations:
            continue

        abstract = item.get("abstract") or ""
        authors = [a.get("name", "") for a in (item.get("authors") or [])]
        pdf_info = item.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url")

        papers.append(Paper(
            paper_id=item["paperId"],
            title=item.get("title", ""),
            abstract=abstract,
            authors=authors,
            year=item.get("year"),
            citation_count=citation_count,
            url=item.get("url", f"https://www.semanticscholar.org/paper/{item['paperId']}"),
            pdf_url=pdf_url,
            category="",
            category_ja="",
        ))

    return papers


def fetch_papers_for_category(category: str) -> list[Paper]:
    """Fetch candidate papers for a given category.

    Runs all queries for the category, deduplicates by paperId,
    and sorts by citation count descending.
    """
    cat_info = PAPER_CATEGORIES[category]
    category_ja = cat_info["name_ja"]
    seen_ids: set[str] = set()
    all_papers: list[Paper] = []

    for i, query in enumerate(cat_info["queries"]):
        if i > 0:
            time.sleep(10)  # Rate limit: 10s between requests

        logger.info("Searching: %s", query)
        results = search_papers(query)

        for paper in results:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                all_papers.append(Paper(
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
                ))

    all_papers.sort(key=lambda p: p.citation_count, reverse=True)
    logger.info("Found %d unique papers for category %s", len(all_papers), category)
    return all_papers


def select_paper(papers: list[Paper], seen_ids: set[str], top_k: int = 10) -> Paper | None:
    """Select a random unseen paper from the top-k candidates by citation count.

    Papers are already sorted by citation count descending.
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
