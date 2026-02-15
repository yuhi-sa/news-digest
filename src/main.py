"""Daily News Digest - Main entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone

from .dedup import Deduplicator
from .feeds import load_config
from .formatter import format_digest
from .parser import Article, fetch_articles
from .pr_creator import create_pr
from .summarizer import generate_briefing, get_summarizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
WEEKLY_BUFFER = PROJECT_ROOT / "data" / "weekly_articles.json"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _article_to_dict(a: Article) -> dict:
    return {
        "id": a.id,
        "title": a.title,
        "link": a.link,
        "summary": a.summary,
        "published": a.published.isoformat(),
        "source_name": a.source_name,
        "category": a.category,
        "category_ja": a.category_ja,
    }


def _dict_to_article(d: dict) -> Article:
    return Article(
        id=d["id"],
        title=d["title"],
        link=d["link"],
        summary=d["summary"],
        published=datetime.fromisoformat(d["published"]),
        source_name=d["source_name"],
        category=d["category"],
        category_ja=d["category_ja"],
    )


def _load_weekly_buffer() -> list[dict]:
    if not WEEKLY_BUFFER.exists():
        return []
    try:
        with open(WEEKLY_BUFFER, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupted weekly buffer, starting fresh")
        return []


def _save_weekly_buffer(articles: list[dict]) -> None:
    WEEKLY_BUFFER.parent.mkdir(parents=True, exist_ok=True)
    with open(WEEKLY_BUFFER, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)


def run_collect(verbose: bool = False) -> None:
    """Daily collection: fetch, dedup, summarize, save to daily digest and weekly buffer."""
    setup_logging(verbose)
    logger.info("Starting daily article collection")

    # 1. Load config
    config = load_config()
    logger.info("Loaded %d feeds from config", len(config.feeds))

    # 2. Fetch articles
    all_articles: list[Article] = []
    feed_stats: dict[str, bool] = {}

    for source in config.feeds:
        articles = fetch_articles(source, max_articles=config.max_articles_per_feed)
        feed_stats[source.name] = len(articles) > 0
        all_articles.extend(articles)

    logger.info("Fetched %d total articles from %d feeds", len(all_articles), len(config.feeds))

    if not all_articles:
        logger.error("All feeds failed, no articles fetched")
        sys.exit(1)

    # 3. Deduplicate
    dedup = Deduplicator()
    dedup.prune(window_days=config.dedup_window_days)
    new_articles = dedup.filter_new(all_articles)

    if not new_articles:
        logger.info("No new articles after dedup")
        dedup.save()
        return

    # 4. Summarize in Japanese
    api_key = os.environ.get("SUMMARIZER_API_KEY")
    summarizer = get_summarizer(api_key)
    summarized = summarizer.summarize(new_articles)

    # 5. Save daily digest file
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    digest_content = format_digest(summarized, date=now, feed_stats=feed_stats)

    digest_path = PROJECT_ROOT / "digests" / f"{date_str}.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(digest_content, encoding="utf-8")
    logger.info("Digest written to %s", digest_path)

    # 6. Append to weekly buffer
    buffer = _load_weekly_buffer()
    for a in summarized:
        buffer.append(_article_to_dict(a))
    _save_weekly_buffer(buffer)
    logger.info("Weekly buffer now has %d articles", len(buffer))

    # 7. Save dedup DB
    dedup.save()

    logger.info("Daily collection complete: %d new articles saved", len(summarized))


def run_weekly(dry_run: bool = False, verbose: bool = False) -> None:
    """Weekly digest: generate briefing from accumulated articles, create PR."""
    setup_logging(verbose)
    logger.info("Starting weekly digest PR creation")

    # 1. Load weekly buffer
    buffer = _load_weekly_buffer()
    if not buffer:
        logger.info("No articles in weekly buffer, skipping")
        return

    articles = [_dict_to_article(d) for d in buffer]
    logger.info("Loaded %d articles from weekly buffer", len(articles))

    # 2. Generate briefing
    api_key = os.environ.get("SUMMARIZER_API_KEY")
    briefing = generate_briefing(articles, api_key)
    if briefing:
        logger.info("Weekly briefing generated (%d chars)", len(briefing))
    else:
        logger.info("No briefing generated (no API key or failure)")

    # 3. Determine week range
    now = datetime.now(timezone.utc)
    dates = sorted(set(a.published.strftime("%Y-%m-%d") for a in articles))
    week_start = dates[0]
    week_end = dates[-1] if len(dates) > 1 else week_start
    week_label = f"{week_start}_{week_end}"

    # 4. Write briefing file (this is the new content for the PR branch)
    briefing_path = PROJECT_ROOT / "digests" / f"briefing-{week_label}.md"
    briefing_path.write_text(briefing or "(No briefing generated)", encoding="utf-8")
    logger.info("Briefing file written to %s", briefing_path)

    if dry_run:
        logger.info("Dry run mode - skipping PR creation")
        print(f"\n{'='*60}")
        print(f"WEEKLY BRIEFING ({week_label})")
        print(f"{'='*60}\n")
        print(briefing or "(no briefing)")
        return

    # 5. Create PR
    pr_url = create_pr(
        briefing_path=briefing_path,
        week_label=week_label,
        article_count=len(articles),
        repo_root=PROJECT_ROOT,
        briefing=briefing or "",
    )

    if pr_url:
        # 6. Clear weekly buffer (committed in PR branch, main keeps it until merged)
        _save_weekly_buffer([])
        logger.info("Weekly buffer cleared")
        logger.info("Weekly digest PR created: %s", pr_url)
    else:
        logger.warning("PR creation failed or was skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily News Digest Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # collect subcommand
    collect_parser = subparsers.add_parser("collect", help="Daily article collection")
    collect_parser.add_argument("--verbose", "-v", action="store_true")

    # weekly subcommand
    weekly_parser = subparsers.add_parser("weekly", help="Create weekly digest PR")
    weekly_parser.add_argument("--dry-run", action="store_true")
    weekly_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect(verbose=args.verbose)
    elif args.command == "weekly":
        run_weekly(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
