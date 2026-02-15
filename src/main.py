"""Daily News Digest - Main entry point."""

from __future__ import annotations

import argparse
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


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(dry_run: bool = False, verbose: bool = False) -> None:
    """Execute the full digest pipeline."""
    setup_logging(verbose)
    logger.info("Starting Daily News Digest pipeline")

    # 1. Load config
    config = load_config()
    logger.info("Loaded %d feeds from config", len(config.feeds))

    # 2. Fetch articles from all feeds
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
        logger.info("No new articles after dedup, skipping digest")
        dedup.save()
        return

    # 4. Summarize
    api_key = os.environ.get("SUMMARIZER_API_KEY")
    summarizer = get_summarizer(api_key)
    summarized = summarizer.summarize(new_articles)

    # 5. Format digest
    now = datetime.now(timezone.utc)
    digest_content = format_digest(summarized, date=now, feed_stats=feed_stats)

    # 6. Save digest file
    date_str = now.strftime("%Y-%m-%d")
    digest_path = PROJECT_ROOT / "digests" / f"{date_str}.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(digest_content, encoding="utf-8")
    logger.info("Digest written to %s", digest_path)

    # Save dedup DB
    dedup.save()

    # 6b. Generate briefing for PR body
    briefing = generate_briefing(summarized, api_key)
    if briefing:
        logger.info("Briefing generated (%d chars)", len(briefing))
    else:
        logger.info("No briefing generated (no API key or failure)")

    if dry_run:
        logger.info("Dry run mode - skipping PR creation")
        print(f"\n{'='*60}")
        print(f"BRIEFING ({date_str})")
        print(f"{'='*60}\n")
        print(briefing or "(no briefing)")
        print(f"\n{'='*60}")
        print(f"DIGEST ({date_str})")
        print(f"{'='*60}\n")
        print(digest_content)
        return

    # 7. Create PR
    seen_db_path = PROJECT_ROOT / "data" / "seen_articles.json"
    pr_url = create_pr(
        digest_path=digest_path,
        seen_db_path=seen_db_path,
        date=now,
        article_count=len(summarized),
        feed_stats=feed_stats,
        repo_root=PROJECT_ROOT,
        briefing=briefing,
    )

    if pr_url:
        logger.info("Pipeline complete. PR: %s", pr_url)
    else:
        logger.warning("Pipeline complete but PR creation failed or was skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily News Digest Generator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate digest locally without creating a PR",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
