"""Paper deduplication - tracks which papers have been featured."""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "seen_papers.json"


class PaperDeduplicator:
    """Manages seen papers and prevents repeats."""

    def __init__(self, db_path: str | pathlib.Path | None = None):
        self.db_path = pathlib.Path(db_path) if db_path else DEFAULT_DB_PATH
        self._seen: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        """Load seen papers from disk."""
        if not self.db_path.exists():
            return {}
        try:
            with open(self.db_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Invalid seen_papers.json format, resetting")
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted seen_papers.json, starting fresh")
            return {}

    def save(self) -> None:
        """Persist seen papers to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._seen, f, indent=2, ensure_ascii=False)

    def prune(self, window_days: int = 30) -> None:
        """Remove entries older than the window (default 30 days)."""
        now = datetime.now(timezone.utc)
        to_remove = []
        for key, entry in self._seen.items():
            try:
                seen_at = datetime.fromisoformat(entry["seen_at"])
                if (now - seen_at).days > window_days:
                    to_remove.append(key)
            except (KeyError, ValueError):
                to_remove.append(key)
        for key in to_remove:
            del self._seen[key]
        if to_remove:
            logger.info("Pruned %d old entries from paper dedup DB", len(to_remove))

    def is_seen(self, paper_id: str) -> bool:
        """Check if a paper has been featured before."""
        return paper_id in self._seen

    def mark_seen(self, paper_id: str, title: str) -> None:
        """Mark a paper as featured."""
        self._seen[paper_id] = {
            "title": title,
            "seen_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_seen_ids(self) -> set[str]:
        """Return the set of all seen paper IDs."""
        return set(self._seen.keys())
