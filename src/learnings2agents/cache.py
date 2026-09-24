"""Content-hash cache for per-directory synthesis results.

Keyed by a hash of the directory's raw learning texts (+ mode + model), so
re-running the tool against an unchanged CSV/target skips redundant LLM
calls (and skips redundant heuristic work, for speed on large CSVs).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from learnings2agents.models import Learning, SynthesizedBullet

logger = logging.getLogger(__name__)


def _cache_key(path: str, learnings: list[Learning], mode: str, model: str) -> str:
    payload = {
        "path": path,
        "mode": mode,
        "model": model if mode == "llm" else "",
        "texts": sorted(f"{learning.file}::{learning.text}" for learning in learnings),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class SynthesisCache:
    """A simple on-disk JSON cache, one file per directory-synthesis result."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(
        self, path: str, learnings: list[Learning], mode: str, model: str
    ) -> list[SynthesizedBullet] | None:
        key = _cache_key(path, learnings, mode, model)
        cache_file = self._path_for(key)
        if not cache_file.is_file():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [
                SynthesizedBullet(
                    text=item["text"],
                    pull_requests=list(item.get("pull_requests", [])),
                    heading=item.get("heading", ""),
                )
                for item in data
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Ignoring unreadable cache entry %s: %s", cache_file, exc)
            return None

    def set(
        self,
        path: str,
        learnings: list[Learning],
        mode: str,
        model: str,
        bullets: list[SynthesizedBullet],
    ) -> None:
        key = _cache_key(path, learnings, mode, model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = [
            {"text": b.text, "pull_requests": b.pull_requests, "heading": b.heading}
            for b in bullets
        ]
        try:
            self._path_for(key).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write cache entry: %s", exc)
