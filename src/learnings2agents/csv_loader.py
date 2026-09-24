"""Parse a CodeRabbit "Learnings" CSV export into `Learning` records.

Expected columns (as exported from CodeRabbit's Learnings page):
    Learning, Repository, File, Pull Request, URL, Created By, Usage,
    Last Used, Created At, Updated At

Only `Learning`, `Repository`, `File`, `Pull Request`, `URL`, `Created By`
and `Usage` are used by the tool; the remaining columns are ignored.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from learnings2agents.models import Learning

REQUIRED_COLUMNS = ("Learning", "Repository", "File")


class CsvFormatError(ValueError):
    """Raised when the input CSV is missing required columns."""


class AmbiguousRepositoryError(ValueError):
    """Raised when the CSV contains multiple repositories and none was selected."""

    def __init__(self, repositories: list[str]):
        self.repositories = repositories
        super().__init__(
            "CSV contains learnings from multiple repositories "
            f"({', '.join(repositories)}). Pass --repository to select one."
        )


def _normalize_file(raw_file: str) -> str:
    """Normalize a `File` cell into a forward-slash relative path with no leading slash."""
    value = (raw_file or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.lstrip("/")
    return value


def _parse_usage(raw_usage: str) -> int:
    try:
        return int(str(raw_usage).strip() or 0)
    except ValueError:
        return 0


def load_learnings(csv_path: str | Path) -> list[Learning]:
    """Parse every row of the CSV into `Learning` objects (all repositories included)."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise CsvFormatError(
                f"CSV {csv_path} is missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(fieldnames)}"
            )

        learnings: list[Learning] = []
        for row in reader:
            text = (row.get("Learning") or "").strip()
            if not text:
                continue
            learnings.append(
                Learning(
                    text=text,
                    repository=(row.get("Repository") or "").strip(),
                    file=_normalize_file(row.get("File") or ""),
                    pull_request=(row.get("Pull Request") or "").strip(),
                    url=(row.get("URL") or "").strip(),
                    created_by=(row.get("Created By") or "").strip(),
                    usage=_parse_usage(row.get("Usage") or "0"),
                )
            )
    return learnings


def select_repository(
    learnings: list[Learning], repository: str | None
) -> tuple[str | None, list[Learning]]:
    """Filter `learnings` down to a single repository.

    If `repository` is given, filters to exact matches (raises if none match).
    If not given, auto-detects when the CSV contains exactly one repository,
    otherwise raises AmbiguousRepositoryError listing the candidates.

    Returns (resolved_repository_name_or_None, filtered_learnings). The
    repository name is None only when the CSV has no Repository values at all
    (e.g. a hand-built CSV without that column populated), in which case all
    learnings are returned unfiltered.
    """
    repo_counts = Counter(
        learning.repository for learning in learnings if learning.repository
    )

    if repository:
        filtered = [
            learning for learning in learnings if learning.repository == repository
        ]
        if not filtered:
            available = ", ".join(sorted(repo_counts)) or "(none found)"
            raise ValueError(
                f"No learnings found for repository '{repository}'. "
                f"Available repositories in CSV: {available}"
            )
        return repository, filtered

    if not repo_counts:
        # Repository column empty/unused; treat every row as relevant.
        return None, learnings

    if len(repo_counts) == 1:
        (only_repo,) = repo_counts
        return only_repo, [
            learning for learning in learnings if learning.repository == only_repo
        ]

    raise AmbiguousRepositoryError(sorted(repo_counts))
