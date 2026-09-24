from __future__ import annotations

import csv
from pathlib import Path

import pytest

from learnings2agents.csv_loader import (
    AmbiguousRepositoryError,
    CsvFormatError,
    load_learnings,
    select_repository,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_learnings.csv"

FIELDNAMES = [
    "Learning",
    "Repository",
    "File",
    "Pull Request",
    "URL",
    "Created By",
    "Usage",
    "Last Used",
    "Created At",
    "Updated At",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in FIELDNAMES}, **row})


def test_load_learnings_reads_fixture():
    learnings = load_learnings(FIXTURE_CSV)
    assert len(learnings) == 8
    # Root-level learning (empty File) preserved as "".
    assert any(learning.file == "" for learning in learnings)
    # File paths are normalized (no leading "./" or "/").
    assert all(
        not learning.file.startswith("/") and not learning.file.startswith("./")
        for learning in learnings
    )
    # Usage parsed as int.
    assert all(isinstance(learning.usage, int) for learning in learnings)


def test_load_learnings_skips_blank_learning_rows(tmp_path):
    path = tmp_path / "learnings.csv"
    _write_csv(
        path,
        [
            {"Learning": "  ", "Repository": "repo", "File": "a.py"},
            {
                "Learning": "Real learning",
                "Repository": "repo",
                "File": "a.py",
                "Usage": "3",
            },
        ],
    )
    learnings = load_learnings(path)
    assert len(learnings) == 1
    assert learnings[0].text == "Real learning"
    assert learnings[0].usage == 3


def test_load_learnings_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Learning", "Repository"])
        writer.writeheader()
        writer.writerow({"Learning": "x", "Repository": "repo"})
    with pytest.raises(CsvFormatError):
        load_learnings(path)


def test_directory_property_for_nested_and_root_files():
    learnings = load_learnings(FIXTURE_CSV)
    by_file = {learning.file: learning for learning in learnings}
    assert by_file["utilities/oadp.py"].directory == "utilities"
    assert by_file["utilities/unittests/test_hco.py"].directory == "utilities/unittests"
    assert by_file[""].directory == ""


def test_select_repository_auto_detects_single_repo():
    learnings = load_learnings(FIXTURE_CSV)
    repository, filtered = select_repository(learnings, repository=None)
    assert repository == "openshift-virtualization-tests"
    assert len(filtered) == len(learnings)


def test_select_repository_explicit_match():
    learnings = load_learnings(FIXTURE_CSV)
    repository, filtered = select_repository(
        learnings, repository="openshift-virtualization-tests"
    )
    assert repository == "openshift-virtualization-tests"
    assert len(filtered) == len(learnings)


def test_select_repository_unknown_repo_raises():
    learnings = load_learnings(FIXTURE_CSV)
    with pytest.raises(ValueError):
        select_repository(learnings, repository="does-not-exist")


def test_select_repository_ambiguous_without_flag_raises(tmp_path):
    path = tmp_path / "multi_repo.csv"
    _write_csv(
        path,
        [
            {"Learning": "Learning A", "Repository": "repo-one", "File": "a.py"},
            {"Learning": "Learning B", "Repository": "repo-two", "File": "b.py"},
        ],
    )
    learnings = load_learnings(path)
    with pytest.raises(AmbiguousRepositoryError) as exc_info:
        select_repository(learnings, repository=None)
    assert exc_info.value.repositories == ["repo-one", "repo-two"]


def test_select_repository_with_flag_filters_multi_repo_csv(tmp_path):
    path = tmp_path / "multi_repo.csv"
    _write_csv(
        path,
        [
            {"Learning": "Learning A", "Repository": "repo-one", "File": "a.py"},
            {"Learning": "Learning B", "Repository": "repo-two", "File": "b.py"},
        ],
    )
    learnings = load_learnings(path)
    repository, filtered = select_repository(learnings, repository="repo-two")
    assert repository == "repo-two"
    assert len(filtered) == 1
    assert filtered[0].text == "Learning B"
