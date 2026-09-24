from __future__ import annotations

from learnings2agents.models import Learning
from learnings2agents.tree import filter_existing_directories, group_by_directory


def _learning(file: str, text: str = "Some learning", **kwargs) -> Learning:
    defaults = {
        "text": text,
        "repository": "repo",
        "file": file,
        "pull_request": "1",
        "url": "",
        "created_by": "someone",
        "usage": 1,
    }
    defaults.update(kwargs)
    return Learning(**defaults)


def test_group_by_directory_groups_and_sorts():
    learnings = [
        _learning("utilities/oadp.py"),
        _learning("tests/network/libs/bgp.py"),
        _learning("utilities/network.py"),
        _learning(""),  # repo root
    ]
    groups = group_by_directory(learnings)
    paths = [g.path for g in groups]
    assert paths == sorted(paths)
    assert "" in paths
    assert "utilities" in paths
    assert "tests/network/libs" in paths

    utilities_group = next(g for g in groups if g.path == "utilities")
    assert len(utilities_group.learnings) == 2


def test_group_by_directory_root_files_go_to_root_group():
    learnings = [_learning("README.md"), _learning("setup.py")]
    groups = group_by_directory(learnings)
    assert len(groups) == 1
    assert groups[0].path == ""
    assert len(groups[0].learnings) == 2


def test_filter_existing_directories_skips_missing_by_default(tmp_path):
    (tmp_path / "utilities").mkdir()
    learnings = [_learning("utilities/oadp.py"), _learning("missing_dir/x.py")]
    groups = group_by_directory(learnings)

    kept = filter_existing_directories(groups, tmp_path, create_missing_dirs=False)
    kept_paths = {g.path for g in kept}
    assert kept_paths == {"utilities"}


def test_filter_existing_directories_root_always_kept(tmp_path):
    learnings = [_learning("")]
    groups = group_by_directory(learnings)
    kept = filter_existing_directories(groups, tmp_path, create_missing_dirs=False)
    assert [g.path for g in kept] == [""]


def test_filter_existing_directories_creates_when_requested(tmp_path):
    learnings = [_learning("missing_dir/x.py")]
    groups = group_by_directory(learnings)

    kept = filter_existing_directories(groups, tmp_path, create_missing_dirs=True)
    assert [g.path for g in kept] == ["missing_dir"]
    assert (tmp_path / "missing_dir").is_dir()
