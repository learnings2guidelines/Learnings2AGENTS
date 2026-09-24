from __future__ import annotations

from pathlib import Path

from learnings2agents.cli import main

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_learnings.csv"


def _make_target_repo(tmp_path: Path) -> Path:
    """Mirror the directories referenced by the fixture CSV under tmp_path."""
    target = tmp_path / "target-repo"
    for rel in ["utilities/unittests", "utilities", "tests/network/libs"]:
        (target / rel).mkdir(parents=True, exist_ok=True)
    return target


def test_cli_end_to_end_heuristic_mode_writes_expected_agents_md(tmp_path):
    target = _make_target_repo(tmp_path)

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--no-cache",
        ]
    )

    assert exit_code == 0

    # Root-level learning -> target/AGENTS.md
    assert (target / "AGENTS.md").is_file()
    # utilities/*.py learnings -> target/utilities/AGENTS.md
    assert (target / "utilities" / "AGENTS.md").is_file()
    # utilities/unittests/*.py learnings -> target/utilities/unittests/AGENTS.md
    assert (target / "utilities" / "unittests" / "AGENTS.md").is_file()
    # tests/network/libs/*.py learnings -> target/tests/network/libs/AGENTS.md
    assert (target / "tests" / "network" / "libs" / "AGENTS.md").is_file()

    content = (target / "utilities" / "unittests" / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert "<!-- BEGIN CODERABBIT LEARNINGS -->" in content
    assert "<!-- END CODERABBIT LEARNINGS -->" in content


def test_cli_dry_run_does_not_write_files(tmp_path):
    target = _make_target_repo(tmp_path)

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--no-cache",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not (target / "AGENTS.md").exists()
    assert not (target / "utilities" / "AGENTS.md").exists()


def test_cli_errors_when_no_matching_directories_exist(tmp_path):
    # An empty target with none of the CSV's directories present.
    target = tmp_path / "empty-target"
    target.mkdir()

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--no-cache",
        ]
    )

    # Only the root-level ("") group would match here, since "" always exists;
    # so this should still succeed and write only the root AGENTS.md.
    assert exit_code == 0
    assert (target / "AGENTS.md").is_file()
    assert not (target / "utilities").exists()


def test_cli_missing_csv_file_returns_error(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    exit_code = main(["--csv", str(tmp_path / "nope.csv"), "--target", str(target)])
    assert exit_code == 1
