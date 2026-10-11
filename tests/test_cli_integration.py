from __future__ import annotations

from pathlib import Path

from learnings2agents.cli import main
from learnings2agents.config import BEGIN_MARKER
from learnings2agents.llm import GeminiClient
from learnings2agents.models import MergePlan, SynthesizedBullet, TextEdit
from learnings2agents.writer import render_section

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
    # This directory had no AGENTS.md before this run, so the file is
    # created from scratch and the generated section omits the CODERABBIT
    # LEARNINGS markers — there's no previous machine-generated block that
    # a future run would need to re-find yet (see `writer.write_agents_md`).
    assert "<!-- BEGIN CODERABBIT LEARNINGS -->" not in content
    assert "<!-- END CODERABBIT LEARNINGS -->" not in content
    assert "## Learnings from CodeRabbit" in content


def test_cli_markers_are_decided_per_directory_not_for_the_whole_repo(tmp_path):
    """Whether a file's generated section gets the CODERABBIT LEARNINGS
    markers depends on whether *that directory's* AGENTS.md already
    existed — not on whether some other directory in the repository
    happens to have one."""
    target = _make_target_repo(tmp_path)
    # Pre-exists only at the root; every other directory in this run is
    # still a brand-new file.
    (target / "AGENTS.md").write_text(
        "# AGENTS.md\n\nPre-existing.\n", encoding="utf-8"
    )

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

    root_content = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN CODERABBIT LEARNINGS -->" in root_content
    assert "Pre-existing." in root_content

    utilities_content = (target / "utilities" / "unittests" / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert "<!-- BEGIN CODERABBIT LEARNINGS -->" not in utilities_content


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


def test_cli_merges_new_learnings_into_existing_agents_md(tmp_path, monkeypatch):
    """End-to-end: when --gemini-api-key is set and a directory's AGENTS.md
    already exists, the CLI should apply the LLM's merge plan (edit for a
    matching rule + a brand-new bullet) directly into that file, preserving
    hand-written content and the previously-existing bullet, instead of
    discarding them.
    """
    target = _make_target_repo(tmp_path)

    existing_bullets = [SynthesizedBullet(text="Old rule.", pull_requests=["1"])]
    existing_content = (
        "# AGENTS.md\n\nHand-written intro.\n\n"
        + render_section(existing_bullets)
        + "\n"
    )
    agents_path = target / "utilities" / "AGENTS.md"
    agents_path.write_text(existing_content, encoding="utf-8")

    def _fake_plan_merge(self, directory, learnings, existing_content):
        return MergePlan(
            edits=[
                TextEdit(
                    old_text="- Old rule. (PR #1)",
                    new_text="- Old rule, refined. (PR #1)",
                )
            ],
            new_bullets=[
                SynthesizedBullet(text="A brand-new rule.", pull_requests=["999"])
            ],
        )

    def _fake_synthesize_directory(self, directory, learnings):
        # Used for directories that don't already have an AGENTS.md (e.g.
        # the repo root, and utilities/unittests in this fixture).
        return [
            SynthesizedBullet(
                text=f"Bullet for {directory or 'root'}.", pull_requests=["1"]
            )
        ]

    def _fake_dedupe_bullets(self, directory, existing_content, bullets):
        # Identity pass-through: this test isn't exercising the redundancy
        # re-check itself (see test_synthesize.py for that), just making
        # sure the CLI's existing merge-plan behavior still works end to end
        # now that this check also runs. Avoids a real network call too.
        return bullets

    monkeypatch.setattr(GeminiClient, "plan_agents_md_merge", _fake_plan_merge)
    monkeypatch.setattr(
        GeminiClient, "synthesize_directory", _fake_synthesize_directory
    )
    monkeypatch.setattr(GeminiClient, "dedupe_bullets", _fake_dedupe_bullets)

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--gemini-api-key",
            "fake-key",
            "--model",
            "fake-model",
            "--no-cache",
        ]
    )

    assert exit_code == 0

    merged = agents_path.read_text(encoding="utf-8")
    assert "Hand-written intro." in merged
    assert "Old rule, refined." in merged
    assert "Old rule. (PR #1)" not in merged
    assert "A brand-new rule." in merged
    # The edit + new bullet were folded into the single existing marker
    # block, not appended as a second, duplicate block.
    assert merged.count(BEGIN_MARKER) == 1

    # A directory with no pre-existing AGENTS.md uses the regular
    # synthesize-from-scratch path instead.
    root_content = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "Bullet for root." in root_content


def test_cli_consolidates_into_a_single_root_agents_md_when_fresh_and_gemini_available(
    tmp_path, monkeypatch
):
    """Requirement #2: a fresh target (no AGENTS.md anywhere yet) + a Gemini
    key makes the CLI attempt to write ONE consolidated root AGENTS.md
    instead of its usual one-file-per-directory split."""
    target = _make_target_repo(tmp_path)

    root_bullets = [
        SynthesizedBullet(text="Consolidated rule one.", heading="General"),
        SynthesizedBullet(text="Consolidated rule two.", heading="Testing"),
    ]

    def _fake_synthesize_root(self, groups):
        return root_bullets

    def _fake_dedupe_bullets(self, directory, existing_content, bullets):
        return bullets

    monkeypatch.setattr(GeminiClient, "synthesize_root", _fake_synthesize_root)
    monkeypatch.setattr(GeminiClient, "dedupe_bullets", _fake_dedupe_bullets)

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--gemini-api-key",
            "fake-key",
            "--model",
            "fake-model",
            "--no-cache",
        ]
    )

    assert exit_code == 0

    content = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "Consolidated rule one." in content
    assert "Consolidated rule two." in content
    # Fresh target -> no markers in the (only) file written.
    assert "<!-- BEGIN CODERABBIT LEARNINGS -->" not in content

    # No per-directory files were written — everything landed in the root.
    assert not (target / "utilities" / "AGENTS.md").exists()
    assert not (target / "utilities" / "unittests" / "AGENTS.md").exists()
    assert not (target / "tests" / "network" / "libs" / "AGENTS.md").exists()


def test_cli_falls_back_to_per_directory_pipeline_when_root_consolidation_fails(
    tmp_path, monkeypatch
):
    """If root consolidation itself fails (LLM error), the CLI falls back to
    the regular one-AGENTS.md-per-directory pipeline rather than aborting."""
    from learnings2agents.llm import LlmSynthesisError

    target = _make_target_repo(tmp_path)

    def _fake_synthesize_root(self, groups):
        raise LlmSynthesisError("boom")

    def _fake_synthesize_directory(self, directory, learnings):
        return [SynthesizedBullet(text=f"Bullet for {directory or 'root'}.")]

    def _fake_dedupe_bullets(self, directory, existing_content, bullets):
        return bullets

    monkeypatch.setattr(GeminiClient, "synthesize_root", _fake_synthesize_root)
    monkeypatch.setattr(
        GeminiClient, "synthesize_directory", _fake_synthesize_directory
    )
    monkeypatch.setattr(GeminiClient, "dedupe_bullets", _fake_dedupe_bullets)

    exit_code = main(
        [
            "--csv",
            str(FIXTURE_CSV),
            "--target",
            str(target),
            "--gemini-api-key",
            "fake-key",
            "--model",
            "fake-model",
            "--no-cache",
        ]
    )

    assert exit_code == 0
    assert (target / "AGENTS.md").is_file()
    assert (target / "utilities" / "AGENTS.md").is_file()
    assert (target / "utilities" / "unittests" / "AGENTS.md").is_file()
    assert (target / "tests" / "network" / "libs" / "AGENTS.md").is_file()
