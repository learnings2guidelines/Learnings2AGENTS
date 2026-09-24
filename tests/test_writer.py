from __future__ import annotations

from learnings2agents.config import BEGIN_MARKER, END_MARKER
from learnings2agents.models import DirGroup, SynthesizedBullet
from learnings2agents.writer import (
    merge_into_existing,
    render_bullets,
    render_section,
    write_agents_md,
    write_all,
)


def test_render_bullets_flat_list_includes_pr_refs():
    bullets = [
        SynthesizedBullet(text="Do X.", pull_requests=["100"]),
        SynthesizedBullet(text="Do Y.", pull_requests=["200", "201"]),
    ]
    rendered = render_bullets(bullets)
    assert "- Do X. (PR #100)" in rendered
    assert "- Do Y. (PRs #200, #201)" in rendered


def test_render_bullets_groups_under_headings_when_present():
    bullets = [
        SynthesizedBullet(
            text="Import as whole module.", pull_requests=["1"], heading="Imports"
        ),
        SynthesizedBullet(
            text="Use TimeoutSampler.", pull_requests=["2"], heading="Testing"
        ),
    ]
    rendered = render_bullets(bullets)
    assert "### Imports" in rendered
    assert "### Testing" in rendered
    assert rendered.index("### Imports") < rendered.index("### Testing")


def test_render_section_wraps_in_markers():
    bullets = [SynthesizedBullet(text="Do X.", pull_requests=["1"])]
    section = render_section(bullets)
    assert section.startswith(BEGIN_MARKER)
    assert section.endswith(END_MARKER)


def test_merge_into_existing_replaces_between_markers():
    existing = (
        "# AGENTS.md\n\nHand-written intro.\n\n"
        f"{BEGIN_MARKER}\nold content\n{END_MARKER}\n\nHand-written outro.\n"
    )
    generated = f"{BEGIN_MARKER}\nnew content\n{END_MARKER}"
    merged = merge_into_existing(existing, generated)
    assert "old content" not in merged
    assert "new content" in merged
    assert "Hand-written intro." in merged
    assert "Hand-written outro." in merged


def test_merge_into_existing_appends_when_no_markers_present():
    existing = "# AGENTS.md\n\nHand-written content only.\n"
    generated = f"{BEGIN_MARKER}\nnew content\n{END_MARKER}"
    merged = merge_into_existing(existing, generated)
    assert "Hand-written content only." in merged
    assert "new content" in merged
    assert merged.index("Hand-written content only.") < merged.index("new content")


def test_write_agents_md_creates_new_file(tmp_path):
    group = DirGroup(
        path="utilities", bullets=[SynthesizedBullet(text="Do X.", pull_requests=["1"])]
    )
    result = write_agents_md(tmp_path, group)
    assert result.action == "created"
    content = result.path.read_text(encoding="utf-8")
    assert "Do X." in content
    assert BEGIN_MARKER in content


def test_write_agents_md_skips_when_no_bullets(tmp_path):
    group = DirGroup(path="utilities", bullets=[])
    result = write_agents_md(tmp_path, group)
    assert result.action == "skipped-empty"
    assert not result.path.exists()


def test_write_agents_md_preserves_hand_written_content_on_update(tmp_path):
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text("# AGENTS.md\n\nManually written notes.\n", encoding="utf-8")

    group = DirGroup(
        path="", bullets=[SynthesizedBullet(text="Do X.", pull_requests=["1"])]
    )
    result = write_agents_md(tmp_path, group)
    assert result.action == "updated"

    content = agents_path.read_text(encoding="utf-8")
    assert "Manually written notes." in content
    assert "Do X." in content


def test_write_agents_md_dry_run_does_not_touch_disk(tmp_path):
    group = DirGroup(
        path="utilities", bullets=[SynthesizedBullet(text="Do X.", pull_requests=["1"])]
    )
    result = write_agents_md(tmp_path, group, dry_run=True)
    assert result.action == "dry-run-created"
    assert not (tmp_path / "utilities" / "AGENTS.md").exists()


def test_write_all_writes_into_correct_subdirectories(tmp_path):
    (tmp_path / "utilities").mkdir()
    (tmp_path / "tests" / "network").mkdir(parents=True)

    groups = [
        DirGroup(
            path="", bullets=[SynthesizedBullet(text="Root rule.", pull_requests=["1"])]
        ),
        DirGroup(
            path="utilities",
            bullets=[SynthesizedBullet(text="Utils rule.", pull_requests=["2"])],
        ),
        DirGroup(
            path="tests/network",
            bullets=[SynthesizedBullet(text="Net rule.", pull_requests=["3"])],
        ),
    ]
    results = write_all(groups, tmp_path)

    assert (tmp_path / "AGENTS.md").is_file()
    assert (tmp_path / "utilities" / "AGENTS.md").is_file()
    assert (tmp_path / "tests" / "network" / "AGENTS.md").is_file()
    assert all(r.action == "created" for r in results)
