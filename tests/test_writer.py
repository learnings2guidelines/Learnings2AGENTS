from __future__ import annotations

from learnings2agents.config import BEGIN_MARKER, END_MARKER
from learnings2agents.models import DirGroup, MergePlan, SynthesizedBullet, TextEdit
from learnings2agents.writer import (
    apply_merge_plan,
    merge_into_existing,
    parse_generated_bullets,
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


def test_parse_generated_bullets_round_trips_render_section_flat():
    bullets = [
        SynthesizedBullet(text="Do X.", pull_requests=["1"]),
        SynthesizedBullet(text="Do Y.", pull_requests=["2", "3"]),
    ]
    section = render_section(bullets)
    content = f"# AGENTS.md\n\nIntro.\n\n{section}\n"

    parsed = parse_generated_bullets(content)

    assert [b.text for b in parsed] == ["Do X.", "Do Y."]
    assert parsed[0].pull_requests == ["1"]
    assert parsed[1].pull_requests == ["2", "3"]
    assert all(b.heading == "" for b in parsed)


def test_parse_generated_bullets_round_trips_headings():
    bullets = [
        SynthesizedBullet(text="Import as whole module.", heading="Imports"),
        SynthesizedBullet(text="Use TimeoutSampler.", heading="Testing"),
    ]
    section = render_section(bullets)

    parsed = parse_generated_bullets(section)

    assert [b.heading for b in parsed] == ["Imports", "Testing"]
    assert [b.text for b in parsed] == [
        "Import as whole module.",
        "Use TimeoutSampler.",
    ]


def test_parse_generated_bullets_returns_empty_when_no_markers():
    assert parse_generated_bullets("# AGENTS.md\n\nJust prose.\n") == []


def test_parse_generated_bullets_returns_empty_when_markers_reversed():
    # END before BEGIN -> malformed, treated as absent.
    content = f"{END_MARKER}\nstuff\n{BEGIN_MARKER}"
    assert parse_generated_bullets(content) == []


def test_apply_merge_plan_applies_a_clean_edit():
    existing = "# AGENTS.md\n\n- Old rule.\n"
    plan = MergePlan(edits=[TextEdit(old_text="- Old rule.", new_text="- New rule.")])

    result = apply_merge_plan(existing, plan)

    assert "- New rule." in result
    assert "- Old rule." not in result


def test_apply_merge_plan_skips_edit_when_old_text_not_found():
    existing = "# AGENTS.md\n\n- Some rule.\n"
    plan = MergePlan(
        edits=[TextEdit(old_text="- Missing rule.", new_text="- New rule.")]
    )

    result = apply_merge_plan(existing, plan)

    assert result == existing


def test_apply_merge_plan_skips_edit_when_old_text_is_ambiguous():
    existing = "# AGENTS.md\n\n- Dup.\n- Dup.\n"
    plan = MergePlan(edits=[TextEdit(old_text="- Dup.", new_text="- Changed.")])

    result = apply_merge_plan(existing, plan)

    assert result == existing


def test_apply_merge_plan_appends_new_bullets_alongside_existing_marker_bullets():
    existing_bullets = [SynthesizedBullet(text="Old bullet.", pull_requests=["1"])]
    existing = f"# AGENTS.md\n\n{render_section(existing_bullets)}\n"
    plan = MergePlan(
        new_bullets=[SynthesizedBullet(text="New bullet.", pull_requests=["2"])]
    )

    result = apply_merge_plan(existing, plan)

    assert "Old bullet." in result
    assert "New bullet." in result
    # Exactly one marker pair -> new bullets were folded into the existing
    # block, not appended as a second, duplicate block.
    assert result.count(BEGIN_MARKER) == 1
    assert result.count(END_MARKER) == 1


def test_apply_merge_plan_creates_marker_block_when_none_exists():
    existing = "# AGENTS.md\n\nHand-written notes only.\n"
    plan = MergePlan(new_bullets=[SynthesizedBullet(text="New bullet.")])

    result = apply_merge_plan(existing, plan)

    assert "Hand-written notes only." in result
    assert BEGIN_MARKER in result
    assert "New bullet." in result


def test_apply_merge_plan_combines_edits_and_new_bullets():
    existing_bullets = [SynthesizedBullet(text="Old bullet.", pull_requests=["1"])]
    existing = f"# AGENTS.md\n\n{render_section(existing_bullets)}\n"
    plan = MergePlan(
        edits=[
            TextEdit(
                old_text="- Old bullet. (PR #1)",
                new_text="- Old bullet, updated. (PR #1)",
            )
        ],
        new_bullets=[SynthesizedBullet(text="Brand new.", pull_requests=["2"])],
    )

    result = apply_merge_plan(existing, plan)

    assert "Old bullet, updated." in result
    assert "Brand new." in result
    assert result.count(BEGIN_MARKER) == 1


def test_write_agents_md_writes_merged_content_verbatim(tmp_path):
    group = DirGroup(path="utilities", merged_content="# AGENTS.md\n\nMerged text.\n")
    result = write_agents_md(tmp_path, group)

    assert result.action == "merged"
    content = result.path.read_text(encoding="utf-8")
    assert content == "# AGENTS.md\n\nMerged text.\n"


def test_write_agents_md_ignores_bullets_when_merged_content_is_set(tmp_path):
    group = DirGroup(
        path="utilities",
        bullets=[SynthesizedBullet(text="Should not appear.")],
        merged_content="# AGENTS.md\n\nOnly this.\n",
    )
    result = write_agents_md(tmp_path, group)

    content = result.path.read_text(encoding="utf-8")
    assert "Should not appear." not in content
    assert "Only this." in content


def test_write_agents_md_dry_run_merged_does_not_touch_disk(tmp_path):
    group = DirGroup(path="utilities", merged_content="content")
    result = write_agents_md(tmp_path, group, dry_run=True)

    assert result.action == "dry-run-merged"
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
