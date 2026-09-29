"""Render synthesized bullets into Markdown and write/merge them into
`AGENTS.md` files under the target repository, one file per directory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from learnings2agents.config import AGENTS_FILENAME, BEGIN_MARKER, END_MARKER
from learnings2agents.models import DirGroup, MergePlan, SynthesizedBullet

logger = logging.getLogger(__name__)

_SECTION_TITLE = "## Learnings from CodeRabbit"
_NEW_FILE_INTRO = (
    "# AGENTS.md\n\n"
    "Guidance derived from CodeRabbit review history for this directory.\n\n"
)


def _format_prs(pull_requests: list[str]) -> str:
    if not pull_requests:
        return ""
    label = "PR" if len(pull_requests) == 1 else "PRs"
    refs = ", ".join(f"#{pr}" for pr in pull_requests)
    return f" ({label} {refs})"


def render_bullets(bullets: list[SynthesizedBullet]) -> str:
    """Render a list of bullets (optionally grouped under headings) as Markdown."""
    has_headings = any(b.heading for b in bullets)
    if not has_headings:
        lines = [f"- {b.text}{_format_prs(b.pull_requests)}" for b in bullets]
        return "\n".join(lines)

    lines: list[str] = []
    seen_headings: list[str] = []
    grouped: dict[str, list[SynthesizedBullet]] = {}
    for bullet in bullets:
        key = bullet.heading or "General"
        if key not in grouped:
            grouped[key] = []
            seen_headings.append(key)
        grouped[key].append(bullet)

    for heading in seen_headings:
        lines.append(f"### {heading}")
        for bullet in grouped[heading]:
            lines.append(f"- {bullet.text}{_format_prs(bullet.pull_requests)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def render_section(bullets: list[SynthesizedBullet]) -> str:
    """Render the full marker-wrapped generated section for one directory."""
    body = render_bullets(bullets)
    return f"{BEGIN_MARKER}\n{_SECTION_TITLE}\n\n{body}\n{END_MARKER}"


def merge_into_existing(existing_content: str, generated_section: str) -> str:
    """Splice `generated_section` into `existing_content`, preserving hand-written text.

    - If both markers are present, the text between (and including) them is replaced.
    - If the file has content but no markers, the section is appended at the end.
    - If there's no existing content, the caller should use a fresh file (see write_agents_md).
    """
    begin_idx = existing_content.find(BEGIN_MARKER)
    end_idx = existing_content.find(END_MARKER)

    if begin_idx != -1 and end_idx != -1 and end_idx > begin_idx:
        end_idx_full = end_idx + len(END_MARKER)
        return (
            existing_content[:begin_idx]
            + generated_section
            + existing_content[end_idx_full:]
        )

    existing = existing_content.rstrip("\n")
    if not existing:
        return generated_section + "\n"
    return existing + "\n\n" + generated_section + "\n"


_HEADING_LINE_RE = re.compile(r"^### (.+)$")
_BULLET_LINE_RE = re.compile(
    r"^- (?P<text>.*?)(?: \((?:PR|PRs) (?P<prs>#\S+(?:, #\S+)*)\))?$"
)


def parse_generated_bullets(content: str) -> list[SynthesizedBullet]:
    """Find the BEGIN/END marker section in `content` and parse the bullets
    between them back into `SynthesizedBullet`s (the inverse of
    `render_bullets`/`render_section`, so it round-trips exactly).

    Returns `[]` if `content` has no well-formed marker section (missing
    markers, or END before BEGIN).
    """
    begin_idx = content.find(BEGIN_MARKER)
    end_idx = content.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1 or end_idx <= begin_idx:
        return []

    section = content[begin_idx + len(BEGIN_MARKER) : end_idx]

    bullets: list[SynthesizedBullet] = []
    heading = ""
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line == _SECTION_TITLE:
            continue

        heading_match = _HEADING_LINE_RE.match(line)
        if heading_match:
            heading = heading_match.group(1).strip()
            continue

        bullet_match = _BULLET_LINE_RE.match(line)
        if not bullet_match:
            continue
        text = bullet_match.group("text").strip()
        if not text:
            continue
        prs_raw = bullet_match.group("prs")
        pull_requests = (
            [pr.strip().lstrip("#") for pr in prs_raw.split(",")] if prs_raw else []
        )
        bullets.append(
            SynthesizedBullet(text=text, pull_requests=pull_requests, heading=heading)
        )

    return bullets


def apply_merge_plan(existing_content: str, plan: MergePlan) -> str:
    """Apply an LLM-proposed `MergePlan` to `existing_content`.

    First applies `plan.edits` as exact string replacements anywhere in the
    file (matched rules, whether hand-written or previously generated, are
    rewritten in place). An edit whose `old_text` isn't found exactly once in
    the current content (not found at all, or ambiguous/multiple matches) is
    skipped with a warning rather than applied incorrectly — this keeps a
    single bad match from corrupting the file or failing the whole directory.

    Then appends `plan.new_bullets` into the marker block, first folding in
    whatever bullets are *already* in that block (via `parse_generated_bullets`)
    so previously-accumulated bullets are never dropped, only added to.
    """
    content = existing_content
    for edit in plan.edits:
        count = content.count(edit.old_text)
        if count != 1:
            logger.warning(
                "Skipping merge edit: old_text matched %d time(s) (expected "
                "exactly 1): %r",
                count,
                edit.old_text[:120],
            )
            continue
        content = content.replace(edit.old_text, edit.new_text, 1)

    if plan.new_bullets:
        combined = parse_generated_bullets(content) + plan.new_bullets
        content = merge_into_existing(content, render_section(combined))

    return content


@dataclass
class WriteResult:
    path: Path
    action: str  # "created" | "updated" | "merged" | "skipped-empty" | "dry-run-*"


def write_agents_md(
    target_dir: Path, group: DirGroup, dry_run: bool = False
) -> WriteResult:
    """Write/merge the AGENTS.md file for one directory's synthesized bullets."""
    agents_path = target_dir / AGENTS_FILENAME

    if group.merged_content is not None:
        # Final content already computed by the LLM merge-plan path (see
        # synthesize.py + apply_merge_plan) — write it verbatim instead of
        # rendering `group.bullets` (which is left empty for this path).
        if dry_run:
            return WriteResult(path=agents_path, action="dry-run-merged")
        target_dir.mkdir(parents=True, exist_ok=True)
        agents_path.write_text(group.merged_content, encoding="utf-8")
        logger.info("merged %s", agents_path)
        return WriteResult(path=agents_path, action="merged")

    if not group.bullets:
        return WriteResult(path=agents_path, action="skipped-empty")

    generated_section = render_section(group.bullets)

    if agents_path.is_file():
        existing_content = agents_path.read_text(encoding="utf-8")
        new_content = merge_into_existing(existing_content, generated_section)
        action = "updated"
    else:
        new_content = _NEW_FILE_INTRO + generated_section + "\n"
        action = "created"

    if dry_run:
        return WriteResult(path=agents_path, action=f"dry-run-{action}")

    target_dir.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(new_content, encoding="utf-8")
    logger.info("%s %s", action, agents_path)
    return WriteResult(path=agents_path, action=action)


def write_all(
    groups: list[DirGroup], target: Path | str, dry_run: bool = False
) -> list[WriteResult]:
    """Write/merge AGENTS.md for every group with a non-empty bullet list."""
    target_path = Path(target)
    results = []
    for group in groups:
        dir_path = target_path / group.path if group.path else target_path
        results.append(write_agents_md(dir_path, group, dry_run=dry_run))
    return results
