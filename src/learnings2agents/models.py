"""Data models shared across the learnings2agents pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Learning:
    """A single row from a CodeRabbit "Learnings" CSV export."""

    text: str
    repository: str
    file: str  # normalized, e.g. "utilities/virt.py" or "" for repo-root learnings
    pull_request: str
    url: str
    created_by: str
    usage: int

    @property
    def directory(self) -> str:
        """Directory portion of `file`, "" for repo-root learnings."""
        if "/" not in self.file:
            return ""
        return self.file.rsplit("/", 1)[0]


@dataclass
class SynthesizedBullet:
    """One finished guideline bullet, ready to render into an AGENTS.md file."""

    text: str
    pull_requests: list[str] = field(default_factory=list)
    heading: str = ""  # optional thematic subheading, e.g. "Imports" (LLM mode only)


@dataclass
class TextEdit:
    """One exact-text replacement to apply to an existing AGENTS.md file.

    `old_text` must match a unique, verbatim substring of the file's current
    content; see `writer.apply_merge_plan` for how (and when) an edit is
    skipped instead of applied.
    """

    old_text: str
    new_text: str


@dataclass
class MergePlan:
    """The LLM's proposed changes to fold new learnings into an existing
    AGENTS.md file: exact-text edits for rules that already exist somewhere
    in the file (merged in place), plus brand-new bullets for learnings that
    match nothing existing (appended into the marker block).
    """

    edits: list[TextEdit] = field(default_factory=list)
    new_bullets: list[SynthesizedBullet] = field(default_factory=list)


@dataclass
class DirGroup:
    """All learnings whose `File` column falls directly under `path`."""

    path: str  # "" == repository root, otherwise e.g. "tests/network/libs"
    learnings: list[Learning] = field(default_factory=list)

    # Filled in by the synthesis step.
    bullets: list[SynthesizedBullet] = field(default_factory=list)
    mode_used: str = ""  # "llm", "llm-merge", or "heuristic"

    # Filled in only by the LLM "merge into existing AGENTS.md" path (see
    # synthesize.py / writer.apply_merge_plan); when set, this is the final,
    # ready-to-write file content and `bullets` is left empty.
    merged_content: str | None = None

    @property
    def display_path(self) -> str:
        return self.path if self.path else "(repository root)"
