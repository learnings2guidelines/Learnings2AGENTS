"""Per-directory bullet synthesis: turn each DirGroup's raw learnings into a
final list of `SynthesizedBullet`s, using Gemini when available and falling
back to the offline heuristic otherwise.

When a target repository path is given and a directory's AGENTS.md already
exists, Gemini is instead asked to plan a small patch (see
`llm.GeminiClient.plan_agents_md_merge` / `writer.apply_merge_plan`) that
merges new learnings into the existing file in place — rewriting matched
rules wherever they already live in the file (hand-written or previously
generated) and appending only genuinely-new bullets — rather than
regenerating just the machine-generated section from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path

from learnings2agents.cache import SynthesisCache
from learnings2agents.config import AGENTS_FILENAME
from learnings2agents.heuristics import synthesize_heuristic
from learnings2agents.llm import GeminiClient, LlmSynthesisError
from learnings2agents.models import DirGroup
from learnings2agents.writer import apply_merge_plan

logger = logging.getLogger(__name__)


def synthesize_all(
    groups: list[DirGroup],
    gemini_client: GeminiClient | None,
    cache: SynthesisCache | None = None,
    target: Path | str | None = None,
) -> None:
    """Fill in `bullets`/`merged_content` and `mode_used` for every group, in place.

    If `gemini_client` is given, LLM mode is attempted first per directory;
    on any failure (network/auth/parsing) it logs a warning and falls back to
    heuristic mode for that directory only, rather than aborting the run.

    If `gemini_client` and `target` are both given and a directory's
    AGENTS.md already exists under `target`, a merge plan is requested
    instead of synthesizing from scratch (see module docstring); on any
    failure it falls back to the regular LLM/heuristic bullets path for that
    directory, exactly as if no existing file had been found. `target=None`
    (e.g. existing callers/tests that don't pass it) makes this whole branch
    a no-op, preserving prior behavior exactly.

    Logs an INFO progress line after each directory finishes (cache hit, LLM,
    merge, or heuristic), so a long run isn't silent between the noisy
    per-request library logs (httpx, google-genai) — e.g.:
        [12/45 dirs, 87/233 learnings] 'tests/network/libs' -> 6 bullet(s) (llm)
    """
    model_name = gemini_client.model if gemini_client else ""

    total_dirs = len(groups)
    total_learnings = sum(len(group.learnings) for group in groups)
    learnings_done = 0

    logger.info(
        "Synthesizing %d director%s covering %d learning(s) total…",
        total_dirs,
        "y" if total_dirs == 1 else "ies",
        total_learnings,
    )

    for dir_index, group in enumerate(groups, start=1):
        mode = "llm" if gemini_client else "heuristic"

        # --- Merge-into-existing-file path: only when there's an existing
        # AGENTS.md to merge into, and only in LLM mode. ---
        existing_content = None
        if gemini_client is not None and target is not None:
            dir_path = Path(target) / group.path if group.path else Path(target)
            agents_path = dir_path / AGENTS_FILENAME
            if agents_path.is_file():
                existing_content = agents_path.read_text(encoding="utf-8")

        if existing_content is not None:
            plan = None
            merge_cache_hit = False
            if cache is not None:
                plan = cache.get_merge_plan(
                    group.path, group.learnings, existing_content, model_name
                )
                merge_cache_hit = plan is not None

            if plan is None:
                try:
                    plan = gemini_client.plan_agents_md_merge(
                        group.path, group.learnings, existing_content
                    )
                    if cache is not None:
                        cache.set_merge_plan(
                            group.path,
                            group.learnings,
                            existing_content,
                            model_name,
                            plan,
                        )
                except LlmSynthesisError as exc:
                    logger.warning(
                        "Merge planning failed for '%s', falling back to "
                        "regular synthesis: %s",
                        group.display_path,
                        exc,
                    )
                    plan = None

            if plan is not None:
                group.merged_content = apply_merge_plan(existing_content, plan)
                group.mode_used = "llm-merge"

                learnings_done += len(group.learnings)
                logger.info(
                    "[%d/%d dirs, %d/%d learnings] '%s' -> merged into "
                    "existing AGENTS.md (%d edit(s), %d new bullet(s)%s)",
                    dir_index,
                    total_dirs,
                    learnings_done,
                    total_learnings,
                    group.display_path,
                    len(plan.edits),
                    len(plan.new_bullets),
                    ", cached" if merge_cache_hit else "",
                )
                continue
            # else: merge planning failed — fall through to the regular
            # bullets-from-scratch path below, exactly as if there had been
            # no existing file.

        # --- Regular path: no existing AGENTS.md (or merge planning
        # failed) — synthesize bullets from scratch, LLM first then
        # heuristic fallback, exactly as before this feature was added. ---
        cache_hit = False

        if cache is not None:
            cached = cache.get(group.path, group.learnings, mode, model_name)
            if cached is not None:
                group.bullets = cached
                group.mode_used = mode
                cache_hit = True

        if not cache_hit:
            bullets = None
            if gemini_client is not None:
                try:
                    bullets = gemini_client.synthesize_directory(
                        group.path, group.learnings
                    )
                    mode = "llm"
                except LlmSynthesisError as exc:
                    logger.warning(
                        "LLM synthesis failed for '%s', falling back to heuristic mode: %s",
                        group.display_path,
                        exc,
                    )
                    bullets = None
                    mode = "heuristic"

            if bullets is None:
                bullets = synthesize_heuristic(group.learnings)
                mode = "heuristic"

            group.bullets = bullets
            group.mode_used = mode

            if cache is not None:
                cache.set(group.path, group.learnings, mode, model_name, bullets)

        learnings_done += len(group.learnings)
        logger.info(
            "[%d/%d dirs, %d/%d learnings] '%s' -> %d bullet(s) (%s mode%s)",
            dir_index,
            total_dirs,
            learnings_done,
            total_learnings,
            group.display_path,
            len(group.bullets),
            mode,
            ", cached" if cache_hit else "",
        )
