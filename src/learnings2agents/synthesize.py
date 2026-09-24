"""Per-directory bullet synthesis: turn each DirGroup's raw learnings into a
final list of `SynthesizedBullet`s, using Gemini when available and falling
back to the offline heuristic otherwise.
"""

from __future__ import annotations

import logging

from learnings2agents.cache import SynthesisCache
from learnings2agents.heuristics import synthesize_heuristic
from learnings2agents.llm import GeminiClient, LlmSynthesisError
from learnings2agents.models import DirGroup

logger = logging.getLogger(__name__)


def synthesize_all(
    groups: list[DirGroup],
    gemini_client: GeminiClient | None,
    cache: SynthesisCache | None = None,
) -> None:
    """Fill in `bullets` and `mode_used` for every group, in place.

    If `gemini_client` is given, LLM mode is attempted first per directory;
    on any failure (network/auth/parsing) it logs a warning and falls back to
    heuristic mode for that directory only, rather than aborting the run.
    """
    model_name = gemini_client.model if gemini_client else ""

    for group in groups:
        mode = "llm" if gemini_client else "heuristic"

        if cache is not None:
            cached = cache.get(group.path, group.learnings, mode, model_name)
            if cached is not None:
                group.bullets = cached
                group.mode_used = mode
                logger.debug("Cache hit for '%s' (%s mode)", group.display_path, mode)
                continue

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
