"""CLI entrypoint: orchestrates the full CSV -> AGENTS.md pipeline.

Usage:
    python -m learnings2agents.cli --csv learnings.csv --target /path/to/repo
    (or `make run CSV=... TARGET=...` from the repository root)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from learnings2agents.cache import SynthesisCache
from learnings2agents.config import (
    DEFAULT_CACHE_DIRNAME,
    DEFAULT_GEMINI_MODEL_CHAIN,
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_MODEL_ENV_VAR,
)
from learnings2agents.csv_loader import (
    AmbiguousRepositoryError,
    CsvFormatError,
    load_learnings,
    select_repository,
)
from learnings2agents.llm import GeminiClient
from learnings2agents.synthesize import synthesize_all
from learnings2agents.tree import filter_existing_directories, group_by_directory
from learnings2agents.writer import write_all

logger = logging.getLogger("learnings2agents")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learnings2agents",
        description=(
            "Turn a CodeRabbit Learnings CSV export into scoped AGENTS.md files, "
            "written directly into a target repository's folders."
        ),
    )
    parser.add_argument(
        "--csv", required=True, help="Path to the CodeRabbit Learnings CSV export."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target repository checkout where AGENTS.md files are written.",
    )
    parser.add_argument(
        "--repository",
        default=None,
        help=(
            "Repository name to select from the CSV's Repository column. "
            "Auto-detected if the CSV contains only one repository."
        ),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get(GEMINI_API_KEY_ENV_VAR),
        help=(
            "Gemini API key for higher-quality LLM-mode synthesis. Optional; "
            f"also read from the {GEMINI_API_KEY_ENV_VAR} env var. "
            "If omitted, the tool runs in fully offline heuristic mode."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(GEMINI_MODEL_ENV_VAR),  # None → use auto-detection chain
        help=(
            "Gemini model to use in LLM mode. "
            f"Also read from the {GEMINI_MODEL_ENV_VAR} env var. "
            "When omitted, the tool probes "
            + ", ".join(DEFAULT_GEMINI_MODEL_CHAIN)
            + " in order and uses the first available one."
        ),
    )
    parser.add_argument(
        "--create-missing-dirs",
        action="store_true",
        help="Create directories referenced by the CSV that don't exist under --target.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the on-disk synthesis cache (always re-run LLM/heuristic synthesis).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Override the cache directory (default: <target>/{DEFAULT_CACHE_DIRNAME}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a summary of what would be written without touching disk.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    csv_path = Path(args.csv)
    target_path = Path(args.target)

    if not csv_path.is_file():
        logger.error("CSV file not found: %s", csv_path)
        return 1
    if not target_path.is_dir():
        logger.error("Target directory not found: %s", target_path)
        return 1

    try:
        learnings = load_learnings(csv_path)
    except CsvFormatError as exc:
        logger.error("%s", exc)
        return 1

    if not learnings:
        logger.error("No learnings found in %s", csv_path)
        return 1

    try:
        repository, learnings = select_repository(learnings, args.repository)
    except (AmbiguousRepositoryError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if repository:
        logger.info("Selected repository: %s", repository)
    logger.info("Loaded %d learning(s).", len(learnings))

    groups = group_by_directory(learnings)
    groups = filter_existing_directories(
        groups, target_path, create_missing_dirs=args.create_missing_dirs
    )
    if not groups:
        logger.error(
            "No directories from the CSV exist under target %s "
            "(pass --create-missing-dirs to create them).",
            target_path,
        )
        return 1
    logger.info(
        "Found %d director%s with learnings to process.",
        len(groups),
        "y" if len(groups) == 1 else "ies",
    )

    gemini_client = None
    if args.gemini_api_key:
        if args.model:
            # User explicitly chose a model — use it directly, no probing.
            gemini_client = GeminiClient(api_key=args.gemini_api_key, model=args.model)
            logger.info(
                "Gemini API key provided: using LLM mode (model=%s).", args.model
            )
        else:
            # Auto-detect: probe the chain and pick the first available model.
            logger.info(
                "Probing Gemini model availability (%s)…",
                " → ".join(DEFAULT_GEMINI_MODEL_CHAIN),
            )
            try:
                gemini_client = GeminiClient.from_model_chain(
                    api_key=args.gemini_api_key,
                    models=DEFAULT_GEMINI_MODEL_CHAIN,
                )
                logger.info("Using Gemini model: %s.", gemini_client.model)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "All Gemini models unavailable (%s); falling back to heuristic mode.",
                    exc,
                )
    else:
        logger.info("No Gemini API key provided: using heuristic mode.")

    cache = None
    if not args.no_cache:
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir
            else target_path / DEFAULT_CACHE_DIRNAME
        )
        cache = SynthesisCache(cache_dir)

    synthesize_all(groups, gemini_client, cache=cache)

    results = write_all(groups, target_path, dry_run=args.dry_run)

    for group, result in zip(groups, results):
        logger.info(
            "%-45s bullets=%-3d mode=%-9s -> %s (%s)",
            group.display_path,
            len(group.bullets),
            group.mode_used or "n/a",
            result.path,
            result.action,
        )

    created = sum(1 for r in results if r.action == "created")
    updated = sum(1 for r in results if r.action == "updated")
    skipped = sum(1 for r in results if r.action == "skipped-empty")

    if args.dry_run:
        would = sum(1 for r in results if r.action.startswith("dry-run"))
        logger.info(
            "[dry-run] Would write/update %d AGENTS.md file(s); %d skipped (no bullets).",
            would,
            skipped,
        )
    else:
        logger.info(
            "Done: %d file(s) created, %d file(s) updated, %d skipped (no bullets).",
            created,
            updated,
            skipped,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
