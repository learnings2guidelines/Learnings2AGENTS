"""Regression tests ensuring the Gemini API key is never printed/logged.

These exercise our own code paths (CLI arg parsing, logging, help text) with
a distinctive sentinel value standing in for a real secret. Network calls are
mocked out entirely, so these tests are fast/offline and specifically isolate
*our* code as the only thing that ever holds the key in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from learnings2agents.cli import build_arg_parser, main
from learnings2agents.llm import GeminiClient, LlmSynthesisError
from learnings2agents.models import MergePlan, SynthesizedBullet

SENTINEL_KEY = "SENTINEL-SECRET-VALUE-should-never-appear-9f3c7a"

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_learnings.csv"


def _make_target_repo(tmp_path: Path) -> Path:
    target = tmp_path / "target-repo"
    for rel in ["utilities/unittests", "utilities", "tests/network/libs"]:
        (target / rel).mkdir(parents=True, exist_ok=True)
    return target


def test_help_text_does_not_contain_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SENTINEL_KEY)
    parser = build_arg_parser()
    help_text = parser.format_help()
    assert SENTINEL_KEY not in help_text


def test_cli_run_never_logs_or_prints_the_api_key(
    tmp_path, monkeypatch, caplog, capsys
):
    target = _make_target_repo(tmp_path)

    def _boom(self, directory, learnings):
        # Deliberately does not reference `self.api_key` anywhere, mirroring
        # what a real (misbehaving) API error should never do either.
        raise LlmSynthesisError("simulated failure for test purposes")

    monkeypatch.setattr(GeminiClient, "synthesize_directory", _boom)

    with caplog.at_level(logging.DEBUG):
        exit_code = main(
            [
                "--csv",
                str(FIXTURE_CSV),
                "--target",
                str(target),
                "--gemini-api-key",
                SENTINEL_KEY,
                "--no-cache",
                "-v",
            ]
        )

    assert exit_code == 0

    captured = capsys.readouterr()
    assert SENTINEL_KEY not in captured.out
    assert SENTINEL_KEY not in captured.err
    assert all(SENTINEL_KEY not in record.getMessage() for record in caplog.records)

    # Every directory should have fallen back to heuristic mode since the
    # (mocked) LLM call always fails.
    assert "heuristic" in captured.err or any(
        "heuristic" in record.getMessage() for record in caplog.records
    )


def test_cli_merge_plan_and_dedupe_paths_never_log_or_print_the_api_key(
    tmp_path, monkeypatch, caplog, capsys
):
    """Covers the two newer LLM-mode code paths (merge-plan-into-existing-
    file, and the redundancy/dedupe re-check) that `test_cli_run_never_...`
    above doesn't exercise, since that test only triggers the regular
    from-scratch synthesis fallback."""
    target = _make_target_repo(tmp_path)
    # Give "utilities" a pre-existing AGENTS.md so the merge-plan path (not
    # the regular from-scratch path) is used for it.
    (target / "utilities" / "AGENTS.md").write_text(
        "# AGENTS.md\n\n- Existing rule.\n", encoding="utf-8"
    )

    def _fake_plan_merge(self, directory, learnings, existing_content):
        return MergePlan(new_bullets=[SynthesizedBullet(text="A new rule.")])

    def _fake_synthesize_directory(self, directory, learnings):
        return [SynthesizedBullet(text=f"Bullet for {directory or 'root'}.")]

    def _dedupe_boom(self, directory, existing_content, bullets):
        # Deliberately does not reference `self.api_key`, exactly like the
        # `synthesize_directory` mock above — a real (misbehaving) API
        # error should never do so either.
        raise LlmSynthesisError("simulated dedupe failure for test purposes")

    monkeypatch.setattr(GeminiClient, "plan_agents_md_merge", _fake_plan_merge)
    monkeypatch.setattr(
        GeminiClient, "synthesize_directory", _fake_synthesize_directory
    )
    monkeypatch.setattr(GeminiClient, "dedupe_bullets", _dedupe_boom)

    with caplog.at_level(logging.DEBUG):
        exit_code = main(
            [
                "--csv",
                str(FIXTURE_CSV),
                "--target",
                str(target),
                "--gemini-api-key",
                SENTINEL_KEY,
                "--model",
                "fake-model",
                "--no-cache",
                "-v",
            ]
        )

    assert exit_code == 0

    captured = capsys.readouterr()
    assert SENTINEL_KEY not in captured.out
    assert SENTINEL_KEY not in captured.err
    assert all(SENTINEL_KEY not in record.getMessage() for record in caplog.records)


def test_cli_root_consolidation_path_never_logs_or_prints_the_api_key(
    tmp_path, monkeypatch, caplog, capsys
):
    """Covers the single-root-AGENTS.md consolidation path (fresh target +
    Gemini key), including a failure in the consolidation call itself and
    in the final dedupe re-check, both of which fall back/continue rather
    than aborting — neither should ever surface the key either."""
    target = _make_target_repo(tmp_path)

    def _root_boom(self, groups):
        raise LlmSynthesisError("simulated root-consolidation failure")

    monkeypatch.setattr(GeminiClient, "synthesize_root", _root_boom)

    with caplog.at_level(logging.DEBUG):
        exit_code = main(
            [
                "--csv",
                str(FIXTURE_CSV),
                "--target",
                str(target),
                "--gemini-api-key",
                SENTINEL_KEY,
                "--model",
                "fake-model",
                "--no-cache",
                "-v",
            ]
        )

    assert exit_code == 0

    captured = capsys.readouterr()
    assert SENTINEL_KEY not in captured.out
    assert SENTINEL_KEY not in captured.err
    assert all(SENTINEL_KEY not in record.getMessage() for record in caplog.records)


def test_gemini_client_from_model_chain_probe_failure_never_logs_the_key(
    monkeypatch, caplog
):
    """`from_model_chain` probes each candidate model and logs a warning
    (with the caught exception) on failure — that exception is never
    constructed from `self.api_key`, so it can't echo it back either."""

    def _boom_get_client(self):
        raise RuntimeError("simulated probe failure, no key referenced")

    monkeypatch.setattr(GeminiClient, "_get_client", _boom_get_client)

    with caplog.at_level(logging.DEBUG):
        try:
            GeminiClient.from_model_chain(api_key=SENTINEL_KEY, models=["fake-model"])
        except LlmSynthesisError:
            pass

    assert all(SENTINEL_KEY not in record.getMessage() for record in caplog.records)


def test_gemini_client_repr_does_not_leak_key():
    client = GeminiClient(api_key=SENTINEL_KEY)
    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in str(client)


def test_makefile_never_interpolates_gemini_api_key_variable():
    """The Makefile must source GEMINI_API_KEY purely from the inherited
    environment (never as an explicit `$(GEMINI_API_KEY)` substitution),
    since GNU Make echoes expanded recipe commands by default and that would
    print the secret to the terminal/CI logs.
    """
    makefile = Path(__file__).parent.parent / "Makefile"
    content = makefile.read_text(encoding="utf-8")
    assert "$(GEMINI_API_KEY)" not in content
    assert "--gemini-api-key" not in content
