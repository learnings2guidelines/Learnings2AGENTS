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
