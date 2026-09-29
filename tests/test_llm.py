"""Tests for llm.py's Gemini API wrapper: prompt building for the merge-plan
path, and the merge-plan parsing/validation logic. These monkeypatch
GeminiClient's internal `_client` so no real network/API calls are made.
"""

from __future__ import annotations

import json

import pytest

from learnings2agents.llm import GeminiClient, LlmSynthesisError, _build_merge_prompt
from learnings2agents.models import Learning


def _learning(
    text: str, pull_request: str = "", file: str = "utilities/foo.py"
) -> Learning:
    return Learning(
        text=text,
        repository="acme-repo",
        file=file,
        pull_request=pull_request,
        url="",
        created_by="someone",
        usage=1,
    )


class _FakeResponse:
    def __init__(self, text: str | None):
        self.text = text


class _FakeModels:
    def __init__(self, response_text: str | None = None, error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.response_text)


class _FakeGenaiClient:
    def __init__(self, response_text: str | None = None, error: Exception | None = None):
        self.models = _FakeModels(response_text=response_text, error=error)


def _client_with_fake_backend(
    response_text: str | None = None, error: Exception | None = None
) -> GeminiClient:
    client = GeminiClient(api_key="fake-key")
    client._client = _FakeGenaiClient(response_text=response_text, error=error)
    return client


def test_build_merge_prompt_includes_full_existing_content_and_pr_tags():
    learnings = [_learning("Do X.", pull_request="42"), _learning("Do Y.")]
    prompt = _build_merge_prompt(
        "utilities", learnings, "# AGENTS.md\n\nOld content.\n"
    )

    assert "Old content." in prompt
    assert "1. [PR #42] Do X." in prompt
    assert "2. [no PR] Do Y." in prompt
    assert "Directory: utilities" in prompt


def test_build_merge_prompt_uses_root_label_for_empty_directory():
    prompt = _build_merge_prompt("", [], "content")
    assert "Directory: (repository root)" in prompt


def test_plan_agents_md_merge_parses_edits_and_new_bullets():
    payload = {
        "edits": [{"old_text": "- Old rule.", "new_text": "- Old rule, updated."}],
        "new_bullets": [{"text": "New rule.", "heading": "", "sources": [1]}],
    }
    client = _client_with_fake_backend(response_text=json.dumps(payload))

    plan = client.plan_agents_md_merge(
        "utilities", [_learning("New rule.", pull_request="7")], "- Old rule.\n"
    )

    assert len(plan.edits) == 1
    assert plan.edits[0].old_text == "- Old rule."
    assert plan.edits[0].new_text == "- Old rule, updated."
    assert len(plan.new_bullets) == 1
    assert plan.new_bullets[0].text == "New rule."
    assert plan.new_bullets[0].pull_requests == ["7"]


def test_plan_agents_md_merge_drops_malformed_edit_items():
    payload = {
        "edits": [
            {"old_text": "", "new_text": "something"},  # empty old_text
            {"old_text": "same", "new_text": "same"},  # no-op edit
            {"old_text": "real old", "new_text": "real new"},  # valid
        ],
        "new_bullets": [],
    }
    client = _client_with_fake_backend(response_text=json.dumps(payload))

    plan = client.plan_agents_md_merge("utilities", [], "real old")

    assert len(plan.edits) == 1
    assert plan.edits[0].old_text == "real old"
    assert plan.edits[0].new_text == "real new"


def test_plan_agents_md_merge_empty_plan_is_a_valid_non_error_result():
    payload = {"edits": [], "new_bullets": []}
    client = _client_with_fake_backend(response_text=json.dumps(payload))

    plan = client.plan_agents_md_merge("utilities", [_learning("Do X.")], "content")

    assert plan.edits == []
    assert plan.new_bullets == []


def test_plan_agents_md_merge_raises_on_api_error():
    client = _client_with_fake_backend(error=RuntimeError("boom"))

    with pytest.raises(LlmSynthesisError):
        client.plan_agents_md_merge("utilities", [_learning("Do X.")], "content")


def test_plan_agents_md_merge_raises_on_unparsable_json():
    client = _client_with_fake_backend(response_text="not json at all")

    with pytest.raises(LlmSynthesisError):
        client.plan_agents_md_merge("utilities", [_learning("Do X.")], "content")


def test_plan_agents_md_merge_raises_when_response_is_not_a_json_object():
    client = _client_with_fake_backend(response_text=json.dumps(["not", "a", "dict"]))

    with pytest.raises(LlmSynthesisError):
        client.plan_agents_md_merge("utilities", [_learning("Do X.")], "content")


def test_plan_agents_md_merge_falls_back_to_all_prs_when_sources_missing():
    payload = {
        "edits": [],
        "new_bullets": [{"text": "New rule.", "heading": "", "sources": []}],
    }
    client = _client_with_fake_backend(response_text=json.dumps(payload))
    learnings = [_learning("A.", pull_request="1"), _learning("B.", pull_request="2")]

    plan = client.plan_agents_md_merge("utilities", learnings, "content")

    assert set(plan.new_bullets[0].pull_requests) == {"1", "2"}
