"""Tests for llm.py's Gemini API wrapper: prompt building for the merge-plan
path, and the merge-plan parsing/validation logic. These monkeypatch
GeminiClient's internal `_client` so no real network/API calls are made.
"""

from __future__ import annotations

import json

import pytest

from learnings2agents.llm import (
    GeminiClient,
    LlmSynthesisError,
    _build_dedupe_prompt,
    _build_merge_prompt,
    _build_root_prompt,
)
from learnings2agents.models import DirGroup, Learning, SynthesizedBullet


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
    def __init__(
        self, response_text: str | None = None, error: Exception | None = None
    ):
        self.response_text = response_text
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.response_text)


class _FakeGenaiClient:
    def __init__(
        self, response_text: str | None = None, error: Exception | None = None
    ):
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


# --- synthesize_root (requirement #2: consolidate everything into one
# root AGENTS.md when the target repo is fresh and a Gemini key exists). ---


def test_build_root_prompt_tags_each_learning_with_its_directory():
    groups = [
        DirGroup(path="utilities", learnings=[_learning("Do X.")]),
        DirGroup(path="", learnings=[_learning("Do Y.")]),
    ]
    prompt = _build_root_prompt(groups)

    assert "[dir: utilities] Do X." in prompt
    assert "[dir: (repository root)] Do Y." in prompt


def test_synthesize_root_parses_bullets_and_attributes_prs():
    groups = [
        DirGroup(path="utilities", learnings=[_learning("Do X.", pull_request="1")]),
        DirGroup(path="tests", learnings=[_learning("Do Y.", pull_request="2")]),
    ]
    payload = [
        {"text": "Do X and Y.", "heading": "General", "sources": [1, 2]},
    ]
    client = _client_with_fake_backend(response_text=json.dumps(payload))

    bullets = client.synthesize_root(groups)

    assert len(bullets) == 1
    assert bullets[0].text == "Do X and Y."
    assert bullets[0].heading == "General"
    assert set(bullets[0].pull_requests) == {"1", "2"}


def test_synthesize_root_raises_when_there_are_no_learnings_at_all():
    client = _client_with_fake_backend(response_text="[]")
    with pytest.raises(LlmSynthesisError):
        client.synthesize_root([DirGroup(path="utilities", learnings=[])])


def test_synthesize_root_raises_on_api_error():
    groups = [DirGroup(path="utilities", learnings=[_learning("Do X.")])]
    client = _client_with_fake_backend(error=RuntimeError("boom"))
    with pytest.raises(LlmSynthesisError):
        client.synthesize_root(groups)


def test_synthesize_root_raises_when_no_usable_bullets_come_back():
    groups = [DirGroup(path="utilities", learnings=[_learning("Do X.")])]
    client = _client_with_fake_backend(response_text=json.dumps([{"text": ""}]))
    with pytest.raises(LlmSynthesisError):
        client.synthesize_root(groups)


# --- dedupe_bullets (requirement #3: re-check new bullets against existing
# content and each other, dropping any that are already covered). ---


def test_build_dedupe_prompt_lists_existing_content_and_numbered_bullets():
    bullets = [
        SynthesizedBullet(text="Do X.", heading="Imports"),
        SynthesizedBullet(text="Do Y."),
    ]
    prompt = _build_dedupe_prompt("utilities", "- Old rule.\n", bullets)

    assert "- Old rule." in prompt
    assert "1. Do X. [heading: Imports]" in prompt
    assert "2. Do Y." in prompt
    assert "Directory: utilities" in prompt


def test_build_dedupe_prompt_notes_brand_new_file_when_existing_content_is_empty():
    prompt = _build_dedupe_prompt("", "", [SynthesizedBullet(text="Do X.")])
    assert "(empty — this is a brand-new file)" in prompt
    assert "Directory: (repository root)" in prompt


def test_dedupe_bullets_returns_input_unchanged_when_empty():
    client = _client_with_fake_backend(response_text="should not be used")
    assert client.dedupe_bullets("utilities", "content", []) == []


def test_dedupe_bullets_keeps_only_indices_in_keep_list():
    bullets = [
        SynthesizedBullet(text="Keep me."),
        SynthesizedBullet(text="Already covered."),
    ]
    client = _client_with_fake_backend(response_text=json.dumps({"keep": [1]}))

    kept = client.dedupe_bullets("utilities", "existing", bullets)

    assert [b.text for b in kept] == ["Keep me."]


def test_dedupe_bullets_keeps_all_when_keep_list_includes_everything():
    bullets = [SynthesizedBullet(text="A."), SynthesizedBullet(text="B.")]
    client = _client_with_fake_backend(response_text=json.dumps({"keep": [1, 2]}))

    kept = client.dedupe_bullets("utilities", "", bullets)

    assert [b.text for b in kept] == ["A.", "B."]


def test_dedupe_bullets_raises_on_api_error():
    client = _client_with_fake_backend(error=RuntimeError("boom"))
    with pytest.raises(LlmSynthesisError):
        client.dedupe_bullets("utilities", "content", [SynthesizedBullet(text="A.")])


def test_dedupe_bullets_raises_on_unparsable_json():
    client = _client_with_fake_backend(response_text="not json at all")
    with pytest.raises(LlmSynthesisError):
        client.dedupe_bullets("utilities", "content", [SynthesizedBullet(text="A.")])


def test_dedupe_bullets_raises_when_response_missing_keep_list():
    client = _client_with_fake_backend(response_text=json.dumps({"other": []}))
    with pytest.raises(LlmSynthesisError):
        client.dedupe_bullets("utilities", "content", [SynthesizedBullet(text="A.")])
