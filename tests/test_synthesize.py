from __future__ import annotations

from learnings2agents.cache import SynthesisCache
from learnings2agents.llm import LlmSynthesisError
from learnings2agents.models import DirGroup, Learning, MergePlan, SynthesizedBullet
from learnings2agents.synthesize import synthesize_all, synthesize_consolidated_root


def _group(path: str = "utilities") -> DirGroup:
    learning = Learning(
        text="Always do the thing.",
        repository="repo",
        file=f"{path}/foo.py" if path else "foo.py",
        pull_request="42",
        url="",
        created_by="x",
        usage=1,
    )
    return DirGroup(path=path, learnings=[learning])


class _FakeGeminiClient:
    def __init__(
        self,
        bullets=None,
        error: Exception | None = None,
        merge_plan: MergePlan | None = None,
        merge_error: Exception | None = None,
        dedupe_keep: list[int] | None = None,
        dedupe_error: Exception | None = None,
        root_bullets=None,
        root_error: Exception | None = None,
    ):
        self.model = "fake-model"
        self._bullets = bullets
        self._error = error
        self._merge_plan = merge_plan
        self._merge_error = merge_error
        # `dedupe_keep=None` means "keep everything unchanged" (the default
        # most tests want, since they aren't exercising the dedupe check
        # itself); otherwise a list of 1-based indices (into whatever
        # `bullets` list is passed to `dedupe_bullets`) to keep.
        self._dedupe_keep = dedupe_keep
        self._dedupe_error = dedupe_error
        self._root_bullets = root_bullets
        self._root_error = root_error
        self.calls = 0
        self.merge_calls = 0
        self.dedupe_calls = 0
        self.root_calls = 0
        self.last_dedupe_args = None

    def synthesize_directory(self, directory, learnings):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._bullets

    def plan_agents_md_merge(self, directory, learnings, existing_content):
        self.merge_calls += 1
        if self._merge_error is not None:
            raise self._merge_error
        return self._merge_plan

    def dedupe_bullets(self, directory, existing_content, bullets):
        self.dedupe_calls += 1
        self.last_dedupe_args = (directory, existing_content, bullets)
        if self._dedupe_error is not None:
            raise self._dedupe_error
        if self._dedupe_keep is None:
            return bullets
        return [b for i, b in enumerate(bullets, start=1) if i in self._dedupe_keep]

    def synthesize_root(self, groups):
        self.root_calls += 1
        if self._root_error is not None:
            raise self._root_error
        return self._root_bullets


def test_synthesize_all_uses_heuristic_when_no_gemini_client():
    group = _group()
    synthesize_all([group], gemini_client=None, cache=None)
    assert group.mode_used == "heuristic"
    assert len(group.bullets) == 1


def test_synthesize_all_uses_llm_when_client_succeeds():
    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets)

    synthesize_all([group], gemini_client=client, cache=None)

    assert group.mode_used == "llm"
    assert group.bullets == bullets
    assert client.calls == 1


def test_synthesize_all_falls_back_to_heuristic_on_llm_error():
    group = _group()
    client = _FakeGeminiClient(error=LlmSynthesisError("boom"))

    synthesize_all([group], gemini_client=client, cache=None)

    assert group.mode_used == "heuristic"
    assert len(group.bullets) == 1
    assert client.calls == 1


def test_synthesize_all_uses_cache_on_second_run(tmp_path, monkeypatch):
    cache = SynthesisCache(tmp_path / "cache")
    call_count = {"n": 0}

    import learnings2agents.synthesize as synthesize_mod

    real_heuristic = synthesize_mod.synthesize_heuristic

    def counting_heuristic(learnings):
        call_count["n"] += 1
        return real_heuristic(learnings)

    monkeypatch.setattr(synthesize_mod, "synthesize_heuristic", counting_heuristic)

    group1 = _group()
    synthesize_all([group1], gemini_client=None, cache=cache)
    assert call_count["n"] == 1

    # Fresh DirGroup with identical learnings content -> should hit the cache.
    group2 = _group()
    synthesize_all([group2], gemini_client=None, cache=cache)
    assert call_count["n"] == 1  # not called again
    assert group2.bullets[0].text == group1.bullets[0].text
    assert group2.mode_used == "heuristic"


def _write_existing_agents_md(target, path: str, content: str) -> None:
    dir_path = target / path if path else target
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "AGENTS.md").write_text(content, encoding="utf-8")


def test_synthesize_all_uses_merge_plan_when_existing_agents_md_present(tmp_path):
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")

    group = _group()
    plan = MergePlan(
        new_bullets=[SynthesizedBullet(text="Merged in.", pull_requests=["42"])]
    )
    client = _FakeGeminiClient(merge_plan=plan)

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert group.mode_used == "llm-merge"
    assert group.merged_content is not None
    assert "Existing." in group.merged_content
    assert "Merged in." in group.merged_content
    assert client.merge_calls == 1
    assert client.calls == 0  # the regular synthesize_directory path is unused


def test_synthesize_all_falls_back_to_bullets_path_when_merge_planning_fails(
    tmp_path,
):
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")

    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets, merge_error=LlmSynthesisError("boom"))

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.merge_calls == 1
    assert client.calls == 1
    assert group.merged_content is None
    assert group.mode_used == "llm"
    assert group.bullets == bullets


def test_synthesize_all_uses_regular_path_when_no_existing_agents_md(tmp_path):
    target = tmp_path / "target"
    (target / "utilities").mkdir(parents=True)  # dir exists, but no AGENTS.md yet

    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets)

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.merge_calls == 0
    assert client.calls == 1
    assert group.mode_used == "llm"
    assert group.merged_content is None


def test_synthesize_all_target_none_preserves_prior_behavior(tmp_path):
    """Callers that don't pass `target` (e.g. older code/tests) never take
    the merge-plan branch, even if an AGENTS.md happens to exist somewhere
    they aren't told about."""
    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets)

    synthesize_all([group], gemini_client=client, cache=None, target=None)

    assert client.merge_calls == 0
    assert client.calls == 1
    assert group.merged_content is None


def test_synthesize_all_caches_merge_plan_across_runs(tmp_path):
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")
    cache = SynthesisCache(tmp_path / "cache")

    plan = MergePlan(new_bullets=[SynthesizedBullet(text="Merged in.")])
    client = _FakeGeminiClient(merge_plan=plan)

    group1 = _group()
    synthesize_all([group1], gemini_client=client, cache=cache, target=target)
    assert client.merge_calls == 1

    group2 = _group()
    synthesize_all([group2], gemini_client=client, cache=cache, target=target)
    assert client.merge_calls == 1  # cache hit -> not called again
    assert group2.merged_content == group1.merged_content
    assert group2.mode_used == "llm-merge"


# --- Redundancy ("dedupe") re-check, requirement #3: when a Gemini client
# is used, newly-added bullets are checked once more against the file's
# current content (and each other) before being written. ---


def test_synthesize_all_drops_redundant_bullets_in_regular_path(tmp_path):
    target = tmp_path / "target"
    (target / "utilities").mkdir(parents=True)  # dir exists, no AGENTS.md yet

    group = _group()
    bullets = [
        SynthesizedBullet(text="Keep me.", pull_requests=["42"]),
        SynthesizedBullet(text="Already covered elsewhere.", pull_requests=["42"]),
    ]
    client = _FakeGeminiClient(bullets=bullets, dedupe_keep=[1])

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.dedupe_calls == 1
    assert [b.text for b in group.bullets] == ["Keep me."]
    # Baseline passed to the dedupe check is "" since no AGENTS.md exists yet.
    assert client.last_dedupe_args[1] == ""


def test_synthesize_all_dedupe_uses_existing_file_as_baseline_when_merge_planning_fails(
    tmp_path,
):
    """If an AGENTS.md already exists but merge-planning fails, the regular
    bullets-from-scratch path is used, and the dedupe check should compare
    against the file's actual existing content, not an empty string."""
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")

    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets, merge_error=LlmSynthesisError("boom"))

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.dedupe_calls == 1
    assert client.last_dedupe_args[1] == "# AGENTS.md\n\nExisting.\n"
    assert group.bullets == bullets


def test_synthesize_all_keeps_bullets_when_dedupe_check_fails(tmp_path):
    target = tmp_path / "target"
    (target / "utilities").mkdir(parents=True)

    group = _group()
    bullets = [SynthesizedBullet(text="Do the thing.", pull_requests=["42"])]
    client = _FakeGeminiClient(bullets=bullets, dedupe_error=LlmSynthesisError("boom"))

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.dedupe_calls == 1
    assert group.bullets == bullets  # unchanged — dedupe failure isn't fatal


def test_synthesize_all_skips_dedupe_without_gemini_client():
    group = _group()
    synthesize_all([group], gemini_client=None, cache=None)
    assert group.mode_used == "heuristic"
    assert len(group.bullets) == 1  # ran fine without ever touching dedupe


def test_synthesize_all_drops_redundant_new_bullets_in_merge_plan_path(tmp_path):
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")

    group = _group()
    plan = MergePlan(
        new_bullets=[
            SynthesizedBullet(text="Keep me."),
            SynthesizedBullet(text="Duplicate of something existing."),
        ]
    )
    client = _FakeGeminiClient(merge_plan=plan, dedupe_keep=[1])

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.dedupe_calls == 1
    assert "Keep me." in group.merged_content
    assert "Duplicate of something existing." not in group.merged_content


def test_synthesize_all_caches_the_post_dedupe_merge_plan(tmp_path):
    """The cached merge plan should already reflect dedupe-dropped bullets,
    so a second run (cache hit) doesn't need to re-run the dedupe check."""
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")
    cache = SynthesisCache(tmp_path / "cache")

    plan = MergePlan(
        new_bullets=[
            SynthesizedBullet(text="Keep me."),
            SynthesizedBullet(text="Dropped by dedupe."),
        ]
    )
    client = _FakeGeminiClient(merge_plan=plan, dedupe_keep=[1])

    group1 = _group()
    synthesize_all([group1], gemini_client=client, cache=cache, target=target)
    assert client.merge_calls == 1
    assert client.dedupe_calls == 1
    assert "Dropped by dedupe." not in group1.merged_content

    group2 = _group()
    synthesize_all([group2], gemini_client=client, cache=cache, target=target)
    assert client.merge_calls == 1  # cache hit
    assert client.dedupe_calls == 1  # not re-run on cache hit
    assert group2.merged_content == group1.merged_content


def test_synthesize_all_keeps_new_bullets_when_merge_dedupe_check_fails(tmp_path):
    target = tmp_path / "target"
    _write_existing_agents_md(target, "utilities", "# AGENTS.md\n\nExisting.\n")

    group = _group()
    plan = MergePlan(new_bullets=[SynthesizedBullet(text="Brand new.")])
    client = _FakeGeminiClient(merge_plan=plan, dedupe_error=LlmSynthesisError("boom"))

    synthesize_all([group], gemini_client=client, cache=None, target=target)

    assert client.dedupe_calls == 1
    assert "Brand new." in group.merged_content


# --- Consolidated root synthesis, requirement #2. ---


def _groups_for_root() -> list[DirGroup]:
    return [
        DirGroup(
            path="utilities",
            learnings=[
                Learning(
                    text="Import modules as a whole.",
                    repository="repo",
                    file="utilities/foo.py",
                    pull_request="1",
                    url="",
                    created_by="x",
                    usage=1,
                )
            ],
        ),
        DirGroup(
            path="tests",
            learnings=[
                Learning(
                    text="Use TimeoutSampler.",
                    repository="repo",
                    file="tests/foo.py",
                    pull_request="2",
                    url="",
                    created_by="x",
                    usage=1,
                )
            ],
        ),
    ]


def test_synthesize_consolidated_root_combines_all_groups():
    groups = _groups_for_root()
    root_bullets = [
        SynthesizedBullet(text="Import modules as a whole.", heading="Imports"),
        SynthesizedBullet(text="Use TimeoutSampler.", heading="Testing"),
    ]
    client = _FakeGeminiClient(root_bullets=root_bullets)

    result = synthesize_consolidated_root(groups, client, cache=None)

    assert client.root_calls == 1
    assert client.dedupe_calls == 1
    assert [b.text for b in result] == [b.text for b in root_bullets]
    # The dedupe baseline is empty: it's a brand-new consolidated file.
    assert client.last_dedupe_args[1] == ""


def test_synthesize_consolidated_root_drops_redundant_bullets_via_dedupe():
    groups = _groups_for_root()
    root_bullets = [
        SynthesizedBullet(text="Keep me.", heading="A"),
        SynthesizedBullet(text="Cross-directory duplicate.", heading="B"),
    ]
    client = _FakeGeminiClient(root_bullets=root_bullets, dedupe_keep=[1])

    result = synthesize_consolidated_root(groups, client, cache=None)

    assert [b.text for b in result] == ["Keep me."]


def test_synthesize_consolidated_root_raises_when_llm_fails():
    groups = _groups_for_root()
    client = _FakeGeminiClient(root_error=LlmSynthesisError("boom"))

    import pytest

    with pytest.raises(LlmSynthesisError):
        synthesize_consolidated_root(groups, client, cache=None)


def test_synthesize_consolidated_root_uses_cache_on_second_run(tmp_path):
    groups = _groups_for_root()
    cache = SynthesisCache(tmp_path / "cache")
    root_bullets = [SynthesizedBullet(text="A rule.", heading="General")]
    client = _FakeGeminiClient(root_bullets=root_bullets)

    result1 = synthesize_consolidated_root(groups, client, cache=cache)
    assert client.root_calls == 1

    result2 = synthesize_consolidated_root(groups, client, cache=cache)
    assert client.root_calls == 1  # cache hit -> not called again
    assert [b.text for b in result2] == [b.text for b in result1]
    # The dedupe re-check still runs fresh every time, even on a cache hit.
    assert client.dedupe_calls == 2
