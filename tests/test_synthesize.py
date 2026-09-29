from __future__ import annotations

from learnings2agents.cache import SynthesisCache
from learnings2agents.llm import LlmSynthesisError
from learnings2agents.models import DirGroup, Learning, MergePlan, SynthesizedBullet
from learnings2agents.synthesize import synthesize_all


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
    ):
        self.model = "fake-model"
        self._bullets = bullets
        self._error = error
        self._merge_plan = merge_plan
        self._merge_error = merge_error
        self.calls = 0
        self.merge_calls = 0

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
