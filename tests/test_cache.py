from __future__ import annotations

from learnings2agents.cache import SynthesisCache
from learnings2agents.models import Learning, MergePlan, SynthesizedBullet, TextEdit


def _learning(text: str, file: str = "utilities/foo.py") -> Learning:
    return Learning(
        text=text,
        repository="acme-repo",
        file=file,
        pull_request="1",
        url="",
        created_by="someone",
        usage=1,
    )


def test_get_returns_none_on_miss(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    assert cache.get("utilities", [_learning("Do X.")], "heuristic", "") is None


def test_set_then_get_round_trips_bullets(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    learnings = [_learning("Do X.")]
    bullets = [SynthesizedBullet(text="Do X.", pull_requests=["1"], heading="Style")]

    cache.set("utilities", learnings, "llm", "gemini-x", bullets)
    cached = cache.get("utilities", learnings, "llm", "gemini-x")

    assert cached is not None
    assert cached[0].text == "Do X."
    assert cached[0].pull_requests == ["1"]
    assert cached[0].heading == "Style"


def test_get_merge_plan_returns_none_on_miss(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    assert (
        cache.get_merge_plan("utilities", [_learning("Do X.")], "content", "model")
        is None
    )


def test_set_then_get_merge_plan_round_trips(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    learnings = [_learning("Do X.")]
    plan = MergePlan(
        edits=[TextEdit(old_text="old", new_text="new")],
        new_bullets=[SynthesizedBullet(text="Brand new.", pull_requests=["1"])],
    )

    cache.set_merge_plan("utilities", learnings, "existing content", "gemini-x", plan)
    cached = cache.get_merge_plan(
        "utilities", learnings, "existing content", "gemini-x"
    )

    assert cached is not None
    assert len(cached.edits) == 1
    assert cached.edits[0].old_text == "old"
    assert cached.edits[0].new_text == "new"
    assert len(cached.new_bullets) == 1
    assert cached.new_bullets[0].text == "Brand new."
    assert cached.new_bullets[0].pull_requests == ["1"]


def test_merge_plan_cache_is_invalidated_when_existing_content_changes(tmp_path):
    cache = SynthesisCache(tmp_path / "cache")
    learnings = [_learning("Do X.")]
    plan = MergePlan(edits=[], new_bullets=[SynthesizedBullet(text="X.")])

    cache.set_merge_plan("utilities", learnings, "content v1", "gemini-x", plan)

    # Same directory + learnings, but the on-disk AGENTS.md content differs
    # (e.g. hand-edited, or written by a previous run) -> must be a cache miss.
    assert (
        cache.get_merge_plan("utilities", learnings, "content v2", "gemini-x") is None
    )
    # The original content is still a hit.
    assert (
        cache.get_merge_plan("utilities", learnings, "content v1", "gemini-x")
        is not None
    )


def test_merge_plan_cache_is_separate_from_regular_bullet_cache(tmp_path):
    """Same path/learnings/model shouldn't collide between the two cache kinds."""
    cache = SynthesisCache(tmp_path / "cache")
    learnings = [_learning("Do X.")]

    cache.set("utilities", learnings, "llm", "gemini-x", [SynthesizedBullet(text="A.")])
    assert cache.get_merge_plan("utilities", learnings, "content", "gemini-x") is None

    cache.set_merge_plan(
        "utilities", learnings, "content", "gemini-x", MergePlan(new_bullets=[])
    )
    assert cache.get("utilities", learnings, "llm", "gemini-x") is not None
