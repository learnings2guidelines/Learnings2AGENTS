from __future__ import annotations

from learnings2agents.heuristics import synthesize_heuristic
from learnings2agents.models import Learning


def _learning(
    text: str, pull_request: str, usage: int = 1, repository: str = "acme-repo"
) -> Learning:
    return Learning(
        text=text,
        repository=repository,
        file="utilities/foo.py",
        pull_request=pull_request,
        url="",
        created_by="someone",
        usage=usage,
    )


def test_near_duplicate_texts_are_merged_into_one_bullet():
    learnings = [
        _learning(
            "In the acme-repo repository, always use TimeoutSampler instead of "
            "bare time.sleep() calls in test code, per maintainer guidance.",
            pull_request="100",
            usage=5,
        ),
        _learning(
            "In the acme-repo repository, always use TimeoutSampler instead of "
            "bare time.sleep() calls in test code, per maintainer feedback.",
            pull_request="200",
            usage=3,
        ),
    ]
    bullets = synthesize_heuristic(learnings)
    assert len(bullets) == 1
    assert set(bullets[0].pull_requests) == {"100", "200"}
    # Boilerplate repo preamble should be stripped.
    assert "acme-repo repository" not in bullets[0].text
    assert bullets[0].text[0].isupper()


def test_distinct_texts_remain_separate_bullets():
    learnings = [
        _learning("Always import logging as a whole module.", pull_request="1"),
        _learning(
            "Assertion messages are required for integration tests.", pull_request="2"
        ),
    ]
    bullets = synthesize_heuristic(learnings)
    assert len(bullets) == 2
    texts = {b.text for b in bullets}
    assert any("logging" in t for t in texts)
    assert any("Assertion" in t for t in texts)


def test_bullets_sorted_by_descending_usage():
    learnings = [
        _learning("Low usage rule about docstrings.", pull_request="1", usage=1),
        _learning("High usage rule about imports.", pull_request="2", usage=50),
    ]
    bullets = synthesize_heuristic(learnings)
    assert bullets[0].text.startswith("High usage")
    assert bullets[1].text.startswith("Low usage")


def test_no_repository_prefix_left_when_repository_missing():
    learning = Learning(
        text="Do the thing consistently.",
        repository="",
        file="a.py",
        pull_request="1",
        url="",
        created_by="x",
        usage=1,
    )
    bullets = synthesize_heuristic([learning])
    assert bullets[0].text == "Do the thing consistently."
