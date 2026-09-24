"""No-LLM fallback: cluster near-duplicate learning texts into clean bullets.

This is the default synthesis path (no Gemini API key required) and is what
makes `make run TARGET=...` work out of the box. It only catches duplicates
that are worded similarly (via `difflib`); see `llm.py` / the plan's "what
does a Gemini key add" section for what LLM mode improves on top of this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from learnings2agents.config import HEURISTIC_SIMILARITY_THRESHOLD
from learnings2agents.models import Learning, SynthesizedBullet


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _strip_boilerplate(text: str, repository: str) -> str:
    """Lightly trim repo-name preambles like "In the X repository, ..."."""
    cleaned = text
    if repository:
        repo_escaped = re.escape(repository)
        short_name = re.escape(repository.rsplit("/", 1)[-1])
        patterns = [
            rf"^In (the )?(?:[^/\s]+/)?{repo_escaped}(?: repository)?,\s*",
            rf"^In (the )?(?:[^/\s]+/)?{short_name}(?: repository)?,\s*",
            rf"^In (the )?(?:[^/\s]+/)?{repo_escaped} project(?: [^\s,]+)?,\s*",
            rf"^In (the )?(?:[^/\s]+/)?{short_name} project(?: [^\s,]+)?,\s*",
        ]
        for pattern in patterns:
            new_cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE, count=1)
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                break
    cleaned = " ".join(cleaned.split())  # collapse embedded newlines/whitespace
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.strip()


@dataclass
class _Cluster:
    """A group of near-duplicate learnings, represented by one bullet."""

    representative: Learning
    members: list[Learning] = field(default_factory=list)

    @property
    def all_learnings(self) -> list[Learning]:
        return [self.representative, *self.members]

    @property
    def pull_requests(self) -> list[str]:
        seen: list[str] = []
        for learning in self.all_learnings:
            if learning.pull_request and learning.pull_request not in seen:
                seen.append(learning.pull_request)
        return seen


def _cluster_learnings(learnings: list[Learning], threshold: float) -> list[_Cluster]:
    """Greedily cluster near-duplicate learning texts using difflib similarity.

    Two learnings land in the same cluster if their raw text similarity ratio
    is >= `threshold`. Within a cluster, the longest/most detailed text is
    kept as the representative bullet; other members only contribute their PR
    number as extra attribution.
    """
    clusters: list[_Cluster] = []
    for learning in learnings:
        best_cluster = None
        best_ratio = 0.0
        for cluster in clusters:
            ratio = _similarity(learning.text, cluster.representative.text)
            if ratio >= threshold and ratio > best_ratio:
                best_cluster, best_ratio = cluster, ratio
        if best_cluster is None:
            clusters.append(_Cluster(representative=learning))
            continue

        best_cluster.members.append(learning)
        if len(learning.text) > len(best_cluster.representative.text):
            # A more detailed duplicate showed up later; promote it and demote
            # the previous representative down to a plain member.
            best_cluster.members.append(best_cluster.representative)
            best_cluster.representative = learning
    return clusters


def synthesize_heuristic(
    learnings: list[Learning],
    threshold: float = HEURISTIC_SIMILARITY_THRESHOLD,
) -> list[SynthesizedBullet]:
    """Cluster `learnings` and turn each cluster into one `SynthesizedBullet`.

    Bullets are sorted by descending total `Usage` across their cluster
    members, so the most-referenced conventions surface first.
    """
    clusters = _cluster_learnings(learnings, threshold=threshold)

    bullets = []
    for cluster in clusters:
        text = _strip_boilerplate(
            cluster.representative.text, cluster.representative.repository
        )
        bullets.append(
            SynthesizedBullet(text=text, pull_requests=cluster.pull_requests)
        )

    usage_by_index = [
        sum(learning.usage for learning in cluster.all_learnings)
        for cluster in clusters
    ]
    order = sorted(range(len(bullets)), key=lambda i: usage_by_index[i], reverse=True)
    return [bullets[i] for i in order]
