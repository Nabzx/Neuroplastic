"""Quantify functional specialisation from per-agent behavioural descriptors.

Given per-agent descriptors -- here, each agent's action distribution over a
greedy evaluation rollout -- we ask: do agents behave *differently* (distinct
functional roles)? We report a continuous ``role_diversity`` (mean pairwise
Jensen-Shannon distance) and a discrete role count/entropy from a simple
distance-threshold clustering (no external clustering dependency).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from core.types import ArrayLike


def role_entropy(labels: Sequence[int]) -> float:
    """Shannon entropy (bits) of the distribution of role-cluster labels.

    ``0`` = all agents share one role; higher = more evenly spread across roles.
    A compact scalar summary of role diversity in the collective.
    """
    import numpy as np

    if len(labels) == 0:
        return float("nan")
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def role_cluster_count(labels: Sequence[int]) -> int:
    """Number of distinct roles present."""
    return len(set(labels))


def _jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (bits, in ``[0, 1]``) between two distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(0.0, divergence)))


def cluster_roles(descriptors: ArrayLike, threshold: float = 0.25) -> list[int]:
    """Cluster agents into roles by connected components under a JS-distance threshold.

    Two agents share a role if their descriptors are within ``threshold`` (JS
    distance). Simple, deterministic and dependency-free -- adequate for the small
    agent counts here; a silhouette-selected k-means remains future work.
    """
    profiles = np.asarray(descriptors, dtype=float)
    n = profiles.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _jensen_shannon_distance(profiles[i], profiles[j]) < threshold:
                parent[find(i)] = find(j)

    roots = [find(i) for i in range(n)]
    relabel = {root: label for label, root in enumerate(sorted(set(roots)))}
    return [relabel[r] for r in roots]


def functional_specialisation(profiles: ArrayLike, threshold: float = 0.25) -> dict[str, float]:
    """Specialisation metrics from per-agent behavioural profiles ``[N, K]``.

    Returns ``role_diversity`` (mean pairwise JS distance, 0 = identical
    behaviour), plus ``role_cluster_count`` and ``role_entropy`` from a
    distance-threshold clustering.
    """
    data = np.asarray(profiles, dtype=float)
    n = data.shape[0]
    if n < 2:
        return {"role_diversity": 0.0, "role_cluster_count": float(n), "role_entropy": 0.0}

    distances = [
        _jensen_shannon_distance(data[i], data[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    labels = cluster_roles(data, threshold=threshold)
    return {
        "role_diversity": float(np.mean(distances)),
        "role_cluster_count": float(role_cluster_count(labels)),
        "role_entropy": role_entropy(labels),
    }


__all__ = [
    "role_entropy",
    "role_cluster_count",
    "cluster_roles",
    "functional_specialisation",
]
