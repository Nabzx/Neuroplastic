"""Quantify functional specialisation from per-agent role descriptors.

Given role descriptors (see :func:`agents.specialisation.extract_role_descriptor`),
we ask: do agents form distinct clusters (roles)? and how differentiated is the
population? ``role_entropy`` is functional given cluster labels; the clustering
of continuous descriptors is deferred so we do not hard-depend on a clustering
library at this stage.
"""

from __future__ import annotations

from typing import Sequence

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


def cluster_roles(descriptors: ArrayLike, max_roles: int = 8) -> list[int]:  # pragma: no cover - deferred
    """Cluster continuous role descriptors into discrete roles (placeholder).

    Intended approach (per docs/experiment_plan.md): silhouette-selected k-means
    (or a gap statistic) over standardised descriptors.
    """
    raise NotImplementedError(
        "cluster_roles is a placeholder; the clustering procedure is specified "
        "in docs/experiment_plan.md."
    )


__all__ = ["role_entropy", "role_cluster_count", "cluster_roles"]
