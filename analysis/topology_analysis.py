"""Analyse how the interaction topology evolves over training.

Given a time-ordered sequence of :class:`~communication.graph.InteractionGraph`
snapshots, these helpers quantify structural change. They are functional and
build on :mod:`evaluation.graph_metrics`.
"""

from __future__ import annotations

from typing import Sequence

from communication.graph import InteractionGraph
from evaluation.graph_metrics import GRAPH_METRICS


def metric_trajectory(
    graphs: Sequence[InteractionGraph],
    metric_names: Sequence[str],
) -> dict[str, list[float]]:
    """Compute each named graph metric across a sequence of snapshots.

    Returns ``{metric_name: [value_at_snapshot_0, value_at_snapshot_1, ...]}``.
    """
    trajectories: dict[str, list[float]] = {name: [] for name in metric_names}
    for graph in graphs:
        for name in metric_names:
            trajectories[name].append(GRAPH_METRICS.get(name)(graph))
    return trajectories


def edge_change_rate(graphs: Sequence[InteractionGraph]) -> list[float]:
    """Fraction of edges that change between consecutive snapshots.

    A proxy for how quickly the topology rewires. Returns one value per
    consecutive pair (so ``len(graphs) - 1`` values).
    """
    import numpy as np

    rates: list[float] = []
    for prev, curr in zip(graphs, graphs[1:]):
        a = prev.adjacency_matrix() > 0
        b = curr.adjacency_matrix() > 0
        union = np.logical_or(a, b).sum()
        changed = np.logical_xor(a, b).sum()
        rates.append(float(changed / union) if union else 0.0)
    return rates


__all__ = ["metric_trajectory", "edge_change_rate"]
