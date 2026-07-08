"""Orchestrates metric computation over recorded rollouts.

The graph-metric path is functional today (see
:func:`evaluation.graph_metrics.compute_graph_metrics`); the full evaluation
over recorded rollouts -- which needs the rollout record format from the
training milestone -- is deferred.
"""

from __future__ import annotations

from typing import Any

from communication.graph import InteractionGraph
from configs.schema import EvaluationConfig
from evaluation.graph_metrics import compute_graph_metrics


class Evaluator:
    """Compute the configured emergent-behaviour metrics."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate_topology(self, graph: InteractionGraph) -> dict[str, float]:
        """Compute the configured graph-theoretic metrics for one topology.

        This is the functional entrypoint: it works on any interaction graph
        (synthetic, or a snapshot recorded during training).
        """
        return compute_graph_metrics(graph, self.config.graph_metrics)

    def evaluate(self, rollouts: Any) -> dict[str, Any]:  # pragma: no cover - deferred
        """Full evaluation over recorded rollouts (placeholder).

        Will combine graph, information and coordination metrics into a single
        report once the rollout record format exists (training milestone).
        """
        raise NotImplementedError(
            "Evaluator.evaluate over rollouts is a placeholder; the rollout "
            "record schema is specified in docs/experiment_plan.md. Use "
            "evaluate_topology() for graph metrics in the meantime."
        )


__all__ = ["Evaluator"]
