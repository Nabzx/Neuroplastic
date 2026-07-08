"""Composite, multi-panel figures summarising a run (placeholder).

Intended layout: interaction-graph snapshots at several training checkpoints
alongside metric trajectories (graph/information/coordination) and a role-cluster
plot -- a single figure that tells the "did structure and specialisation emerge?"
story. Composed from the functional primitives in :mod:`visualisation.graphs`
and :mod:`visualisation.dynamics`.
"""

from __future__ import annotations

from typing import Any


def run_summary_dashboard(result: Any):  # pragma: no cover - deferred
    """Build the multi-panel run summary (placeholder).

    Depends on the run-result format from the training milestone
    (see docs/experiment_plan.md).
    """
    raise NotImplementedError(
        "run_summary_dashboard is a placeholder; it composes visualisation."
        "graphs + visualisation.dynamics once the run-result format exists."
    )


__all__ = ["run_summary_dashboard"]
