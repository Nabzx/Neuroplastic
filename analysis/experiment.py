"""Experiment result records and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentResult:
    """A single run's results, keyed for cross-condition comparison.

    ``metrics`` holds final scalar metrics; ``topology_snapshots`` holds
    (step, adjacency) pairs recorded during training for temporal analysis.
    """

    name: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    topology_snapshots: list[tuple[int, Any]] = field(default_factory=list)


def load_run(path: str) -> ExperimentResult:  # pragma: no cover - deferred
    """Load a saved run directory into an :class:`ExperimentResult` (placeholder).

    The on-disk run format is defined by the training milestone
    (see docs/experiment_plan.md).
    """
    raise NotImplementedError(
        "load_run is a placeholder; the run directory format is specified in "
        "docs/experiment_plan.md."
    )


__all__ = ["ExperimentResult", "load_run"]
