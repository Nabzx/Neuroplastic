"""Task-level coordination outcomes.

These summarise *whether the collective actually coordinated*, complementing the
structural (graph) and informational measures. ``episode_return`` is functional;
task-specific success and the coordination index depend on recorded rollouts /
environment info and are deferred.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from core.registry import Registry
from core.types import AgentID

COORDINATION_METRICS: Registry = Registry("coordination_metric")


def episode_return(reward_stream: Sequence[Mapping[AgentID, float]]) -> float:
    """Total team return: sum of all agents' rewards over an episode."""
    return float(sum(sum(step.values()) for step in reward_stream))


def success_rate(episode_infos: Sequence[Mapping[str, object]]) -> float:  # pragma: no cover - deferred
    """Fraction of episodes flagged successful by the environment (placeholder)."""
    raise NotImplementedError(
        "success_rate depends on per-benchmark success signals; specified in "
        "docs/experiment_plan.md."
    )


def coordination_index(*args, **kwargs) -> float:  # pragma: no cover - deferred
    """A normalised measure of how much better the team does than independent
    agents on the same task (placeholder). See docs/experiment_plan.md."""
    raise NotImplementedError("coordination_index is a placeholder.")


COORDINATION_METRICS.register("episode_return", episode_return)
COORDINATION_METRICS.register("success_rate", success_rate)
COORDINATION_METRICS.register("coordination_index", coordination_index)


__all__ = [
    "COORDINATION_METRICS",
    "episode_return",
    "success_rate",
    "coordination_index",
]
