"""Runtime hooks for measuring *functional specialisation*.

A central research question is whether agents differentiate into distinct
functional roles. This module defines how a per-agent "role descriptor" is
extracted at runtime; the statistical analysis of those descriptors (clustering,
role entropy, stability over training) lives in
:mod:`analysis.specialisation_analysis`.

The extraction itself is a placeholder: candidate descriptors include the
agent's action distribution, its outgoing-message statistics, and its position
in the interaction graph.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import ArrayLike


@runtime_checkable
class RoleProbe(Protocol):
    """Anything that can summarise an agent's behaviour as a role vector."""

    def role_descriptor(self) -> ArrayLike:
        """Return a fixed-length vector characterising the agent's role."""
        ...


def extract_role_descriptor(agent: object) -> ArrayLike:  # pragma: no cover - deferred
    """Extract a role descriptor from ``agent``.

    Placeholder. Intended to aggregate, over a rollout window:
    action-distribution summaries, message-emission statistics, and
    graph-centrality of the agent's node. See docs/experiment_plan.md.
    """
    raise NotImplementedError(
        "extract_role_descriptor is a placeholder; the descriptor design is "
        "specified in docs/experiment_plan.md (functional specialisation)."
    )


__all__ = ["RoleProbe", "extract_role_descriptor"]
