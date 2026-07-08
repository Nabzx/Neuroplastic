"""Interaction-topology strategies: *who* communicates with *whom*.

A :class:`Topology` decides, each step, the directed edge set of the
:class:`~communication.graph.InteractionGraph`. Strategies range from fixed
(fully-connected, static) to dynamic/learned (adaptive), which is the setting of
interest for this project.

Concrete strategies register under ``TOPOLOGY_REGISTRY`` and are selected via
``config.communication.topology``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import permutations
from typing import Any, Mapping

from communication.graph import InteractionGraph
from communication.message import Message
from core.registry import Registry
from core.types import AgentID

TOPOLOGY_REGISTRY: Registry[type["Topology"]] = Registry("topology")


class Topology(ABC):
    """Base class for topology strategies."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def update(
        self,
        graph: InteractionGraph,
        messages: Mapping[AgentID, Message],
        step: int,
    ) -> None:
        """Mutate ``graph`` in place to reflect this step's edge set."""

    def reset(self) -> None:
        """Reset any per-episode internal state. No-op by default."""


@TOPOLOGY_REGISTRY.register("fully_connected")
class FullyConnectedTopology(Topology):
    """Every agent talks to every other agent (dense broadcast)."""

    def update(self, graph, messages, step) -> None:
        graph.set_edges(permutations(graph.agents, 2), weight=1.0)


@TOPOLOGY_REGISTRY.register("static")
class StaticTopology(Topology):
    """A fixed topology set once and never changed.

    Defaults to fully-connected on first use; a fixed edge list can be supplied
    via ``config.kwargs`` in a later milestone.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._initialised = False

    def reset(self) -> None:
        self._initialised = False

    def update(self, graph, messages, step) -> None:
        if not self._initialised:
            graph.set_edges(permutations(graph.agents, 2), weight=1.0)
            self._initialised = True


@TOPOLOGY_REGISTRY.register("k_nearest")
class KNearestTopology(Topology):
    """Connect each agent to its ``max_neighbours`` most-relevant senders.

    Placeholder: the relevance score (spatial proximity, message similarity, or
    a learned gate) is defined in docs/experiment_plan.md.
    """

    def update(self, graph, messages, step) -> None:  # pragma: no cover - deferred
        raise NotImplementedError(
            "KNearestTopology.update is a placeholder; the neighbour-selection "
            "score is specified in docs/experiment_plan.md."
        )


@TOPOLOGY_REGISTRY.register("adaptive")
class AdaptiveTopology(Topology):
    """A dynamic, learned topology -- the core object of study.

    Edges are (re)wired from a learned gating signal, and their weights are then
    shaped by Hebbian-inspired plasticity. Placeholder until the training
    milestone.
    """

    def update(self, graph, messages, step) -> None:  # pragma: no cover - deferred
        raise NotImplementedError(
            "AdaptiveTopology.update is a placeholder; the gating/rewiring "
            "mechanism is specified in docs/experiment_plan.md."
        )


def make_topology(config: Any) -> Topology:
    """Construct the topology named by ``config.topology``."""
    return TOPOLOGY_REGISTRY.get(config.topology)(config)


__all__ = [
    "Topology",
    "FullyConnectedTopology",
    "StaticTopology",
    "KNearestTopology",
    "AdaptiveTopology",
    "TOPOLOGY_REGISTRY",
    "make_topology",
]
