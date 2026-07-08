"""The communication channel: routes one step of message passing.

A :class:`CommunicationChannel` composes a
:class:`~communication.topology.Topology`, a
:class:`~communication.protocol.Protocol` and an optional plasticity rule over a
shared :class:`~communication.graph.InteractionGraph`. Each step it:

1. asks the topology to (re)wire the graph,
2. asks the protocol to aggregate each receiver's incoming messages, and
3. (optionally) applies a plasticity update to the edge weights.

Steps 1-2 are functional for the fully-connected/static topologies and the
mean-pool protocol, so the routing logic can be tested with synthetic messages
before any policy network is trained.
"""

from __future__ import annotations

from typing import Any, Mapping

from communication.graph import InteractionGraph
from communication.message import Message
from communication.protocol import make_protocol
from communication.topology import make_topology
from core.types import AgentID


class CommunicationChannel:
    """Route messages among agents for a single environment step."""

    def __init__(self, agent_ids, config: Any, plasticity_rule: Any | None = None) -> None:
        self.config = config
        self.graph = InteractionGraph(agent_ids)
        self.topology = make_topology(config)
        self.protocol = make_protocol(config)
        self.plasticity_rule = plasticity_rule

    def reset(self) -> None:
        self.topology.reset()

    def step(
        self,
        messages: Mapping[AgentID, Message],
        step: int = 0,
    ) -> dict[AgentID, Any]:
        """Return each agent's aggregated message context for this step.

        Parameters
        ----------
        messages:
            Mapping of sender id -> outgoing :class:`Message`.
        step:
            The current environment timestep (used by dynamic topologies).
        """
        if not self.config.enabled:
            return {agent: None for agent in self.graph.agents}

        self.topology.update(self.graph, messages, step)

        contexts: dict[AgentID, Any] = {}
        for receiver in self.graph.agents:
            contexts[receiver] = self.protocol.aggregate(receiver, messages, self.graph)

        if self.plasticity_rule is not None:
            # Plasticity consumes pre/post communication activity; wiring the
            # exact signals is part of the training milestone.
            self.plasticity_rule.apply(self.graph, messages, contexts, step)

        return contexts

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CommunicationChannel(topology={type(self.topology).__name__}, "
            f"protocol={type(self.protocol).__name__}, "
            f"plasticity={type(self.plasticity_rule).__name__ if self.plasticity_rule else None})"
        )


__all__ = ["CommunicationChannel"]
