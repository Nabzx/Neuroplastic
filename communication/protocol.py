"""Message-aggregation protocols: *how* received messages are combined.

Given the incoming messages on a receiver's edges (and their plastic weights), a
:class:`Protocol` produces the aggregated "message context" that the receiving
policy consumes. Strategies register under ``PROTOCOL_REGISTRY`` and are chosen
via ``config.communication.protocol``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from communication.graph import InteractionGraph
from communication.message import Message
from core.registry import Registry
from core.types import Action, AgentID  # noqa: F401 (Action re-exported for typing convenience)

PROTOCOL_REGISTRY: Registry[type["Protocol"]] = Registry("protocol")


class Protocol(ABC):
    """Base class for message-aggregation protocols."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def aggregate(
        self,
        receiver: AgentID,
        incoming: Mapping[AgentID, Message],
        graph: InteractionGraph,
    ):
        """Return the aggregated message context for ``receiver``."""


@PROTOCOL_REGISTRY.register("mean")
class MeanPoolProtocol(Protocol):
    """Weight-averaged pooling of neighbour messages.

    Fully specified and cheap; kept as the reference protocol. The averaging is
    weighted by the (plastic) edge weights in ``graph``.
    """

    def aggregate(self, receiver, incoming, graph):
        import numpy as np

        senders = [s for s in graph.neighbours(receiver) if s in incoming]
        if not senders:
            return None
        weights = np.array([graph.weight(s, receiver) for s in senders], dtype=float)
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total
        stack = np.stack([np.asarray(incoming[s].content, dtype=float) for s in senders])
        return (weights[:, None] * stack).sum(axis=0)


@PROTOCOL_REGISTRY.register("attention")
class AttentionProtocol(Protocol):
    """Attention-weighted aggregation over neighbour messages.

    This numpy Protocol (the analysis/channel path) is a placeholder. The
    *runnable* attention mechanism used during training is the differentiable
    :class:`communication.adaptive.AdaptiveCommunication` (PyTorch), selected via
    ``communication.protocol: attention``; it maintains a learned weighted edge
    matrix and aggregates messages with it.
    """

    def aggregate(self, receiver, incoming, graph):  # pragma: no cover - deferred
        raise NotImplementedError(
            "AttentionProtocol.aggregate (numpy path) is a placeholder; the "
            "runnable attention lives in communication.adaptive.AdaptiveCommunication."
        )


@PROTOCOL_REGISTRY.register("gnn")
class GNNProtocol(Protocol):
    """Graph-neural-network message passing over the interaction graph.

    Placeholder for a multi-round GNN aggregator.
    """

    def aggregate(self, receiver, incoming, graph):  # pragma: no cover - deferred
        raise NotImplementedError(
            "GNNProtocol.aggregate is a placeholder; the message-passing layers "
            "are specified in docs/experiment_plan.md."
        )


def make_protocol(config: Any) -> Protocol:
    """Construct the protocol named by ``config.protocol``."""
    return PROTOCOL_REGISTRY.get(config.protocol)(config)


__all__ = [
    "Protocol",
    "MeanPoolProtocol",
    "AttentionProtocol",
    "GNNProtocol",
    "PROTOCOL_REGISTRY",
    "make_protocol",
]
