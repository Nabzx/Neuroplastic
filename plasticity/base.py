"""The plasticity-rule interface.

A :class:`PlasticityRule` observes communication activity for a step and updates
the plastic edge weights of a :class:`~communication.graph.InteractionGraph`.
Rules register under ``PLASTICITY_REGISTRY`` and are selected via
``config.plasticity.rule``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from communication.graph import InteractionGraph
from communication.message import Message
from core.registry import Registry
from core.types import AgentID

PLASTICITY_REGISTRY: Registry[type["PlasticityRule"]] = Registry("plasticity_rule")


class PlasticityRule(ABC):
    """Base class for weight-update rules on the interaction graph."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.learning_rate = getattr(config, "learning_rate", 1e-3)
        self.decay = getattr(config, "decay", 0.0)
        self.update_every = getattr(config, "update_every", 1)
        self._step_counter = 0

    def apply(
        self,
        graph: InteractionGraph,
        messages: Mapping[AgentID, Message],
        contexts: Mapping[AgentID, Any],
        step: int,
    ) -> None:
        """Throttle by ``update_every`` then delegate to :meth:`update`."""
        self._step_counter += 1
        if self._step_counter % max(1, self.update_every) != 0:
            return
        self.update(graph, messages, contexts, step)

    @abstractmethod
    def update(
        self,
        graph: InteractionGraph,
        messages: Mapping[AgentID, Message],
        contexts: Mapping[AgentID, Any],
        step: int,
    ) -> None:
        """Update ``graph`` edge weights in place from communication activity."""

    def reset(self) -> None:
        self._step_counter = 0


class NoPlasticity(PlasticityRule):
    """A no-op rule used when plasticity is disabled."""

    def update(self, graph, messages, contexts, step) -> None:
        return None


PLASTICITY_REGISTRY.register("none", NoPlasticity)


def make_plasticity(config: Any) -> PlasticityRule | None:
    """Construct the plasticity rule named by ``config.rule``.

    Returns ``None`` when plasticity is disabled so callers can cheaply skip the
    update. Importing the built-in rules here triggers their registration.
    """
    import plasticity.hebbian  # noqa: F401 (registration side-effect)

    if not getattr(config, "enabled", True) or config.rule in ("none", None):
        return None
    return PLASTICITY_REGISTRY.get(config.rule)(config)


__all__ = ["PlasticityRule", "NoPlasticity", "PLASTICITY_REGISTRY", "make_plasticity"]
