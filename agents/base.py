"""The agent interface driven by the training loop.

An *agent* in this project is more than a policy: it is a node in a
communication graph. Each step it (1) encodes its observation, (2) emits a
message, (3) integrates messages received from its neighbours, and (4) selects
an action. Plasticity acts on the *incoming* message weights that gate step (3).

This module defines the abstract contract only. Concrete, trainable networks
live in :mod:`agents.recurrent_policy` and register themselves under
``AGENT_REGISTRY`` so they can be selected from config by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from communication.message import Message
from core.registry import Registry
from core.types import Action, AgentID, Observation

#: Registry of agent architectures. Selected via ``config.agent.type``.
AGENT_REGISTRY: Registry[type["BaseAgent"]] = Registry("agent")


@dataclass
class AgentStep:
    """The bundle returned by :meth:`BaseAgent.act` for one decision.

    ``value`` and ``log_prob`` are optional so the interface also fits
    value-free or scripted agents; the RL trainer will require them.
    """

    action: Action
    message: Message | None = None
    value: float | None = None
    log_prob: float | None = None
    extras: dict[str, Any] | None = None


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Parameters
    ----------
    agent_id:
        Stable identifier within an episode (matches the environment).
    obs_dim, action_dim:
        Shapes provided by the environment at construction time.
    config:
        The resolved ``agent`` section of the experiment config.
    """

    def __init__(
        self,
        agent_id: AgentID,
        obs_dim: int,
        action_dim: int,
        config: Any,
    ) -> None:
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        """Reset any per-episode internal state (e.g. recurrent hidden state)."""

    # -- perception / communication / action -------------------------------
    @abstractmethod
    def encode_message(self, observation: Observation) -> Message:
        """Produce the outgoing message for this step from ``observation``."""

    @abstractmethod
    def integrate(self, incoming: Mapping[AgentID, Message]) -> None:
        """Fold neighbour messages into internal state.

        ``incoming`` is keyed by sender id. The relative influence of each
        sender is gated by plastic weights (see :mod:`plasticity`), which is
        where Hebbian adaptation enters the forward pass.
        """

    @abstractmethod
    def act(self, observation: Observation, deterministic: bool = False) -> AgentStep:
        """Select an action given ``observation`` and integrated messages."""

    # -- convenience -------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(id={self.agent_id!r})"


def make_agent(
    agent_id: AgentID,
    obs_dim: int,
    action_dim: int,
    config: Any,
) -> BaseAgent:
    """Instantiate the agent named by ``config.type`` from ``AGENT_REGISTRY``."""
    # Importing here ensures built-in agents have registered themselves without
    # creating an import cycle at module load time.
    import agents.recurrent_policy  # noqa: F401  (registration side-effect)

    cls = AGENT_REGISTRY.get(config.type)
    return cls(agent_id=agent_id, obs_dim=obs_dim, action_dim=action_dim, config=config)


__all__ = ["BaseAgent", "AgentStep", "AGENT_REGISTRY", "make_agent"]
