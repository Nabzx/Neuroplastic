"""The default recurrent, communicating agent.

Wraps :class:`~agents.policy.PolicyNetwork` behind the
:class:`~agents.base.BaseAgent` interface and registers itself as
``"recurrent_policy"``. Behavioural methods are placeholders until the training
milestone; the class exists now so that wiring, construction and config
plumbing can be exercised and tested end-to-end.
"""

from __future__ import annotations

from typing import Any, Mapping

from agents.base import AGENT_REGISTRY, AgentStep, BaseAgent
from communication.message import Message
from core.types import AgentID, Observation


@AGENT_REGISTRY.register("recurrent_policy")
class RecurrentPolicyAgent(BaseAgent):
    """A recurrent actor-critic agent that emits and integrates messages."""

    def __init__(
        self,
        agent_id: AgentID,
        obs_dim: int,
        action_dim: int,
        config: Any,
    ) -> None:
        super().__init__(agent_id, obs_dim, action_dim, config)
        # The torch network is constructed lazily so that non-training tooling
        # (config validation, graph analysis) does not require a torch install.
        self._net = None
        self._hidden = None
        self._message_context = None

    def _ensure_net(self):
        if self._net is None:
            from agents.policy import PolicyNetwork

            message_dim = getattr(self.config, "message_dim", 16)
            self._net = PolicyNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.hidden_dim,
                obs_embedding_dim=self.config.obs_embedding_dim,
                message_dim=message_dim,
                recurrent=self.config.recurrent,
            )
        return self._net

    def reset(self) -> None:
        net = self._ensure_net()
        self._hidden = net.initial_state(batch_size=1)
        self._message_context = None

    def encode_message(self, observation: Observation) -> Message:  # pragma: no cover
        raise NotImplementedError(
            "encode_message will run the encoder + message head. Deferred to "
            "the training milestone (see docs/experiment_plan.md)."
        )

    def integrate(self, incoming: Mapping[AgentID, Message]) -> None:  # pragma: no cover
        raise NotImplementedError(
            "integrate will pool neighbour messages under plastic gating and "
            "update the message context. Deferred to the training milestone."
        )

    def act(self, observation: Observation, deterministic: bool = False) -> AgentStep:  # pragma: no cover
        raise NotImplementedError(
            "act will run the full policy forward pass. Deferred to the "
            "training milestone (see docs/experiment_plan.md)."
        )


__all__ = ["RecurrentPolicyAgent"]
