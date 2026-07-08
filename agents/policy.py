"""Reusable neural building blocks for policies (PyTorch).

These modules define the *shapes and wiring* of the networks. Forward passes
that require trained weights or nontrivial compute are left as clearly marked
placeholders -- the goal at this stage is a stable, reviewable interface, not a
trained model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encode a flat observation into a latent embedding."""

    def __init__(self, obs_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MessageHead(nn.Module):
    """Map a hidden state to an outgoing message vector."""

    def __init__(self, hidden_dim: int, message_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, message_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class ActorCriticHead(nn.Module):
    """A shared-torso actor-critic head.

    Emits action logits (discrete action spaces are assumed for the initial
    benchmarks) and a scalar state value.
    """

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value


class PolicyNetwork(nn.Module):
    """Full per-agent network: encoder -> (recurrent) core -> heads.

    The message-integration step (combining neighbour messages under plastic
    gating) is intentionally not implemented here yet; see the experiment plan
    for the intended design. The constructor wires every submodule so that
    parameter counts and I/O shapes are already correct and testable.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        obs_embedding_dim: int = 64,
        message_dim: int = 16,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim

        self.encoder = ObservationEncoder(obs_dim, obs_embedding_dim)
        # Core input = observation embedding + integrated message context.
        core_in = obs_embedding_dim + message_dim
        if recurrent:
            self.core: nn.Module = nn.GRUCell(core_in, hidden_dim)
        else:
            self.core = nn.Sequential(nn.Linear(core_in, hidden_dim), nn.ReLU())

        self.message_head = MessageHead(hidden_dim, message_dim)
        self.actor_critic = ActorCriticHead(hidden_dim, action_dim)

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zeroed recurrent hidden state."""
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, *args, **kwargs):  # pragma: no cover - deliberately deferred
        raise NotImplementedError(
            "PolicyNetwork.forward is a placeholder. The full forward pass "
            "(message integration under plastic gating + action/value/message "
            "outputs) is specified in docs/experiment_plan.md and will be "
            "implemented in the training milestone."
        )


__all__ = [
    "ObservationEncoder",
    "MessageHead",
    "ActorCriticHead",
    "PolicyNetwork",
]
