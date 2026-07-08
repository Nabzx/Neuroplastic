"""On-policy rollout storage with GAE.

The buffer stores *joint* transitions (all ``N`` agents per timestep) because the
communication context couples agents within a timestep -- minibatches must keep
whole timesteps together so the message aggregation can be recomputed during the
PPO update. Everything lives on the training device as flat tensors.
"""

from __future__ import annotations

from typing import Any

import torch


class RolloutBuffer:
    """Fixed-size buffer of joint transitions with GAE advantage computation."""

    def __init__(self, rollout_length: int, num_agents: int, obs_dim: int, device: str) -> None:
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.device = device

        shape_tn = (rollout_length, num_agents)
        self.obs = torch.zeros((*shape_tn, obs_dim), device=device)
        self.actions = torch.zeros(shape_tn, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(shape_tn, device=device)
        self.values = torch.zeros(shape_tn, device=device)
        self.rewards = torch.zeros(shape_tn, device=device)
        self.dones = torch.zeros(rollout_length, device=device)  # episode boundary per step

        self.advantages = torch.zeros(shape_tn, device=device)
        self.returns = torch.zeros(shape_tn, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: float,
    ) -> None:
        """Store one joint transition (tensors are shaped ``[N, ...]``)."""
        i = self.ptr
        if i >= self.rollout_length:
            raise IndexError("RolloutBuffer is full; call reset() before reuse.")
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns, bootstrapping from ``last_value``.

        ``last_value`` is ``V(s_T)`` for the observation held after the rollout.
        Storage convention: ``dones[t]`` is 1 iff the episode ended *at* step
        ``t`` (transition ``t`` is terminal), so the bootstrap mask for step
        ``t`` is ``1 - dones[t]`` -- this stops advantages leaking across episode
        boundaries.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_agents, device=self.device)
        for t in reversed(range(self.rollout_length)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.rollout_length - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        self.advantages = advantages
        self.returns = advantages + self.values

    def get(self) -> dict[str, Any]:
        """Return the stored rollout as a dict of tensors."""
        return {
            "obs": self.obs,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def reset(self) -> None:
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr


__all__ = ["RolloutBuffer"]
