"""Shared-parameter PPO learner (the first learning baseline).

One :class:`~agents.policy.PolicyNetwork` is shared by all agents (parameter
sharing -- the standard baseline for homogeneous cooperative agents). Actions are
sampled per agent; the PPO update recomputes the forward pass over sampled
timesteps so that gradients flow through the communication aggregation and the
message head is learned end-to-end.

Plasticity is intentionally absent here: communication is a *fixed* mask. This
class establishes the clean baseline that neuroplastic variants will be compared
against.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.policy import PolicyNetwork


class SharedPPOLearner:
    """PPO with parameter sharing and a fixed communication mask."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        adjacency: torch.Tensor | None,
        config: Any,
        device: str,
    ) -> None:
        self.device = device
        self.num_agents = num_agents
        self.adjacency = adjacency  # fixed [N, N] mask, or None (no communication)

        t = config.training
        self.net = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.agent.hidden_dim,
            obs_embedding_dim=config.agent.obs_embedding_dim,
            message_dim=config.communication.message_dim,
            recurrent=False,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=t.lr, eps=1e-5)

        # PPO hyper-parameters
        self.gamma = t.gamma
        self.gae_lambda = t.gae_lambda
        self.clip_coef = t.clip_coef
        self.entropy_coef = t.entropy_coef
        self.value_coef = t.value_coef
        self.max_grad_norm = t.max_grad_norm
        self.update_epochs = t.update_epochs
        self.minibatch_size = t.minibatch_size

    # -- acting (rollout) --------------------------------------------------
    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(actions, log_probs, values)`` for one joint state ``[N, obs]``."""
        logits, value, _ = self.net(obs.unsqueeze(0), self.adjacency)
        logits, value = logits[0], value[0]
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``V(obs)`` for one joint state ``[N, obs]`` (bootstrap helper)."""
        _, value, _ = self.net(obs.unsqueeze(0), self.adjacency)
        return value[0]

    # -- learning (update) -------------------------------------------------
    def _evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, _ = self.net(obs, self.adjacency)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def update(self, buffer: Any) -> dict[str, float]:
        """Run PPO epochs over the buffer; return mean scalar metrics."""
        data = buffer.get()
        obs, actions = data["obs"], data["actions"]
        old_log_probs, advantages, returns = (
            data["log_probs"],
            data["advantages"],
            data["returns"],
        )
        num_steps = obs.shape[0]
        # Minibatch over timesteps (agents stay grouped so comm can be recomputed).
        mb_steps = max(1, self.minibatch_size // self.num_agents)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _ in range(self.update_epochs):
            order = np.random.permutation(num_steps)
            for start in range(0, num_steps, mb_steps):
                idx = torch.as_tensor(order[start : start + mb_steps], device=self.device)
                mb_obs, mb_actions = obs[idx], actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv, mb_ret = advantages[idx], returns[idx]

                # Per-minibatch advantage normalisation.
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logp, entropy, value = self._evaluate(mb_obs, mb_actions)
                ratio = (new_logp - mb_old_logp).exp()

                pg_unclipped = -mb_adv * ratio
                pg_clipped = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(pg_unclipped, pg_clipped).mean()
                value_loss = 0.5 * (value - mb_ret).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += entropy_loss.item()
                    stats["approx_kl"] += (mb_old_logp - new_logp).mean().item()
                n_updates += 1

        return {k: v / max(1, n_updates) for k, v in stats.items()}


__all__ = ["SharedPPOLearner"]
