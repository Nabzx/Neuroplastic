"""Hebbian-inspired plastic communication edges (PyTorch, training path).

A persistent edge-weight matrix ``P[i, j]`` (receiver ``i`` <- sender ``j``) that
is **not** learned by backprop but by a simple, reward-gated Hebbian rule -- the
neuroplastic mechanism the project is named for. It is the training-loop
counterpart of the numpy reference rule in :mod:`plasticity.hebbian` (analysis
path) and plugs into the same communication substrate as
:class:`communication.adaptive.AdaptiveCommunication`.

The rule (per plasticity update, once the coefficients are in hand)::

    dP = lr * modulation * coactivity - decay * P        # Hebbian + weight decay
    P  = clamp(P + dP, 0, max_weight)                     # stability
    P  = P * structural_mask                              # keep candidate edges
    if homeostasis: rescale each receiver's incoming weights to <= max_weight

where

* ``coactivity[i, j]`` = mean cosine similarity between agents ``i`` and ``j``'s
  messages over the rollout ("do they communicate in a correlated way?"),
* ``modulation`` = a scalar success signal (reward relative to a running
  baseline), so edges strengthen during *successful* coordination and weaken
  when coordination is unhelpful,
* ``decay`` forgets unused edges (their coactivity ~ 0, so only decay acts),
* ``clamp`` + optional homeostasis keep the weights bounded.

Messages are still learned by the policy loss; only ``P`` is plastic. Aggregation
uses the row-normalised ``P`` so the message context stays a stable weighted
average regardless of the absolute weight scale.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_EPS = 1e-8


class PlasticEdges(nn.Module):
    """A persistent, Hebbian-updated weighted communication graph.

    Parameters
    ----------
    num_agents:
        Number of agents ``N``.
    structural_mask:
        Binary ``[N, N]`` receiver-by-sender mask of candidate edges, or ``None``
        for a complete graph. Non-candidate edges stay at zero forever.
    config:
        The ``plasticity`` config section (supplies the coefficients).
    """

    def __init__(self, num_agents: int, structural_mask: torch.Tensor | None, config: Any) -> None:
        super().__init__()
        self.learning_rate = float(getattr(config, "learning_rate", 0.1))
        self.decay = float(getattr(config, "decay", 0.01))
        self.max_weight = float(getattr(config, "max_weight", 2.0))
        self.homeostasis = bool(getattr(config, "homeostasis", True))
        self.modulation_mode = str(getattr(config, "modulation", "reward_gated"))
        self.baseline_momentum = float(getattr(config, "baseline_momentum", 0.05))

        if structural_mask is None:
            mask = torch.ones(num_agents, num_agents)
            mask.fill_diagonal_(0.0)
        else:
            mask = structural_mask.clone().float()
            mask.fill_diagonal_(0.0)

        # Persistent state (buffers move with .to(device) and are checkpointed).
        self.register_buffer("mask", mask)
        self.register_buffer("edge_weights", mask.clone())  # start ~ uniform over candidates
        self.register_buffer("reward_baseline", torch.tensor(float("nan")))

    # -- message aggregation ----------------------------------------------
    def aggregation_weights(self) -> torch.Tensor:
        """Row-normalised ``[N, N]`` weights used to aggregate messages."""
        weights = self.edge_weights * self.mask
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Aggregate ``messages`` ``[B, N, M]`` with the current edge weights."""
        return torch.einsum("ij,bjm->bim", self.aggregation_weights(), messages)

    # -- Hebbian update ----------------------------------------------------
    def _modulation(self, reward: float) -> float:
        """Scalar success signal: reward relative to a running baseline."""
        if self.modulation_mode == "none":
            return 1.0
        if torch.isnan(self.reward_baseline):
            self.reward_baseline.fill_(reward)
        modulation = reward - float(self.reward_baseline)
        self.reward_baseline.add_(self.baseline_momentum * (reward - self.reward_baseline))
        return float(max(-1.0, min(1.0, modulation)))  # bounded for stability

    @torch.no_grad()
    def hebbian_update(self, coactivity: torch.Tensor, reward: float) -> dict[str, float]:
        """Apply one Hebbian update from ``coactivity`` and ``reward``.

        Returns scalar statistics for logging.
        """
        modulation = self._modulation(reward)
        delta = self.learning_rate * modulation * coactivity - self.decay * self.edge_weights

        self.edge_weights.add_(delta)
        self.edge_weights.clamp_(min=0.0, max=self.max_weight)
        self.edge_weights.mul_(self.mask)

        if self.homeostasis:
            # Synaptic scaling: cap each receiver's total incoming weight.
            row_sum = self.edge_weights.sum(dim=-1, keepdim=True)
            scale = (self.max_weight / row_sum.clamp_min(_EPS)).clamp(max=1.0)
            self.edge_weights.mul_(scale)

        active = self.mask > 0
        return {
            "plast_modulation": modulation,
            "plast_mean_weight": float(self.edge_weights[active].mean()) if active.any() else 0.0,
            "plast_max_weight": float(self.edge_weights.max()),
            "plast_update_norm": float(delta[active].abs().mean()) if active.any() else 0.0,
        }

    def current_matrix(self) -> torch.Tensor:
        """A detached copy of the raw plastic edge-weight matrix."""
        return self.edge_weights.detach().clone()


__all__ = ["PlasticEdges"]
