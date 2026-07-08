"""Adaptive graph-based communication (PyTorch).

Where the fixed baselines aggregate messages with a *constant, uniform* mask,
this module computes a **dynamic, weighted edge matrix** from the agents' current
states -- i.e. attention over the interaction graph. Each step it produces a
weight ``W[i, j]`` (how much receiver ``i`` attends to sender ``j``) and returns
the weighted aggregation of incoming messages.

This is the "adaptive weighted communication" infrastructure only: the weights
are a differentiable function of agent embeddings, learned end-to-end through the
policy loss. It is deliberately **not** Hebbian plasticity (which would update a
persistent weight matrix from activity correlations over time); that comes later
and can reuse the same edge-matrix representation exposed here.

Shapes
------
``embeddings``: ``[B, N, E]`` · ``messages``: ``[B, N, M]`` ·
``structural_mask``: ``[N, N]`` binary receiver-by-sender (which edges may exist)
or ``None`` (complete graph). Returns ``context [B, N, M]`` and ``weights
[B, N, N]`` (rows sum to 1 over each receiver's allowed senders).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_MASK_FILL = -1.0e9  # effectively -inf for softmax, but finite (avoids NaN grads)


class AdaptiveCommunication(nn.Module):
    """Scaled dot-product attention over the interaction graph.

    Parameters
    ----------
    embedding_dim:
        Size of the per-agent embedding used to derive queries/keys.
    message_dim:
        Size of the message vectors being aggregated.
    attention_dim:
        Internal query/key dimensionality.
    """

    def __init__(self, embedding_dim: int, message_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.message_dim = message_dim
        self.attention_dim = attention_dim
        self.query = nn.Linear(embedding_dim, attention_dim)
        self.key = nn.Linear(embedding_dim, attention_dim)
        for layer in (self.query, self.key):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def edge_weights(
        self, embeddings: torch.Tensor, structural_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the adaptive edge-weight matrix ``W`` of shape ``[B, N, N]``.

        ``W[b, i, j]`` is the (softmax-normalised) weight from sender ``j`` to
        receiver ``i``. Self-loops and structurally-absent edges are zeroed.
        """
        query = self.query(embeddings)                       # [B, N, A]
        key = self.key(embeddings)                           # [B, N, A]
        scores = torch.einsum("bid,bjd->bij", query, key)    # [B, N, N]
        scores = scores / (self.attention_dim**0.5)

        n = scores.shape[-1]
        self_loops = torch.eye(n, dtype=torch.bool, device=scores.device).unsqueeze(0)
        disallowed = self_loops
        if structural_mask is not None:
            allowed = structural_mask.to(torch.bool).unsqueeze(0)  # [1, N, N]
            disallowed = disallowed | ~allowed

        scores = scores.masked_fill(disallowed, _MASK_FILL)
        weights = torch.softmax(scores, dim=-1)              # over senders j

        # Receivers with no allowed senders would softmax to NaN -> zero them.
        has_sender = (~disallowed).any(dim=-1, keepdim=True)
        weights = torch.where(has_sender, weights, torch.zeros_like(weights))
        return weights

    def forward(
        self,
        embeddings: torch.Tensor,
        messages: torch.Tensor,
        structural_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context, weights)``: weighted message aggregation + the weights."""
        weights = self.edge_weights(embeddings, structural_mask)
        context = torch.einsum("bij,bjm->bim", weights, messages)
        return context, weights


__all__ = ["AdaptiveCommunication"]
