"""A small MLP that exposes its hidden activations.

The plasticity diagnostics (dormant units, effective rank) need the per-layer
hidden activations, so :meth:`MLP.forward` optionally returns them. Kept
deliberately simple and small -- the continual-learning benchmarks here are
designed to expose plasticity loss with modest networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward net with ReLU hidden layers and a linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev = input_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.head = nn.Linear(prev, output_dim)
        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the output; optionally also the list of hidden activations."""
        activations: list[torch.Tensor] = []
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            if return_activations:
                activations.append(h)
        out = self.head(h).squeeze(-1)
        if return_activations:
            return out, activations
        return out

    def hidden_units(self) -> int:
        return sum(layer.out_features for layer in self.hidden_layers)


__all__ = ["MLP"]
