"""Metrics that quantify loss of plasticity in a network.

These are the standard diagnostics from the plasticity-loss literature (Lyle et
al. 2023; Sokar et al. 2023; Dohare et al. 2024). Computed on a *fixed probe
batch* so they track the network's representational health over training,
independent of the current task:

* ``dormant_fraction``  -- fraction of hidden units that (almost) never fire,
* ``effective_rank``    -- effective dimensionality of the last-layer features,
* ``weight_magnitude``  -- mean absolute weight (grows as plasticity is lost).

A network losing plasticity shows rising dormant fraction, collapsing effective
rank and/or growing weights.
"""

from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def dormant_fraction(activations: Sequence[torch.Tensor], tau: float = 0.0) -> float:
    """Fraction of hidden units whose normalised mean activation is <= ``tau``.

    Following Sokar et al.'s dormant-neuron score: a unit's score is its mean
    absolute activation over the batch divided by the layer mean. ``tau=0`` counts
    only truly dead units; a small ``tau`` (e.g. 0.1) counts near-dormant ones too.
    """
    dormant = 0
    total = 0
    for act in activations:
        score = act.abs().mean(dim=0)                       # [units]
        denom = score.mean().clamp_min(1e-9)
        normalised = score / denom
        dormant += int((normalised <= tau).sum().item())
        total += act.shape[1]
    return dormant / max(1, total)


@torch.no_grad()
def effective_rank(features: torch.Tensor) -> float:
    """Roy-Vetterli effective rank of a ``[B, H]`` feature matrix.

    ``exp(entropy of the normalised singular-value distribution)`` -- a smooth
    measure of how many dimensions the representation actually uses. Collapse
    toward 1 signals a degenerate representation.
    """
    singular = torch.linalg.svdvals(features.float())
    singular = singular[singular > 1e-12]
    if singular.numel() == 0:
        return 0.0
    p = singular / singular.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def weight_magnitude(model: torch.nn.Module) -> float:
    """Mean absolute value over all weight matrices (biases excluded)."""
    total = 0.0
    count = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += float(param.abs().sum().item())
            count += param.numel()
    return total / max(1, count)


@torch.no_grad()
def representation_diagnostics(
    model: torch.nn.Module, probe: torch.Tensor, tau: float = 0.1
) -> dict[str, float]:
    """Run ``model`` on the probe batch and return all representation diagnostics."""
    was_training = model.training
    model.eval()
    _, activations = model(probe, return_activations=True)
    model.train(was_training)
    return {
        "dead_fraction": dormant_fraction(activations, tau=0.0),
        "dormant_fraction": dormant_fraction(activations, tau=tau),
        "effective_rank": effective_rank(activations[-1]),
        "weight_magnitude": weight_magnitude(model),
    }


__all__ = [
    "dormant_fraction",
    "effective_rank",
    "weight_magnitude",
    "representation_diagnostics",
]
