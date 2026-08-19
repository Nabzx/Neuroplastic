"""Baseline remedies for loss of plasticity (for positioning against the hero).

* **L2 weight decay** — the classic implicit regulariser (bounds weights).
* **Shrink-and-perturb** (Ash & Adams 2020) — periodically shrink weights and add
  noise.
* **Continual Backprop** (Dohare et al. 2024) — the state-of-the-art remedy;
  continual utility-based unit replacement (a config of structural plasticity,
  no maturation protection).
* **ReDo** (Sokar et al. 2023) — periodically recycle *dormant* units.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mechanisms.base import MECHANISM_REGISTRY, Mechanism, linear_layers
from mechanisms.biological import StructuralPlasticity


@MECHANISM_REGISTRY.register("l2")
class L2WeightDecay(Mechanism):
    """Multiplicative weight decay each step (equivalent to L2 regularisation)."""

    requires_activations = False

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.decay = float((config or {}).get("decay", 1e-4))

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        for layer in linear_layers(model):
            layer.weight.mul_(1.0 - self.decay)


@MECHANISM_REGISTRY.register("shrink_perturb")
class ShrinkAndPerturb(Mechanism):
    """Periodically ``w <- shrink * w + noise``."""

    requires_activations = False

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.shrink = float(c.get("shrink", 0.9))
        self.noise_std = float(c.get("noise_std", 0.01))
        self.period = int(c.get("period", 200))

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        if step_index % self.period != 0:
            return
        for layer in linear_layers(model):
            layer.weight.mul_(self.shrink).add_(self.noise_std * torch.randn_like(layer.weight))


@MECHANISM_REGISTRY.register("continual_backprop")
class ContinualBackprop(StructuralPlasticity):
    """SOTA remedy: continual utility-based unit replacement (Dohare et al.)."""

    def __init__(self, config: Any = None) -> None:
        merged = {"selection": "utility", "maturity": 100, "replacement_rate": 1e-3, "period": 100}
        merged.update(config or {})
        super().__init__(merged)


@MECHANISM_REGISTRY.register("redo")
class ReDo(StructuralPlasticity):
    """Recycle dormant units periodically (Sokar et al.)."""

    def __init__(self, config: Any = None) -> None:
        merged = {"selection": "dormant", "dormant_tau": 0.1, "maturity": 100, "period": 1000}
        merged.update(config or {})
        super().__init__(merged)


__all__ = ["L2WeightDecay", "ShrinkAndPerturb", "ContinualBackprop", "ReDo"]
