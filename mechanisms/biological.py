"""Biologically-inspired plasticity mechanisms (the study's hero methods).

* **Homeostatic synaptic scaling** — rescale each neuron's incoming weights
  toward a homeostatic set-point (its initial norm), bounding weight growth and
  keeping units in their operating range (Turrigiano).
* **Structural plasticity** — track per-unit utility; periodically *prune*
  low-utility (or dormant) units and *regrow* fresh ones (neurogenesis), with a
  maturation window protecting new units. Closely related to Continual Backprop
  (Dohare et al.) and ReDo (Sokar et al.); our additions are the maturation
  protection and composability with homeostasis.
* **Combined** — both together (the principled combination of H3).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from mechanisms.base import MECHANISM_REGISTRY, Mechanism, linear_layers


@MECHANISM_REGISTRY.register("homeostatic")
class HomeostaticScaling(Mechanism):
    """Multiplicative synaptic scaling toward each row's initial weight norm."""

    requires_activations = False

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.rate = float(c.get("rate", 0.1))
        self.every = int(c.get("every", 1))
        self._targets: list[torch.Tensor] | None = None

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        layers = linear_layers(model)
        if self._targets is None:  # set-point = norm at first sight (~initialisation)
            self._targets = [layer.weight.norm(dim=1, keepdim=True).clamp_min(1e-8).clone() for layer in layers]
        if step_index % self.every != 0:
            return
        for layer, target in zip(layers, self._targets):
            norm = layer.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
            scale = (1.0 - self.rate) + self.rate * (target / norm)
            layer.weight.mul_(scale)


@MECHANISM_REGISTRY.register("structural")
class StructuralPlasticity(Mechanism):
    """Utility-gated pruning + neurogenesis on hidden units.

    Expects an MLP exposing ``hidden_layers`` (ModuleList) and ``head``. Tracks an
    EMA of each hidden unit's mean activation as its utility; every ``period``
    steps, resets the least-useful mature units (``selection="utility"``) or all
    mature dormant units (``selection="dormant"``): incoming weights are
    re-initialised (a fresh "young" unit) and outgoing weights zeroed so the new
    unit starts neutral, then relearns.
    """

    requires_activations = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.replacement_rate = float(c.get("replacement_rate", 1e-3))
        self.period = int(c.get("period", 100))
        self.maturity = int(c.get("maturity", 500))
        self.selection = str(c.get("selection", "utility"))
        self.dormant_tau = float(c.get("dormant_tau", 0.1))
        self.decay = float(c.get("utility_decay", 0.99))
        self._util: list[torch.Tensor] | None = None
        self._age: list[torch.Tensor] | None = None
        self._steps = 0

    def observe(self, activations: list[torch.Tensor]) -> None:
        if self._util is None:
            self._util = [torch.zeros(a.shape[1]) for a in activations]
            self._age = [torch.zeros(a.shape[1]) for a in activations]
        for i, act in enumerate(activations):
            mean_act = act.abs().mean(dim=0).cpu()
            self._util[i].mul_(self.decay).add_((1.0 - self.decay) * mean_act)
            self._age[i] += 1

    def _select(self, util: torch.Tensor, mature: torch.Tensor, n_units: int) -> torch.Tensor:
        if self.selection == "dormant":
            score = util / util.mean().clamp_min(1e-9)
            return torch.nonzero((score <= self.dormant_tau) & mature).flatten()
        n_reset = int(round(self.replacement_rate * n_units * self.period))
        if n_reset == 0:
            return torch.tensor([], dtype=torch.long)
        masked = util.clone()
        masked[~mature] = float("inf")                      # protect immature units
        order = torch.argsort(masked)
        chosen = order[:n_reset]
        return chosen[torch.isfinite(masked[chosen])]

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        self._steps += 1
        if self._util is None or self._steps % self.period != 0:
            return
        hidden = model.hidden_layers
        for i, incoming in enumerate(hidden):
            outgoing = hidden[i + 1] if i + 1 < len(hidden) else model.head
            util, age = self._util[i], self._age[i]
            to_reset = self._select(util, age >= self.maturity, incoming.out_features)
            for j in to_reset.tolist():
                nn.init.kaiming_uniform_(incoming.weight[j : j + 1], a=math.sqrt(5))
                if incoming.bias is not None:
                    incoming.bias[j] = 0.0
                outgoing.weight[:, j] = 0.0                  # new unit starts neutral
                util[j] = util.mean()
                age[j] = 0.0


@MECHANISM_REGISTRY.register("combined")
class Combined(Mechanism):
    """Homeostatic scaling + structural plasticity together."""

    requires_activations = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.parts: list[Mechanism] = [
            HomeostaticScaling(c.get("homeostatic")),
            StructuralPlasticity(c.get("structural")),
        ]

    def observe(self, activations: list[torch.Tensor]) -> None:
        for part in self.parts:
            if part.requires_activations:
                part.observe(activations)

    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        for part in self.parts:
            part.after_optimizer_step(model, step_index)


__all__ = ["HomeostaticScaling", "StructuralPlasticity", "Combined"]
