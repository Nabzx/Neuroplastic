"""Metaplasticity — activity-history modulation of each synapse's learning rate.

Metaplasticity (Abraham & Bear, 1996) is the *plasticity of plasticity*: a synapse's
recent activity history sets its *future* capacity to change, rather than its current
weight. In machine learning this idea has been used almost entirely for the opposite
problem to ours — catastrophic *forgetting* (EWC, Synaptic Intelligence, binarized-net
metaplasticity), where consolidation is *raised* to *protect* important weights. Loss
of plasticity needs the inverse pressure: *increase* the effective learning rate of
dormant / saturated units so gradient can move them back into their active regime.

* **MetaplasticLR** (hero) — a BCM sliding-threshold rule. Each hidden unit's activity
  EMA sets a per-unit gain that *boosts* the learning rate of under-active units and
  *damps* hyper-active ones. It rescales the *realised* weight update, so it is an
  exact per-unit learning rate for **any** optimiser (SGD and Adam alike) — a smooth,
  per-synapse alternative to the discrete unit resets of Continual Backprop / ReDo.
* **ConsolidationMetaplasticity** (foil) — the forgetting-oriented direction: protect
  large, settled weights by *lowering* their plasticity. Included as an honest
  negative control (we predict it does not cure loss of plasticity).
* **MetaplasticHomeostatic** — metaplasticity + homeostatic scaling, to test whether
  activity-gated learning rates stack with set-point weight rescaling.

All three rescale the update applied by the optimiser, so they use ``before_optimizer_step``
(snapshot the weights) together with ``after_optimizer_step`` (rescale the realised
delta). ``before_optimizer_step`` is a no-op on every other mechanism.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mechanisms.base import MECHANISM_REGISTRY, Mechanism


@MECHANISM_REGISTRY.register("metaplastic")
class MetaplasticLR(Mechanism):
    """Per-unit metaplastic learning-rate modulation (BCM sliding threshold).

    Maintains an EMA ``a_i`` of each hidden unit's activity and a set-point ``a*``
    (the layer's mean activity at first sight). The per-unit gain

        ``g_i = clip( (a* / (a_i + eps)) ** beta,  g_min,  g_max )``

    scales the realised update of that unit's *incoming* synapses (weight row + bias):
    ``g_i > 1`` re-awakens under-active units, ``g_i < 1`` damps hyper-active ones.
    """

    requires_activations = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.beta = float(c.get("beta", 1.0))
        self.g_min = float(c.get("g_min", 0.5))
        self.g_max = float(c.get("g_max", 5.0))
        self.decay = float(c.get("ema_decay", 0.99))
        self.eps = float(c.get("eps", 1e-8))
        self._act: list[torch.Tensor] | None = None        # per-layer activity EMA (per unit)
        self._setpoint: list[torch.Tensor] | None = None   # per-layer scalar set-point a*
        self._snap: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None

    def observe(self, activations: list[torch.Tensor]) -> None:
        if self._act is None:  # set-point = mean activity at first sight (~initialisation)
            self._act = [a.abs().mean(dim=0).detach().to("cpu") for a in activations]
            self._setpoint = [a.mean().clamp_min(self.eps).clone() for a in self._act]
            return
        for i, act in enumerate(activations):
            mean_act = act.abs().mean(dim=0).detach().to("cpu")
            self._act[i].mul_(self.decay).add_((1.0 - self.decay) * mean_act)

    def layer_gain(self, i: int) -> torch.Tensor:
        """Per-unit gain vector for hidden layer ``i`` (on CPU)."""
        assert self._act is not None and self._setpoint is not None
        ratio = self._setpoint[i] / (self._act[i] + self.eps)
        return ratio.pow(self.beta).clamp(self.g_min, self.g_max)

    @torch.no_grad()
    def before_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        self._snap = [
            (layer.weight.detach().clone(),
             None if layer.bias is None else layer.bias.detach().clone())
            for layer in model.hidden_layers
        ]

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        if self._snap is None or self._act is None:
            return
        for i, layer in enumerate(model.hidden_layers):
            w_snap, b_snap = self._snap[i]
            g = self.layer_gain(i).to(layer.weight.device, layer.weight.dtype)
            layer.weight.copy_(w_snap + g.unsqueeze(1) * (layer.weight - w_snap))
            if b_snap is not None:
                layer.bias.copy_(b_snap + g * (layer.bias - b_snap))
        self._snap = None


@MECHANISM_REGISTRY.register("metaplastic_consolidation")
class ConsolidationMetaplasticity(Mechanism):
    """Forgetting-direction foil: protect large, settled weights (predicted not to help).

    Each weight accrues a consolidation state ``c`` (EMA of its magnitude); the realised
    update is scaled by ``1 / (1 + lambda * c)``, so consolidated weights become *less*
    plastic. This is the mechanism behind metaplastic anti-forgetting methods — the
    opposite pressure to what loss of plasticity needs. Included only to test HM5.
    """

    requires_activations = False

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.lam = float(c.get("lambda", 1.0))
        self.decay = float(c.get("ema_decay", 0.99))
        self._c: list[torch.Tensor] | None = None
        self._snap: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None

    @torch.no_grad()
    def before_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        self._snap = [
            (layer.weight.detach().clone(),
             None if layer.bias is None else layer.bias.detach().clone())
            for layer in model.hidden_layers
        ]

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        if self._snap is None:
            return
        if self._c is None:
            self._c = [torch.zeros_like(layer.weight) for layer in model.hidden_layers]
        for i, layer in enumerate(model.hidden_layers):
            w_snap, _ = self._snap[i]
            scale = 1.0 / (1.0 + self.lam * self._c[i])       # <= 1: consolidated -> protected
            layer.weight.copy_(w_snap + scale * (layer.weight - w_snap))
            self._c[i].mul_(self.decay).add_((1.0 - self.decay) * layer.weight.abs())
        self._snap = None


@MECHANISM_REGISTRY.register("metaplastic_homeostatic")
class MetaplasticHomeostatic(Mechanism):
    """Metaplastic learning-rate modulation + homeostatic synaptic scaling (HM4)."""

    requires_activations = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        from mechanisms.biological import HomeostaticScaling  # local import: avoid cycle

        c = config or {}
        # metaplastic rescales the applied delta first; homeostatic then pulls norms.
        self.parts: list[Mechanism] = [
            MetaplasticLR(c.get("metaplastic")),
            HomeostaticScaling(c.get("homeostatic")),
        ]

    def observe(self, activations: list[torch.Tensor]) -> None:
        for part in self.parts:
            if part.requires_activations:
                part.observe(activations)

    def before_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        for part in self.parts:
            part.before_optimizer_step(model, step_index)

    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        for part in self.parts:
            part.after_optimizer_step(model, step_index)


__all__ = ["MetaplasticLR", "ConsolidationMetaplasticity", "MetaplasticHomeostatic"]
