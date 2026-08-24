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
  exact per-unit learning rate for **any** optimiser (SGD and Adam alike).
* **ConsolidationMetaplasticity** (foil) — the forgetting-oriented direction: protect
  large, settled weights by *lowering* their plasticity. Honest negative control.

**The zero-gradient limitation and the hybrid fix.** In the study, MetaplasticLR fails
to prevent loss of plasticity: a *dead* ReLU unit has (near-)zero gradient, so scaling
its learning rate scales zero — the units it most needs to revive are exactly the ones
it cannot move. The remedy is to pair the metaplastic *sensing* of dormancy with a
**gradient-free actuator** that can cross that barrier:

* **IntrinsicPlasticity** — homeostatic regulation of each unit's *excitability* (its
  bias / firing threshold), the biological *intrinsic* plasticity of Desai & Turrigiano
  — distinct from synaptic scaling. Raising a dead unit's bias lifts its pre-activation
  back above threshold, so its ReLU fires again and gradient flow is restored.
* **MetaplasticReactivation** = MetaplasticLR + IntrinsicPlasticity (the hybrid).
* **MetaplasticStructural** = MetaplasticLR + structural reset (discrete revival).
* **MetaplasticHomeostatic** = MetaplasticLR + homeostatic synaptic scaling.

Update-rescaling mechanisms use ``before_optimizer_step`` (snapshot the weights) with
``after_optimizer_step`` (rescale the realised delta). ``before_optimizer_step`` is a
no-op on every other mechanism.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mechanisms.base import MECHANISM_REGISTRY, Mechanism


class _ActivitySetPoint:
    """Shared helper: a per-unit activity EMA and a per-layer set-point ``a*``.

    Both the learning-rate rule and intrinsic plasticity are driven by how far a
    unit's recent activity has drifted from the activity it had at initialisation.
    """

    def __init__(self, decay: float, eps: float) -> None:
        self.decay = decay
        self.eps = eps
        self.act: list[torch.Tensor] | None = None        # per-layer activity EMA (per unit)
        self.setpoint: list[torch.Tensor] | None = None   # per-layer scalar set-point a*

    def update(self, activations: list[torch.Tensor]) -> None:
        if self.act is None:  # set-point = mean activity at first sight (~initialisation)
            self.act = [a.abs().mean(dim=0).detach().to("cpu") for a in activations]
            self.setpoint = [a.mean().clamp_min(self.eps).clone() for a in self.act]
            return
        for i, act in enumerate(activations):
            mean_act = act.abs().mean(dim=0).detach().to("cpu")
            self.act[i].mul_(self.decay).add_((1.0 - self.decay) * mean_act)


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
        self.eps = float(c.get("eps", 1e-8))
        self._sp = _ActivitySetPoint(float(c.get("ema_decay", 0.99)), self.eps)
        self._snap: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None

    def observe(self, activations: list[torch.Tensor]) -> None:
        self._sp.update(activations)

    def layer_gain(self, i: int) -> torch.Tensor:
        """Per-unit gain vector for hidden layer ``i`` (on CPU)."""
        assert self._sp.act is not None and self._sp.setpoint is not None
        ratio = self._sp.setpoint[i] / (self._sp.act[i] + self.eps)
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
        if self._snap is None or self._sp.act is None:
            return
        for i, layer in enumerate(model.hidden_layers):
            w_snap, b_snap = self._snap[i]
            g = self.layer_gain(i).to(layer.weight.device, layer.weight.dtype)
            layer.weight.copy_(w_snap + g.unsqueeze(1) * (layer.weight - w_snap))
            if b_snap is not None:
                layer.bias.copy_(b_snap + g * (layer.bias - b_snap))
        self._snap = None


@MECHANISM_REGISTRY.register("intrinsic")
class IntrinsicPlasticity(Mechanism):
    """Homeostatic regulation of unit *excitability* (bias) toward an activity set-point.

    The biological *intrinsic* plasticity of Desai & Turrigiano — a neuron adjusts its
    own firing threshold to keep its mean activity near a target, independent of its
    synapses. Here: ``b_i += rate * (a* - a_i)`` — an under-active (dormant) unit has
    its bias *raised* until its ReLU fires again, a hyper-active unit has it lowered.
    This is a **gradient-free** revival: it works precisely on the dead units whose
    zero gradient defeats learning-rate modulation.
    """

    requires_activations = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.rate = float(c.get("rate", 0.01))
        self.eps = float(c.get("eps", 1e-8))
        # dead_threshold > 0 => act ONLY on units below (threshold * set-point): revive the
        # genuinely dead ones and leave healthy sparse units untouched (preserve selectivity).
        # 0 => drive every unit toward the set-point (homogenising; hurts, per the study).
        self.dead_threshold = float(c.get("dead_threshold", 0.0))
        self._sp = _ActivitySetPoint(float(c.get("ema_decay", 0.99)), self.eps)

    def observe(self, activations: list[torch.Tensor]) -> None:
        self._sp.update(activations)

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        if self._sp.act is None:
            return
        for i, layer in enumerate(model.hidden_layers):
            if layer.bias is None:
                continue
            gap = self._sp.setpoint[i] - self._sp.act[i]       # positive where under-active
            if self.dead_threshold > 0.0:
                dead = self._sp.act[i] < self.dead_threshold * self._sp.setpoint[i]
                gap = gap * dead.to(gap.dtype)                 # only revive genuinely dead units
            layer.bias.add_(self.rate * gap.to(layer.bias.device, layer.bias.dtype))


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


class _Composite(Mechanism):
    """Run several mechanisms in order, dispatching every hook to each part."""

    requires_activations = True
    parts: list[Mechanism]

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


@MECHANISM_REGISTRY.register("metaplastic_homeostatic")
class MetaplasticHomeostatic(_Composite):
    """Metaplastic learning-rate modulation + homeostatic synaptic scaling (HM4)."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        from mechanisms.biological import HomeostaticScaling  # local import: avoid cycle

        c = config or {}
        self.parts = [MetaplasticLR(c.get("metaplastic")), HomeostaticScaling(c.get("homeostatic"))]


@MECHANISM_REGISTRY.register("metaplastic_reactivation")
class MetaplasticReactivation(_Composite):
    """Hybrid fix: metaplastic learning-rate boost + intrinsic (bias) reactivation.

    Intrinsic plasticity revives dead units via a gradient-free bias nudge; the
    metaplastic learning-rate boost then lets the reawakened unit relearn quickly.
    Together they target the zero-gradient failure mode of MetaplasticLR alone.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        c = config or {}
        self.parts = [MetaplasticLR(c.get("metaplastic")), IntrinsicPlasticity(c.get("intrinsic"))]


@MECHANISM_REGISTRY.register("metaplastic_structural")
class MetaplasticStructural(_Composite):
    """Hybrid: metaplastic learning-rate boost + discrete structural reset of dead units."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        from mechanisms.biological import StructuralPlasticity  # local import: avoid cycle

        c = config or {}
        self.parts = [MetaplasticLR(c.get("metaplastic")), StructuralPlasticity(c.get("structural"))]


@MECHANISM_REGISTRY.register("selective_intrinsic")
class SelectiveIntrinsic(IntrinsicPlasticity):
    """Intrinsic plasticity that revives *only genuinely dead* units (preserves selectivity).

    Plain intrinsic plasticity drives every unit to the activity set-point, homogenising
    the representation and hurting the task. Restricting the bias nudge to units far below
    the set-point revives dead capacity without disturbing healthy sparse, selective units.
    """

    def __init__(self, config: Any = None) -> None:
        merged = {"dead_threshold": 0.15, "rate": 0.02}
        merged.update(config or {})
        super().__init__(merged)


@MECHANISM_REGISTRY.register("selective_intrinsic_homeostatic")
class SelectiveIntrinsicHomeostatic(_Composite):
    """Norm control (homeostatic) + targeted dead-unit revival (selective intrinsic).

    The winning ingredients, gradient-free: homeostatic scaling keeps weight norms near
    their set-point (the study's strongest single mechanism) while selective intrinsic
    plasticity recovers only the dead units, via bias rather than a discrete reset.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        from mechanisms.biological import HomeostaticScaling  # local import: avoid cycle

        c = config or {}
        self.parts = [SelectiveIntrinsic(c.get("intrinsic")), HomeostaticScaling(c.get("homeostatic"))]


@MECHANISM_REGISTRY.register("metaplastic_combined")
class MetaplasticCombined(_Composite):
    """Metaplastic learning-rate boost on top of the combined winner (homeostatic + structural).

    Structural reset revives dead units (nonzero gradient again) and homeostatic scaling
    controls norms; the metaplastic boost then accelerates the relearning of the young,
    low-activity units — applied where units are alive-but-young, not dead (so it dodges
    the zero-gradient limitation that sinks metaplasticity alone).
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        from mechanisms.biological import HomeostaticScaling, StructuralPlasticity  # avoid cycle

        c = config or {}
        self.parts = [
            MetaplasticLR(c.get("metaplastic")),
            HomeostaticScaling(c.get("homeostatic")),
            StructuralPlasticity(c.get("structural")),
        ]


__all__ = [
    "MetaplasticLR",
    "IntrinsicPlasticity",
    "SelectiveIntrinsic",
    "ConsolidationMetaplasticity",
    "MetaplasticHomeostatic",
    "MetaplasticReactivation",
    "MetaplasticStructural",
    "SelectiveIntrinsicHomeostatic",
    "MetaplasticCombined",
]
