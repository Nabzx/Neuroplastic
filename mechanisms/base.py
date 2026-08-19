"""Mechanism interface: how a plasticity mechanism hooks into training.

A mechanism observes the network's activations (if it needs per-unit utility) and
acts *after* each optimiser step -- rescaling weights (homeostasis), resetting
low-utility units (structural plasticity), etc. ``"vanilla"`` is the no-op
baseline.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from core.registry import Registry

MECHANISM_REGISTRY: Registry = Registry("mechanism")


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    """All ``nn.Linear`` sub-modules of ``model`` (weights the mechanisms act on)."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


class Mechanism:
    """Base class for plasticity mechanisms (default = no-op)."""

    #: Set True if the mechanism needs the forward-pass hidden activations.
    requires_activations: bool = False

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def observe(self, activations: list[torch.Tensor]) -> None:
        """Accumulate per-unit statistics (utility) from a step's activations."""

    def after_optimizer_step(self, model: nn.Module, step_index: int) -> None:
        """Modify the model in place after an optimiser step."""


@MECHANISM_REGISTRY.register("vanilla")
class Vanilla(Mechanism):
    """No intervention -- the loss-of-plasticity baseline."""


def make_mechanism(name: str, config: Any = None) -> Mechanism | None:
    """Construct a mechanism by name; ``None`` / ``"vanilla"`` -> plain training.

    Importing the concrete mechanisms here registers them.
    """
    if name in (None, "vanilla", "none"):
        return None
    import mechanisms.biological  # noqa: F401  (registration side-effect)
    import mechanisms.baselines  # noqa: F401

    return MECHANISM_REGISTRY.get(name)(config)


__all__ = ["Mechanism", "Vanilla", "MECHANISM_REGISTRY", "make_mechanism"]
