"""Neuromodulation of plasticity (the "third factor").

In three-factor learning rules, a global or agent-local modulatory signal (often
reward-related, analogous to dopamine) gates whether and how strongly Hebbian
co-activity translates into weight change. This module centralises that gating
factor.
"""

from __future__ import annotations

from typing import Any


def modulation_factor(config: Any, step: int, reward: float | None = None) -> float:
    """Return the scalar plasticity gate for this update.

    Modes (``config.modulation``):

    * ``"none"``         -- constant gate of ``1.0`` (pure Hebbian).
    * ``"reward_gated"`` -- gate scales with a supplied ``reward`` signal.
    * ``"three_factor"`` -- eligibility-trace / neuromodulated update.

    Only the ``"none"`` branch is fully implemented; the reward-driven modes
    depend on the training loop's reward stream and are completed in the
    training milestone. Until then they fall back to ``1.0`` so that Hebbian
    updates still run.
    """
    mode = getattr(config, "modulation", "none")
    if mode == "none":
        return 1.0
    if mode == "reward_gated":
        if reward is None:
            return 1.0  # reward stream not wired yet; see docs/experiment_plan.md
        return float(reward)
    if mode == "three_factor":  # pragma: no cover - deferred
        raise NotImplementedError(
            "three_factor modulation requires eligibility traces; specified in "
            "docs/experiment_plan.md."
        )
    raise ValueError(f"Unknown modulation mode: {mode!r}")


__all__ = ["modulation_factor"]
