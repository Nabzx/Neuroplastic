"""Task-level coordination outcomes.

These summarise *whether the collective actually coordinated*, complementing the
structural (graph) and informational measures. ``episode_return`` is functional;
task-specific success and the coordination index depend on recorded rollouts /
environment info and are deferred.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.registry import Registry
from core.types import AgentID, ArrayLike

COORDINATION_METRICS: Registry = Registry("coordination_metric")


def episode_return(reward_stream: Sequence[Mapping[AgentID, float]]) -> float:
    """Total team return: sum of all agents' rewards over an episode."""
    return float(sum(sum(step.values()) for step in reward_stream))


def success_rate(episode_infos: Sequence[Mapping[str, object]]) -> float:  # pragma: no cover - deferred
    """Fraction of episodes flagged successful by the environment (placeholder)."""
    raise NotImplementedError(
        "success_rate depends on per-benchmark success signals; specified in "
        "docs/experiment_plan.md."
    )


def coordination_index(*args, **kwargs) -> float:  # pragma: no cover - deferred
    """A normalised measure of how much better the team does than independent
    agents on the same task (placeholder). See docs/experiment_plan.md."""
    raise NotImplementedError("coordination_index is a placeholder.")


COORDINATION_METRICS.register("episode_return", episode_return)
COORDINATION_METRICS.register("success_rate", success_rate)
COORDINATION_METRICS.register("coordination_index", coordination_index)


# --------------------------------------------------------------------------- #
# Learning-curve metrics (operate on a per-iteration reward series)
# --------------------------------------------------------------------------- #
def cumulative_reward(returns: ArrayLike) -> float:
    """Sum of the per-iteration mean returns -- area under the reward curve.

    A single number rewarding both *how high* and *how soon* performance rises
    (NaNs, e.g. iterations with no completed episode, are ignored).
    """
    return float(np.nansum(np.asarray(returns, dtype=float)))


def final_reward(returns: ArrayLike, window_frac: float = 0.1) -> float:
    """Mean return over the final ``window_frac`` of iterations (converged level)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    window = max(1, int(len(r) * window_frac))
    return float(r[-window:].mean())


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing-aligned moving average kept to the same length as ``x``."""
    window = max(1, min(window, len(x)))
    if window == 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def convergence_speed(
    returns: ArrayLike,
    steps: ArrayLike | None = None,
    threshold: float = 0.9,
    smooth: int = 5,
) -> dict[str, float]:
    """When (in env steps) the smoothed reward first reaches ``threshold`` of its gain.

    Returns ``{convergence_steps, convergence_fraction}`` where ``convergence_steps``
    is the env-step index at which the smoothed return first crosses
    ``baseline + threshold * (final - baseline)``. Lower = faster. Both are NaN if
    the run never improved.
    """
    r = np.asarray(returns, dtype=float)
    valid = ~np.isnan(r)
    r = r[valid]
    if r.size < 2:
        return {"convergence_steps": float("nan"), "convergence_fraction": float("nan")}
    s = np.asarray(steps, dtype=float)[valid] if steps is not None else np.arange(r.size, dtype=float)

    smoothed = _smooth(r, smooth)
    offset = len(r) - len(smoothed)  # index shift introduced by the 'valid' window
    baseline = smoothed[0]
    final = smoothed[-max(1, len(smoothed) // 10):].mean()
    improvement = final - baseline
    if improvement <= 1e-9:
        return {"convergence_steps": float("nan"), "convergence_fraction": float("nan")}

    target = baseline + threshold * improvement
    crossings = np.where(smoothed >= target)[0]
    idx = int(crossings[0]) + offset if crossings.size else len(r) - 1
    conv_steps = float(s[idx])
    return {
        "convergence_steps": conv_steps,
        "convergence_fraction": float(conv_steps / s[-1]) if s[-1] > 0 else float("nan"),
    }


__all__ = [
    "COORDINATION_METRICS",
    "episode_return",
    "success_rate",
    "coordination_index",
    "cumulative_reward",
    "final_reward",
    "convergence_speed",
]
