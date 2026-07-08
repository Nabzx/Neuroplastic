"""Training-dynamics plots: reward / metric curves and hub centrality over time.

Matplotlib only. Matplotlib is imported lazily inside each function so importing
this module never requires a plotting backend; callers running headless should
select the ``Agg`` backend (the scripts do).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _new_ax(ax: Any | None, figsize: tuple[float, float] = (7.0, 4.0)) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _smooth_xy(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing moving average; returns the aligned ``(x, y)`` for the smoothed core."""
    window = max(1, min(window, len(y)))
    if window == 1:
        return x, y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    return x[window - 1:], smoothed


def plot_training_curves(
    runs: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    ax: Any | None = None,
    xlabel: str = "environment steps",
    ylabel: str = "value",
    title: str | None = None,
    smooth: int = 1,
) -> Any:
    """Overlay one ``(steps, values)`` curve per run/label on a shared axis.

    NaNs are masked out. With ``smooth > 1`` the raw curve is drawn faintly and a
    moving-average line on top.
    """
    ax = _new_ax(ax)
    for label, (steps, values) in runs.items():
        x = np.asarray(steps, dtype=float)
        y = np.asarray(values, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]
        if smooth > 1 and len(y) > smooth:
            ax.plot(x, y, alpha=0.25, linewidth=1)
            xs, ys = _smooth_xy(x, y, smooth)
            ax.plot(xs, ys, linewidth=2, label=label)
        else:
            ax.plot(x, y, linewidth=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_centrality_over_time(
    steps: Sequence[float],
    centralities: np.ndarray,
    agent_ids: Sequence[str],
    ax: Any | None = None,
    ylabel: str = "hub strength (out-degree)",
    title: str | None = "Hub formation over time",
) -> Any:
    """One line per agent showing how its centrality evolves over training.

    ``centralities`` has shape ``[T, N]`` (one column per agent). Divergence of a
    line above the others indicates a communication hub forming.
    """
    ax = _new_ax(ax)
    x = np.asarray(steps, dtype=float)
    values = np.asarray(centralities, dtype=float)
    for j, agent in enumerate(agent_ids):
        ax.plot(x, values[:, j], linewidth=1.5, label=agent)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return ax


__all__ = ["plot_training_curves", "plot_centrality_over_time"]
