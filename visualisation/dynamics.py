"""Plot training dynamics: metric trajectories and plastic-weight evolution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def plot_metric_trajectories(
    trajectories: Mapping[str, Sequence[float]],
    steps: Sequence[int] | None = None,
    ax: Any | None = None,
):
    """Line-plot one or more metric trajectories on a shared axis.

    ``trajectories`` maps metric name -> per-snapshot values (as produced by
    :func:`analysis.topology_analysis.metric_trajectory`). Returns the Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "plot_metric_trajectories requires matplotlib. Install viz extras: "
            "`pip install -e .[viz]`."
        ) from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for name, values in trajectories.items():
        x = list(steps) if steps is not None else list(range(len(values)))
        ax.plot(x, list(values), label=name, marker="o", markersize=3)
    ax.set_xlabel("training snapshot")
    ax.set_ylabel("metric value")
    ax.legend(fontsize=8)
    return ax


__all__ = ["plot_metric_trajectories"]
