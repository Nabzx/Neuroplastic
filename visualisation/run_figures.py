"""Generate the project's standard figure set from saved experiment logs.

Reads ``metrics.csv`` (per-iteration reward / graph density / communication
entropy) and ``edge_weights.npz`` (``[T, N, N]`` weight snapshots) from one or
more run directories and writes matplotlib figures to an output directory
(default ``results/figures``):

* ``reward_curves.png``               -- episode return vs. steps (all runs)
* ``graph_density.png``               -- communication graph density over training
* ``communication_entropy.png``       -- communication (weight) entropy over training
* ``edge_weight_heatmaps_<run>.png``  -- weight matrix at early / mid / late training
* ``graph_snapshots_<run>.png``       -- communication graph at early / mid / late
* ``hub_centrality_<run>.png``        -- per-agent out-strength (hub formation) over time

Everything is matplotlib-only and runs headless (Agg backend).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluation.report import load_metrics
from visualisation.dynamics import plot_centrality_over_time, plot_training_curves
from visualisation.graphs import draw_communication_graph, plot_edge_weight_heatmap

DEFAULT_OUTPUT = "results/figures"
_SNAPSHOT_LABELS = ("early", "mid", "late")


def _slug(label: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", label).strip("_") or "run"


def _save(figure, path: Path, saved: list[Path]) -> None:
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    saved.append(path)


def _snapshot_indices(n: int) -> list[int]:
    if n <= 1:
        return [0]
    return sorted({0, n // 2, n - 1})


def visualise_runs(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT,
    labels: Sequence[str] | None = None,
    threshold: float = 1e-3,
    smooth: int = 5,
) -> list[Path]:
    """Generate the standard figure set for ``run_dirs``; return the saved paths."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never opens a window
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(d) for d in run_dirs]
    labels = list(labels) if labels else [d.name for d in run_dirs]
    metrics = {label: load_metrics(d) for label, d in zip(labels, run_dirs)}
    saved: list[Path] = []

    # -- training curves (overlay all runs) -------------------------------
    def _series(column: str) -> dict:
        out = {}
        for label, cols in metrics.items():
            if column in cols and "env_steps" in cols and not np.isnan(cols[column]).all():
                out[label] = (cols["env_steps"], cols[column])
        return out

    reward = _series("episode_return_mean")
    if reward:
        ax = plot_training_curves(reward, ylabel="episode return", title="Reward curves", smooth=smooth)
        _save(ax.figure, output / "reward_curves.png", saved)

    density = _series("comm_edge_density")
    if density:
        ax = plot_training_curves(density, ylabel="graph density", title="Communication graph density over training")
        _save(ax.figure, output / "graph_density.png", saved)

    entropy = _series("comm_weight_entropy")
    if entropy:
        ax = plot_training_curves(entropy, ylabel="weight entropy (bits)", title="Communication entropy over training")
        _save(ax.figure, output / "communication_entropy.png", saved)

    # -- per-run graph figures (need edge-weight snapshots) ---------------
    for label, directory in zip(labels, run_dirs):
        archive = directory / "edge_weights.npz"
        if not archive.exists():
            continue
        data = np.load(archive, allow_pickle=True)
        weights = np.asarray(data["weights"], dtype=float)          # [T, N, N]
        steps = np.asarray(data["steps"], dtype=float)
        agents = [str(a) for a in data["agents"]]
        if weights.ndim != 3 or weights.shape[0] == 0:
            continue

        slug = _slug(label)
        indices = _snapshot_indices(weights.shape[0])
        titles = [f"{name} (step {int(steps[i])})" for name, i in zip(_SNAPSHOT_LABELS, indices)]
        off = ~np.eye(weights.shape[1], dtype=bool)
        vmax = float(weights[indices][:, off].max()) if weights.shape[1] > 1 else 1.0
        vmax = vmax if vmax > 0 else 1.0

        # edge-weight heatmaps (shared colour scale)
        fig, axes = plt.subplots(1, len(indices), figsize=(4.6 * len(indices), 4.2), squeeze=False)
        image = None
        for ax, idx, title in zip(axes[0], indices, titles):
            _, image = plot_edge_weight_heatmap(weights[idx], agents, ax=ax, title=title, vmin=0.0, vmax=vmax)
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="edge weight")
        fig.suptitle(f"Edge-weight heatmaps — {label}")
        _save(fig, output / f"edge_weight_heatmaps_{slug}.png", saved)

        # communication graph snapshots (shared edge scale)
        fig, axes = plt.subplots(1, len(indices), figsize=(4.8 * len(indices), 4.8), squeeze=False)
        for ax, idx, title in zip(axes[0], indices, titles):
            draw_communication_graph(weights[idx], agents, ax=ax, threshold=threshold, max_weight=vmax, title=title)
        fig.suptitle(f"Communication graph — {label}")
        _save(fig, output / f"graph_snapshots_{slug}.png", saved)

        # hub formation: per-agent out-strength (influence) over training
        out_strength = weights.sum(axis=1)  # [T, N] (sum over receivers -> per-sender influence)
        ax = plot_centrality_over_time(steps, out_strength, agents, title=f"Hub formation — {label}")
        _save(ax.figure, output / f"hub_centrality_{slug}.png", saved)

    return saved


__all__ = ["visualise_runs", "DEFAULT_OUTPUT"]
