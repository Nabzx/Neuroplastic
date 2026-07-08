"""Visualisation: matplotlib-only plots generated from saved experiment logs.

Requires matplotlib (``[viz]`` extra); skipped otherwise. Uses the Agg backend so
it runs headless.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from visualisation.dynamics import plot_centrality_over_time, plot_training_curves  # noqa: E402
from visualisation.graphs import draw_communication_graph, plot_edge_weight_heatmap  # noqa: E402
from visualisation.run_figures import visualise_runs  # noqa: E402

AGENTS = ["a0", "a1", "a2", "a3"]


def _weight_matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 1.0, size=(4, 4))
    np.fill_diagonal(m, 0.0)
    return m


def _write_run(directory, *, with_comm: bool, n_iters: int = 6) -> None:
    """Create a synthetic run directory (metrics.csv [+ edge_weights.npz])."""
    directory.mkdir(parents=True, exist_ok=True)
    steps = [(i + 1) * 500 for i in range(n_iters)]
    returns = np.linspace(-30, -12, n_iters)
    lines = ["env_steps,episode_return_mean,comm_edge_density,comm_weight_entropy"]
    for i in range(n_iters):
        if with_comm:
            lines.append(f"{steps[i]},{returns[i]:.3f},1.0,{1.5 + 0.05 * i:.3f}")
        else:
            lines.append(f"{steps[i]},{returns[i]:.3f},,")  # no comm columns
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n")

    if with_comm:
        weights = np.stack([_weight_matrix(i) for i in range(n_iters)])  # [T, N, N]
        np.savez(
            directory / "edge_weights.npz",
            steps=np.array(steps),
            weights=weights,
            agents=np.array(AGENTS),
        )


# --------------------------------------------------------------------------- #
# Individual plot functions
# --------------------------------------------------------------------------- #
def test_heatmap_returns_axes_and_saves(tmp_path):
    import matplotlib.pyplot as plt

    ax, image = plot_edge_weight_heatmap(_weight_matrix(), AGENTS, title="w")
    assert image is not None
    out = tmp_path / "heatmap.png"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


def test_draw_communication_graph_runs(tmp_path):
    import matplotlib.pyplot as plt

    ax, scatter = draw_communication_graph(_weight_matrix(), AGENTS, title="graph")
    out = tmp_path / "graph.png"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


def test_training_curves_and_centrality(tmp_path):
    import matplotlib.pyplot as plt

    runs = {"a": ([100, 200, 300], [-30, -20, -10]), "b": ([100, 200, 300], [-28, -22, -18])}
    ax = plot_training_curves(runs, ylabel="return", title="reward", smooth=1)
    assert ax is not None
    plt.close(ax.figure)

    centralities = np.random.default_rng(0).uniform(0, 1, size=(6, 4))
    ax = plot_centrality_over_time([0, 1, 2, 3, 4, 5], centralities, AGENTS)
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# End-to-end figure generation from saved logs
# --------------------------------------------------------------------------- #
def test_visualise_single_run_creates_all_figures(tmp_path):
    run = tmp_path / "plastic_run"
    _write_run(run, with_comm=True)
    figures = tmp_path / "figures"

    saved = visualise_runs([run], output_dir=figures)
    names = {p.name for p in saved}
    assert "reward_curves.png" in names
    assert "graph_density.png" in names
    assert "communication_entropy.png" in names
    assert "edge_weight_heatmaps_plastic_run.png" in names
    assert "graph_snapshots_plastic_run.png" in names
    assert "hub_centrality_plastic_run.png" in names
    for path in saved:
        assert path.exists() and path.stat().st_size > 0


def test_visualise_handles_no_communication_run(tmp_path):
    comm_run = tmp_path / "fixed"
    nocomm_run = tmp_path / "no_comm"
    _write_run(comm_run, with_comm=True)
    _write_run(nocomm_run, with_comm=False)
    figures = tmp_path / "figures"

    saved = visualise_runs([comm_run, nocomm_run], output_dir=figures)
    names = {p.name for p in saved}
    # shared reward curve is produced (both runs contribute)
    assert "reward_curves.png" in names
    # per-run graph figures exist only for the run that has edge weights
    assert "graph_snapshots_fixed.png" in names
    assert "graph_snapshots_no_comm.png" not in names
    assert "hub_centrality_no_comm.png" not in names
