"""Evaluation tools: coordination, graph-structure, stability, and comparison.

The metric library is torch-free and tested directly on synthetic data. The
end-to-end comparison of the three communication settings requires the RL stack
and is skipped without it.
"""

import numpy as np
import pytest

from evaluation.coordination import convergence_speed, cumulative_reward, final_reward
from evaluation.report import (
    evaluate_run,
    flatten_report,
    format_comparison,
    graph_report,
    load_run,
)
from evaluation.stability import edge_weight_stability

AGENTS = ["a0", "a1", "a2", "a3"]


def _uniform_matrix(n: int = 4) -> np.ndarray:
    m = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(m, 0.0)
    return m


# --------------------------------------------------------------------------- #
# Coordination / learning curve
# --------------------------------------------------------------------------- #
def test_cumulative_and_final_reward():
    returns = [-30.0, -20.0, -10.0, -5.0]
    assert cumulative_reward(returns) == pytest.approx(-65.0)
    assert final_reward(returns, window_frac=0.5) == pytest.approx(-7.5)  # last 2


def test_cumulative_reward_ignores_nan():
    assert cumulative_reward([np.nan, -1.0, -2.0]) == pytest.approx(-3.0)


def test_convergence_speed_on_improving_curve():
    returns = np.linspace(-30.0, -10.0, 50)
    steps = np.arange(50) * 100
    result = convergence_speed(returns, steps, threshold=0.9, smooth=3)
    assert 0 < result["convergence_steps"] <= steps[-1]
    assert 0 < result["convergence_fraction"] <= 1.0


def test_convergence_speed_flat_curve_is_nan():
    result = convergence_speed(np.full(20, -20.0), np.arange(20))
    assert np.isnan(result["convergence_steps"])


# --------------------------------------------------------------------------- #
# Communication-graph metrics
# --------------------------------------------------------------------------- #
def test_graph_report_uniform():
    report = graph_report(_uniform_matrix(4), AGENTS)
    assert report["density"] == pytest.approx(1.0)             # complete graph
    assert report["degree_heterogeneity"] == pytest.approx(0.0, abs=1e-6)
    assert report["global_clustering"] == pytest.approx(1.0)
    assert report["comm_effective_degree"] == pytest.approx(3.0)
    assert report["comm_weight_entropy"] == pytest.approx(np.log2(3), abs=1e-6)


def test_graph_report_hub_is_heterogeneous():
    hub = np.full((4, 4), 0.02)
    hub[:, 1] = 0.9                # sender a1 broadcasts strongly to everyone
    np.fill_diagonal(hub, 0.0)
    report = graph_report(hub, AGENTS)
    assert report["degree_heterogeneity"] > 0.1               # uneven degree distribution
    assert report["eigenvector_centrality_max"] > 0.0


def test_graph_report_none_is_all_nan():
    report = graph_report(None, None)
    assert all(np.isnan(v) for v in report.values())
    assert "density" in report and "comm_weight_entropy" in report


# --------------------------------------------------------------------------- #
# Edge-weight stability
# --------------------------------------------------------------------------- #
def test_stability_constant_graph_is_perfectly_stable():
    snapshots = np.stack([_uniform_matrix(4)] * 6)  # fixed comm: never changes
    stats = edge_weight_stability(snapshots)
    assert stats["edge_weight_drift"] == pytest.approx(0.0)
    assert stats["edge_weight_stability"] == pytest.approx(1.0)
    assert stats["edge_weight_variance"] == pytest.approx(0.0)


def test_stability_evolving_graph_drifts():
    rng = np.random.default_rng(0)
    base = _uniform_matrix(4)
    snapshots = np.stack([base + 0.1 * t + 0.01 * rng.standard_normal((4, 4)) for t in range(6)])
    stats = edge_weight_stability(snapshots)
    assert stats["edge_weight_drift"] > 0.0
    assert stats["edge_weight_final_shift"] > 0.0


def test_stability_none_is_nan():
    assert np.isnan(edge_weight_stability(None)["edge_weight_stability"])


# --------------------------------------------------------------------------- #
# Report bundling + IO
# --------------------------------------------------------------------------- #
def test_evaluate_run_bundles_sections():
    returns = np.linspace(-30, -10, 10)
    steps = np.arange(10) * 100
    snapshots = np.stack([_uniform_matrix(4)] * 10)
    report = evaluate_run(returns, steps, snapshots, AGENTS)
    assert set(report) == {"coordination", "graph", "stability"}
    assert np.isfinite(report["coordination"]["final_reward"])
    assert report["graph"]["density"] == pytest.approx(1.0)


def test_evaluate_run_without_communication():
    report = evaluate_run(np.linspace(-30, -20, 5), np.arange(5), None, None)
    assert np.isnan(report["graph"]["density"])
    assert np.isnan(report["stability"]["edge_weight_stability"])
    assert np.isfinite(report["coordination"]["cumulative_reward"])


def test_load_run_roundtrip(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "iteration,env_steps,episode_return_mean\n1,500,-30.0\n2,1000,-20.0\n"
    )
    np.savez(
        tmp_path / "edge_weights.npz",
        steps=np.array([500, 1000]),
        weights=np.stack([_uniform_matrix(4), _uniform_matrix(4)]),
        agents=np.array(AGENTS),
    )
    returns, steps, snapshots, agent_ids = load_run(tmp_path)
    assert list(returns) == [-30.0, -20.0]
    assert list(steps) == [500.0, 1000.0]
    assert snapshots.shape == (2, 4, 4)
    assert agent_ids == AGENTS
    # a loaded run evaluates end-to-end
    report = evaluate_run(returns, steps, snapshots, agent_ids)
    assert np.isfinite(report["coordination"]["final_reward"])


def test_format_comparison_table():
    reports = {
        "no_comm": evaluate_run([-30, -25], [100, 200], None, None),
        "fixed": evaluate_run([-25, -20], [100, 200], np.stack([_uniform_matrix(4)] * 2), AGENTS),
    }
    table = format_comparison(reports)
    assert "no_comm" in table and "fixed" in table
    assert "coordination.final_reward" in table
    assert "graph.density" in table


# --------------------------------------------------------------------------- #
# End-to-end comparison (requires PettingZoo + torch)
# --------------------------------------------------------------------------- #
TINY = [
    "training.total_steps=160",
    "training.rollout_length=40",
    "training.minibatch_size=40",
    "training.update_epochs=2",
    "training.log_every=40",
    "env.num_agents=4",
    "env.max_cycles=10",
    "agent.hidden_dim=32",
    "agent.obs_embedding_dim=16",
    "communication.message_dim=8",
]


def test_compare_settings_end_to_end():
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")

    from training.compare import DEFAULT_SETTINGS, compare_settings

    try:
        reports = compare_settings(DEFAULT_SETTINGS, TINY)
    except ImportError as exc:  # MPE unavailable
        pytest.skip(f"environment unavailable: {exc}")

    assert set(reports) == {"no_comm", "fixed", "neuroplastic"}
    for report in reports.values():
        assert set(report) == {"coordination", "graph", "stability"}
        assert np.isfinite(report["coordination"]["final_reward"])

    # no communication -> no graph
    assert np.isnan(reports["no_comm"]["graph"]["density"])
    # fixed communication never changes; neuroplastic evolves
    assert reports["fixed"]["stability"]["edge_weight_drift"] == pytest.approx(0.0, abs=1e-6)
    assert reports["neuroplastic"]["stability"]["edge_weight_drift"] > 0.0
