"""Benchmark suite: statistics, aggregation, specialisation, interpretation, run.

The analysis/statistics/interpretation pieces are torch-free and tested on
synthetic data; the end-to-end grid run requires the RL stack and is skipped
otherwise.
"""

import numpy as np
import pytest

from analysis.aggregate import (
    aggregate,
    build_summary_tables,
    per_seed_rows,
    significance_tests,
)
from analysis.interpret import generate_interpretation
from analysis.specialisation_analysis import cluster_roles, functional_specialisation
from analysis.statistics import permutation_test, summarise
from evaluation.coordination import reward_variance


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_summarise_basic():
    stats = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["n"] == 5
    assert stats["std"] > 0
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_summarise_ignores_nan_and_single():
    assert summarise([np.nan, np.nan])["n"] == 0
    single = summarise([2.0, np.nan])
    assert single["n"] == 1 and single["std"] == 0.0


def test_permutation_test_separates_groups():
    same = permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert same["p_value"] > 0.5
    diff = permutation_test([10, 11, 12, 13, 14], [0, 1, 2, 3, 4])
    assert diff["p_value"] < 0.05
    assert diff["difference"] == pytest.approx(10.0)


def test_reward_variance():
    assert reward_variance([0, 0, 0, 0]) == pytest.approx(0.0)
    assert reward_variance([0, 10, 0, 10], window_frac=1.0) > 0


# --------------------------------------------------------------------------- #
# Functional specialisation
# --------------------------------------------------------------------------- #
def test_specialisation_identical_agents():
    profiles = np.tile([0.5, 0.5], (3, 1))
    stats = functional_specialisation(profiles)
    assert stats["role_diversity"] == pytest.approx(0.0, abs=1e-9)
    assert stats["role_cluster_count"] == pytest.approx(1.0)


def test_specialisation_distinct_agents():
    profiles = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    stats = functional_specialisation(profiles)
    assert stats["role_diversity"] > 0.5
    assert stats["role_cluster_count"] == pytest.approx(2.0)
    assert cluster_roles(profiles) == [0, 1, 0] or cluster_roles(profiles) == [1, 0, 1]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _synthetic_results():
    return {
        "envA": {
            "no_comm": [{"coordination.final_reward": -30.0, "graph.density": float("nan")} for _ in range(3)],
            "fully_connected": [
                {"coordination.final_reward": v, "graph.density": 1.0} for v in (-22, -21, -20)
            ],
            "neuroplastic": [
                {"coordination.final_reward": v, "graph.density": 1.0} for v in (-19, -18, -20)
            ],
        }
    }


def test_aggregate_and_tables():
    results = _synthetic_results()
    summary = aggregate(results)
    assert summary["envA"]["fully_connected"]["coordination.final_reward"]["mean"] == pytest.approx(-21.0)
    assert summary["envA"]["no_comm"]["graph.density"]["n"] == 0  # all NaN dropped

    tables = build_summary_tables(summary)
    assert any(row["metric"] == "coordination.final_reward" for row in tables)
    assert "neuroplastic" in tables[0]

    rows = per_seed_rows(results)
    assert len(rows) == 9  # 3 methods x 3 seeds
    assert {"environment", "method", "seed_index"} <= set(rows[0])


def test_significance_tests_structure():
    tests = significance_tests(_synthetic_results())
    assert "envA" in tests
    assert "coordination.final_reward" in tests["envA"]
    assert "p_value" in tests["envA"]["coordination.final_reward"]


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def test_interpretation_is_honest_when_not_significant():
    results = _synthetic_results()
    summary = aggregate(results)
    significance = significance_tests(results)
    meta = {"methods": ["no_comm", "fully_connected", "neuroplastic"], "seeds": 3, "steps": 20000}
    text = generate_interpretation(summary, significance, meta)
    assert "Benchmark Summary" in text
    assert "Limitations" in text
    assert "cautiously" in text.lower()
    # 3 seeds, tiny effect -> permutation test should not be significant
    assert "no statistically significant improvement" in text.lower() or "mixed" in text.lower()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_benchmark_figures(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from visualisation.benchmark_figures import generate_benchmark_figures

    results = _synthetic_results()
    summary = aggregate(results)
    history = [
        {"env_steps": s, "episode_return_mean": -30 + s / 1000, "comm_mean_weight": 0.5 - s / 100000}
        for s in (500, 1000, 1500, 2000)
    ]
    agents = ["a0", "a1", "a2"]
    matrix = np.full((3, 3), 0.4)
    np.fill_diagonal(matrix, 0.0)
    bundle = {
        "results": results,
        "curves": {"envA": {"fully_connected": [history, history], "neuroplastic": [history, history]}},
        "finals": {"envA": {"fully_connected": (matrix, agents), "neuroplastic": (matrix, agents)}},
    }
    saved = generate_benchmark_figures(bundle, summary, tmp_path)
    names = {p.name for p in saved}
    assert "reward_curves_envA.png" in names
    assert "convergence_envA.png" in names
    assert "graph_stats_envA.png" in names
    assert "communication_heatmaps_envA.png" in names
    for path in saved:
        assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# End-to-end grid (requires PettingZoo + torch)
# --------------------------------------------------------------------------- #
def test_run_benchmark_small():
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")

    from training.benchmark import run_benchmark

    overrides = [
        "training.total_steps=120",
        "training.rollout_length=40",
        "training.minibatch_size=40",
        "training.update_epochs=1",
        "training.log_every=120",
        "env.num_agents=3",
        "env.max_cycles=10",
        "agent.hidden_dim=16",
        "agent.obs_embedding_dim=8",
        "communication.message_dim=4",
    ]
    methods = {"no_comm": "configs/baselines/no_comm.yaml", "neuroplastic": "configs/plastic.yaml"}
    try:
        bundle = run_benchmark(["simple_spread"], methods, seeds=[0, 1], overrides=overrides, spec_episodes=2)
    except ImportError as exc:  # MPE unavailable
        pytest.skip(f"environment unavailable: {exc}")

    results = bundle["results"]["simple_spread"]
    assert len(results["no_comm"]) == 2 and len(results["neuroplastic"]) == 2
    assert "coordination.final_reward" in results["neuroplastic"][0]
    assert "specialisation.role_diversity" in results["neuroplastic"][0]

    summary = aggregate(bundle["results"])
    assert "simple_spread" in summary
