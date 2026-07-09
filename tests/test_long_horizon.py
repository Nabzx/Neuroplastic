"""Long-horizon (budget sweep) interpretation and figures — torch-free tests."""

import numpy as np
import pytest

from analysis.aggregate import aggregate, significance_tests
from analysis.interpret import generate_long_horizon_interpretation


def _budget_results():
    """Synthetic results keyed by budget label -> method -> per-seed metric dicts."""

    def cell(final, drift, het):
        return {
            "coordination.final_reward": final,
            "coordination.convergence_steps": 10000.0,
            "coordination.reward_variance": 1.0,
            "graph.comm_weight_entropy": 1.585,
            "graph.degree_heterogeneity": het,
            "stability.edge_weight_drift": drift,
            "specialisation.role_diversity": 0.15,
        }

    results = {}
    for budget, (fx, ad, npl, drift, het) in {
        "20000": (-29.0, -29.1, -29.0, 0.005, 0.01),
        "100000": (-20.0, -19.8, -20.1, 0.004, 0.02),
        "250000": (-15.0, -14.8, -15.1, 0.003, 0.03),
    }.items():
        results[budget] = {
            "fully_connected": [cell(fx + 0.1 * s, 0.0, 0.0) for s in range(5)],
            "adaptive": [cell(ad + 0.1 * s, drift * 3, het * 1.5) for s in range(5)],
            "neuroplastic": [cell(npl + 0.1 * s, drift, het) for s in range(5)],
        }
    return results


def _significance(results):
    sig_fixed = significance_tests(results, "neuroplastic", "fully_connected")
    sig_adaptive = significance_tests(results, "neuroplastic", "adaptive")
    return {
        b: {"vs_fixed": sig_fixed.get(b, {}), "vs_adaptive": sig_adaptive.get(b, {})}
        for b in results
    }


def test_long_horizon_interpretation_is_honest():
    results = _budget_results()
    summary = aggregate(results)
    significance = _significance(results)
    meta = {"methods": ["fully_connected", "adaptive", "neuroplastic"], "seeds": 5, "budgets": [20000, 100000, 250000]}
    text = generate_long_horizon_interpretation(summary, significance, meta)

    assert "Long-Horizon Benchmark" in text
    assert "Research question" in text
    assert "Final reward by budget" in text
    # a table row per budget
    for budget in ("20000", "100000", "250000"):
        assert budget in text
    # one of the four outcomes is stated
    assert "Outcome" in text
    # neuroplastic ~ fixed here -> not a significant win; honest direct answer
    assert "Was the earlier negative result caused by too little training?" in text
    assert "no." in text.lower() or "partly" in text.lower()


def test_long_horizon_significance_has_both_comparisons():
    results = _budget_results()
    significance = _significance(results)
    assert "vs_fixed" in significance["250000"] and "vs_adaptive" in significance["250000"]
    assert "coordination.final_reward" in significance["250000"]["vs_fixed"]


def test_long_horizon_figures(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from visualisation.benchmark_figures import generate_long_horizon_figures

    results = _budget_results()
    summary = aggregate(results)
    budgets = [20000, 100000, 250000]

    history = [{"env_steps": s, "episode_return_mean": -30 + s / 10000, "comm_mean_weight": 0.5} for s in (500, 1000, 1500)]
    matrix = np.full((4, 4), 0.3)
    np.fill_diagonal(matrix, 0.0)
    curves_by_budget = {
        str(b): {"fully_connected": [history, history], "adaptive": [history, history], "neuroplastic": [history, history]}
        for b in budgets
    }
    finals_by_budget = {
        str(b): {"fully_connected": (matrix, ["a0", "a1", "a2", "a3"]), "neuroplastic": (matrix, ["a0", "a1", "a2", "a3"])}
        for b in budgets
    }

    saved = generate_long_horizon_figures(curves_by_budget, finals_by_budget, summary, budgets, tmp_path)
    names = {p.name for p in saved}
    assert "reward_curves_250000steps.png" in names
    assert "final_reward_vs_budget.png" in names
    assert "edge_weight_stability_vs_budget.png" in names
    assert "communication_heatmaps_250000steps.png" in names
    for path in saved:
        assert path.exists() and path.stat().st_size > 0
