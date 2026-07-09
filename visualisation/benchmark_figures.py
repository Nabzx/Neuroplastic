"""Publication-style benchmark figures (matplotlib only).

Consumes the ``{results, curves, finals}`` bundle from
:func:`training.benchmark.run_benchmark` plus the aggregated ``summary`` and
writes per-environment figures: reward curves with cross-seed CIs, convergence
and entropy bar charts, a graph-statistics panel, edge-weight evolution, and
final communication heatmaps / graph snapshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from visualisation.graphs import draw_communication_graph, plot_edge_weight_heatmap

_ORDER = ["no_comm", "fully_connected", "sparse", "adaptive", "neuroplastic"]


def _ordered(methods: Sequence[str]) -> list[str]:
    present = list(methods)
    return [m for m in _ORDER if m in present] + [m for m in present if m not in _ORDER]


def _stack(histories: Sequence[Sequence[Mapping[str, Any]]], column: str):
    """Return ``(steps, matrix[seeds, T])`` for ``column`` across seed histories."""
    series, steps = [], None
    for history in histories:
        values = np.array([row.get(column, np.nan) for row in history], dtype=float)
        series.append(values)
        if steps is None:
            steps = np.array([row.get("env_steps", np.nan) for row in history], dtype=float)
    if not series:
        return None, None
    length = min(len(s) for s in series)
    return steps[:length], np.stack([s[:length] for s in series])


def _save(fig, path: Path, saved: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def _ci_curves(ax, curves_by_method: Mapping[str, list], column: str) -> bool:
    """Plot mean +/- 95% CI (across seeds) per method; return True if anything drawn."""
    drawn = False
    for method in _ordered(list(curves_by_method)):
        histories = curves_by_method.get(method, [])
        if not histories:
            continue
        steps, matrix = _stack(histories, column)
        if steps is None or np.isnan(matrix).all():
            continue
        mean = np.nanmean(matrix, axis=0)
        n = matrix.shape[0]
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        (line,) = ax.plot(steps, mean, linewidth=2, label=method)
        ax.fill_between(steps, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.18, color=line.get_color())
        drawn = True
    return drawn


def _bar(ax, summary_env: Mapping[str, Any], metric: str, ylabel: str, title: str) -> None:
    methods = _ordered(list(summary_env))
    means = [summary_env[m].get(metric, {}).get("mean", np.nan) for m in methods]
    stds = [summary_env[m].get(metric, {}).get("std", np.nan) for m in methods]
    ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color="#4c72b0", alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def generate_benchmark_figures(
    bundle: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_dir: str | Path,
    threshold: float = 1e-3,
) -> list[Path]:
    """Write the per-environment figure set; return the saved paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, curves, finals = bundle["results"], bundle["curves"], bundle["finals"]
    saved: list[Path] = []

    for env in results:
        env_curves = curves.get(env, {})
        env_summary = summary.get(env, {})
        env_finals = finals.get(env, {})

        # 1. reward curves with cross-seed CI
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if _ci_curves(ax, env_curves, "episode_return_mean"):
            ax.set_xlabel("environment steps")
            ax.set_ylabel("episode return")
            ax.set_title(f"Reward curves (mean +/- 95% CI) — {env}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            _save(fig, output / f"reward_curves_{env}.png", saved)
        else:
            plt.close(fig)

        # 2. convergence + 3. communication entropy (bar charts)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _bar(ax, env_summary, "coordination.convergence_steps", "env steps to converge", f"Convergence — {env}")
        _save(fig, output / f"convergence_{env}.png", saved)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        _bar(ax, env_summary, "graph.comm_weight_entropy", "weight entropy (bits)", f"Communication entropy — {env}")
        _save(fig, output / f"communication_entropy_{env}.png", saved)

        # 4. graph-statistics panel
        panel = [
            ("graph.density", "density"),
            ("graph.weighted_clustering", "weighted clustering"),
            ("graph.modularity", "modularity"),
            ("graph.comm_effective_degree", "effective degree"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for ax, (metric, label) in zip(axes, panel):
            _bar(ax, env_summary, metric, label, label)
        fig.suptitle(f"Communication-graph statistics — {env}")
        _save(fig, output / f"graph_stats_{env}.png", saved)

        # 5. edge-weight evolution (mean edge weight over training, CI)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if _ci_curves(ax, env_curves, "comm_mean_weight"):
            ax.set_xlabel("environment steps")
            ax.set_ylabel("mean edge weight")
            ax.set_title(f"Edge-weight evolution (mean +/- 95% CI) — {env}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            _save(fig, output / f"edge_weight_evolution_{env}.png", saved)
        else:
            plt.close(fig)

        # 6. communication heatmaps + 7. graph snapshots (final, per comm method)
        comm_methods = [m for m in _ordered(list(env_finals)) if env_finals.get(m)]
        if comm_methods:
            matrices = [env_finals[m][0] for m in comm_methods]
            off = ~np.eye(matrices[0].shape[0], dtype=bool)
            vmax = max((np.asarray(m)[off].max() for m in matrices), default=1.0) or 1.0

            fig, axes = plt.subplots(1, len(comm_methods), figsize=(4.4 * len(comm_methods), 4.0), squeeze=False)
            image = None
            for ax, method in zip(axes[0], comm_methods):
                matrix, agents = env_finals[method]
                _, image = plot_edge_weight_heatmap(matrix, agents, ax=ax, title=method, vmin=0.0, vmax=vmax)
            fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="edge weight")
            fig.suptitle(f"Final communication heatmaps — {env}")
            _save(fig, output / f"communication_heatmaps_{env}.png", saved)

            fig, axes = plt.subplots(1, len(comm_methods), figsize=(4.6 * len(comm_methods), 4.6), squeeze=False)
            for ax, method in zip(axes[0], comm_methods):
                matrix, agents = env_finals[method]
                draw_communication_graph(matrix, agents, ax=ax, threshold=threshold, max_weight=vmax, title=method)
            fig.suptitle(f"Final communication graphs — {env}")
            _save(fig, output / f"graph_snapshots_{env}.png", saved)

    return saved


def _metric_vs_budget(ax, summary_by_budget, budgets, metric, ylabel, title) -> bool:
    """Plot ``metric`` mean +/- 95% CI against training budget, one line per method."""
    drawn = False
    methods = _ordered(list(next(iter(summary_by_budget.values()), {})))
    for method in methods:
        xs, ys, lo, hi = [], [], [], []
        for budget in budgets:
            stats = summary_by_budget.get(str(budget), {}).get(method, {}).get(metric, {})
            mean = stats.get("mean", float("nan"))
            if mean != mean:
                continue
            xs.append(budget)
            ys.append(mean)
            lo.append(mean - stats.get("ci_low", mean))
            hi.append(stats.get("ci_high", mean) - mean)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4, linewidth=2, label=method)
            drawn = True
    ax.set_xlabel("training budget (env steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=8)
    return drawn


def generate_long_horizon_figures(
    curves_by_budget: Mapping[str, Mapping[str, list]],
    finals_by_budget: Mapping[str, Mapping[str, tuple]],
    summary_by_budget: Mapping[str, Any],
    budgets: Sequence[int],
    output_dir: str | Path,
    threshold: float = 1e-3,
) -> list[Path]:
    """Figures for the budget sweep: per-budget reward curves + metric-vs-budget trends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # per-budget reward curves with cross-seed CI
    for budget in budgets:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if _ci_curves(ax, curves_by_budget.get(str(budget), {}), "episode_return_mean"):
            ax.set_xlabel("environment steps")
            ax.set_ylabel("episode return")
            ax.set_title(f"Reward curves (mean +/- 95% CI) — {budget} steps")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            _save(fig, output / f"reward_curves_{budget}steps.png", saved)
        else:
            plt.close(fig)

    # metric-vs-budget trend panels (does anything separate as budget grows?)
    trends = [
        ("coordination.final_reward", "final reward", "final_reward_vs_budget.png"),
        ("coordination.convergence_steps", "steps to converge", "convergence_vs_budget.png"),
        ("coordination.reward_variance", "reward variance", "reward_variance_vs_budget.png"),
        ("graph.comm_weight_entropy", "communication entropy (bits)", "communication_entropy_vs_budget.png"),
        ("graph.degree_heterogeneity", "degree heterogeneity", "graph_heterogeneity_vs_budget.png"),
        ("stability.edge_weight_drift", "edge-weight drift", "edge_weight_stability_vs_budget.png"),
        ("specialisation.role_diversity", "role diversity", "specialisation_vs_budget.png"),
    ]
    for metric, ylabel, filename in trends:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if _metric_vs_budget(ax, summary_by_budget, budgets, metric, ylabel, f"{ylabel} vs budget"):
            _save(fig, output / filename, saved)
        else:
            plt.close(fig)

    # final communication heatmaps at the longest budget
    longest = str(max(budgets))
    comm = [m for m in _ordered(list(finals_by_budget.get(longest, {}))) if finals_by_budget[longest].get(m)]
    if comm:
        matrices = [finals_by_budget[longest][m][0] for m in comm]
        off = ~np.eye(matrices[0].shape[0], dtype=bool)
        vmax = max((np.asarray(m)[off].max() for m in matrices), default=1.0) or 1.0
        fig, axes = plt.subplots(1, len(comm), figsize=(4.4 * len(comm), 4.0), squeeze=False)
        image = None
        for ax, method in zip(axes[0], comm):
            matrix, agents = finals_by_budget[longest][method]
            _, image = plot_edge_weight_heatmap(matrix, agents, ax=ax, title=method, vmin=0.0, vmax=vmax)
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="edge weight")
        fig.suptitle(f"Final communication heatmaps at {longest} steps")
        _save(fig, output / f"communication_heatmaps_{longest}steps.png", saved)

    return saved


__all__ = ["generate_benchmark_figures", "generate_long_horizon_figures"]
