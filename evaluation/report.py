"""Bundle the evaluation metrics into a single run report.

Ties together coordination (learning-curve), communication-graph structure and
edge-weight stability into one nested dict, and can build that report either from
in-memory training results or from a saved run directory
(``metrics.csv`` + ``edge_weights.npz``). Pure NumPy / NetworkX / stdlib.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from communication.graph import InteractionGraph
from communication.statistics import weight_matrix_statistics
from core.types import AgentID, ArrayLike
from evaluation.coordination import convergence_speed, cumulative_reward, final_reward
from evaluation.graph_metrics import GRAPH_METRICS
from evaluation.stability import edge_weight_stability

#: Structural graph metrics computed for a communication graph, in report order.
GRAPH_METRIC_NAMES = [
    "density",
    "mean_degree",
    "degree_heterogeneity",
    "global_clustering",
    "weighted_clustering",
    "degree_centralisation",
    "betweenness_centralisation",
    "eigenvector_centrality_max",
    "modularity",
]
#: Weight-distribution (entropy) statistics from communication.statistics.
COMM_STAT_KEYS = [
    "comm_edge_density",
    "comm_mean_weight",
    "comm_max_weight",
    "comm_weight_entropy",
    "comm_effective_degree",
]


def graph_report(
    matrix: ArrayLike | None,
    agent_ids: Sequence[AgentID] | None,
    threshold: float = 1e-3,
) -> dict[str, float]:
    """Structural + weight-distribution metrics for a ``[N, N]`` weight matrix.

    ``matrix[i, j]`` is the weight from sender ``j`` to receiver ``i``. Returns a
    flat dict; all values are NaN when ``matrix`` is ``None`` (no communication).
    """
    keys = GRAPH_METRIC_NAMES + COMM_STAT_KEYS
    if matrix is None or agent_ids is None:
        return {k: float("nan") for k in keys}

    graph = InteractionGraph.from_weight_matrix(agent_ids, np.asarray(matrix), threshold=threshold)
    report: dict[str, float] = {}
    for name in GRAPH_METRIC_NAMES:
        try:
            report[name] = float(GRAPH_METRICS.get(name)(graph))
        except Exception:  # pragma: no cover - defensive: never let one metric abort the report
            report[name] = float("nan")
    report.update(weight_matrix_statistics(matrix))
    return report


def coordination_report(returns: ArrayLike, steps: ArrayLike | None = None) -> dict[str, float]:
    """Learning-curve metrics: cumulative reward, final reward, convergence speed."""
    return {
        "final_reward": final_reward(returns),
        "cumulative_reward": cumulative_reward(returns),
        **convergence_speed(returns, steps),
    }


def evaluate_run(
    returns: ArrayLike,
    steps: ArrayLike | None,
    snapshots: ArrayLike | None,
    agent_ids: Sequence[AgentID] | None,
    threshold: float = 1e-3,
) -> dict[str, dict[str, float]]:
    """Full report: ``{"coordination": ..., "graph": ..., "stability": ...}``.

    ``snapshots`` is the ``[T, N, N]`` edge-weight history (or ``None``); the graph
    metrics use its final matrix.
    """
    snapshots = None if snapshots is None or len(snapshots) == 0 else np.asarray(snapshots)
    final_matrix = snapshots[-1] if snapshots is not None else None
    return {
        "coordination": coordination_report(returns, steps),
        "graph": graph_report(final_matrix, agent_ids, threshold),
        "stability": edge_weight_stability(snapshots),
    }


def load_run(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str] | None]:
    """Load ``(returns, steps, snapshots, agent_ids)`` from a saved run directory."""
    directory = Path(run_dir)
    returns: list[float] = []
    steps: list[float] = []
    with (directory / "metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            returns.append(float(row["episode_return_mean"]))
            steps.append(float(row["env_steps"]))

    snapshots: np.ndarray | None = None
    agent_ids: list[str] | None = None
    archive = directory / "edge_weights.npz"
    if archive.exists():
        data = np.load(archive, allow_pickle=True)
        snapshots = data["weights"]
        agent_ids = [str(a) for a in data["agents"]]
    return np.array(returns), np.array(steps), snapshots, agent_ids


def load_metrics(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Load every column of a run's ``metrics.csv`` as a float array.

    Non-numeric / missing cells become NaN, so downstream plotting can mask them.
    Useful for per-training-step curves (reward, density, entropy, ...).
    """
    rows: list[dict[str, str]] = []
    with (Path(run_dir) / "metrics.csv").open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows.extend(reader)

    columns: dict[str, np.ndarray] = {}
    for name in fieldnames:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[name] = np.array(values, dtype=float)
    return columns


def flatten_report(report: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Flatten a nested report to ``{"section.metric": value}`` (order preserved)."""
    flat: dict[str, float] = {}
    for section, metrics in report.items():
        for name, value in metrics.items():
            flat[f"{section}.{name}"] = value
    return flat


def format_comparison(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Render a side-by-side comparison table (metrics x settings) as text."""
    flats = {name: flatten_report(rep) for name, rep in reports.items()}
    settings = list(flats)
    metrics = list(next(iter(flats.values())).keys()) if flats else []

    metric_width = max([len("metric")] + [len(m) for m in metrics]) if metrics else 6
    col_width = max([10] + [len(s) for s in settings])

    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    header = "metric".ljust(metric_width) + "  " + "  ".join(s.rjust(col_width) for s in settings)
    lines = [header, "-" * len(header)]
    for metric in metrics:
        row = metric.ljust(metric_width) + "  "
        row += "  ".join(fmt(flats[s].get(metric, "nan")).rjust(col_width) for s in settings)
        lines.append(row)
    return "\n".join(lines)


__all__ = [
    "graph_report",
    "coordination_report",
    "evaluate_run",
    "load_run",
    "load_metrics",
    "flatten_report",
    "format_comparison",
    "GRAPH_METRIC_NAMES",
    "COMM_STAT_KEYS",
]
