"""Aggregate per-seed benchmark results into summary statistics and tables.

Given ``results[env][method] = [flat_metric_dict_per_seed, ...]`` (metric keys are
``"section.metric"`` strings), produce mean/std/CI summaries, human-readable
comparison tables, and significance tests between two methods. Pure NumPy/stdlib.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analysis.statistics import permutation_test, summarise

#: Section display order for tables.
_SECTION_ORDER = {"coordination": 0, "graph": 1, "stability": 2, "specialisation": 3}
#: Metrics compared for significance (treatment vs control).
DEFAULT_SIGNIFICANCE_METRICS = [
    "coordination.final_reward",
    "coordination.cumulative_reward",
    "coordination.convergence_steps",
    "coordination.reward_variance",
]


def _metric_sort_key(name: str) -> tuple[int, str]:
    section = name.split(".", 1)[0]
    return (_SECTION_ORDER.get(section, 99), name)


def _collect_by_metric(seed_dicts: Sequence[Mapping[str, float]]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for record in seed_dicts:
        for name, value in record.items():
            metrics[name].append(value)
    return metrics


def all_metric_names(results: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]]) -> list[str]:
    """Sorted union of metric names appearing anywhere in ``results``."""
    names: set[str] = set()
    for methods in results.values():
        for seed_dicts in methods.values():
            for record in seed_dicts:
                names.update(record)
    return sorted(names, key=_metric_sort_key)


def aggregate(results: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]]) -> dict:
    """Return ``summary[env][method][metric] = {mean, std, ci_low, ci_high, n}``."""
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for env, methods in results.items():
        summary[env] = {}
        for method, seed_dicts in methods.items():
            metrics = _collect_by_metric(seed_dicts)
            summary[env][method] = {name: summarise(values) for name, values in metrics.items()}
    return summary


def significance_tests(
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]],
    treatment: str = "neuroplastic",
    control: str = "fully_connected",
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Permutation tests of ``treatment`` vs ``control`` per environment/metric."""
    metric_names = list(metrics or DEFAULT_SIGNIFICANCE_METRICS)
    tests: dict[str, dict[str, dict[str, float]]] = {}
    for env, methods in results.items():
        if treatment not in methods or control not in methods:
            continue
        treatment_dicts = methods[treatment]
        control_dicts = methods[control]
        tests[env] = {}
        for name in metric_names:
            a = [d.get(name, float("nan")) for d in treatment_dicts]
            b = [d.get(name, float("nan")) for d in control_dicts]
            tests[env][name] = permutation_test(a, b)
    return tests


def build_summary_tables(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    """Rows of ``{environment, metric, <method>: "mean ± std"}`` for a CSV table."""
    rows: list[dict[str, str]] = []
    for env, methods in summary.items():
        method_names = list(methods)
        metric_names = sorted(
            {m for stats in methods.values() for m in stats}, key=_metric_sort_key
        )
        for metric in metric_names:
            row: dict[str, str] = {"environment": env, "metric": metric}
            for method in method_names:
                stats = methods[method].get(metric, {})
                mean, std = stats.get("mean", float("nan")), stats.get("std", float("nan"))
                row[method] = f"{mean:.4g} ± {std:.4g}"
            rows.append(row)
    return rows


def per_seed_rows(
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]],
) -> list[dict[str, Any]]:
    """Long-form rows ``{environment, method, seed, <metric>: value}`` for a CSV."""
    rows: list[dict[str, Any]] = []
    for env, methods in results.items():
        for method, seed_dicts in methods.items():
            for seed_index, record in enumerate(seed_dicts):
                row: dict[str, Any] = {"environment": env, "method": method, "seed_index": seed_index}
                row.update(record)
                rows.append(row)
    return rows


__all__ = [
    "aggregate",
    "significance_tests",
    "build_summary_tables",
    "per_seed_rows",
    "all_metric_names",
    "DEFAULT_SIGNIFICANCE_METRICS",
]
