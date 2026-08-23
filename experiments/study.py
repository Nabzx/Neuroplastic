"""Run the multi-seed mechanism study and reduce it to comparable scalars.

For each (mechanism, seed) it trains online across the task stream and records the
per-task history; per-seed scalar metrics are then extracted and aggregated across
seeds with CIs, plus permutation tests against the key baselines.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from analysis.statistics import holm_bonferroni, permutation_test, summarise
from core.seeding import set_global_seed
from data.streams import PermutedRegressionStream
from mechanisms.base import make_mechanism
from models.mlp import MLP
from training.continual import ContinualTrainer, accuracy

DEFAULT_CONFIG: dict[str, Any] = {
    "benchmark": "permuted_regression",
    "input_dim": 16,
    "teacher_hidden": 64,
    "num_tasks": 250,
    "task_length": 200,
    "batch_size": 1,
    "hidden_dim": 32,
    "layers": 2,
    "lr": 0.01,
    "optimizer": "sgd",          # sgd | adam  (robustness: Continual Backprop's edge grows with Adam)
    "shared_teacher": True,      # False => independent teacher per task (probes the ratio<1 anomaly)
    "mechanism_config": None,    # hyper-parameter overrides for the mechanism (sensitivity analysis)
}
METHODS = ["vanilla", "l2", "shrink_perturb", "redo", "continual_backprop", "homeostatic", "structural", "combined", "metaplastic"]
PRIMARY_METRIC = "final_late_loss"


def _build_run(cfg: Mapping[str, Any], seed: int):
    """Construct ``(stream, model, loss_fn, metric_fn)`` for the configured benchmark."""
    if cfg.get("benchmark") == "permuted_mnist":
        from data.mnist import PermutedMNISTStream

        stream = PermutedMNISTStream(
            num_tasks=cfg["num_tasks"], task_length=cfg["task_length"],
            batch_size=cfg["batch_size"], seed=seed,
        )
        model = MLP(784, cfg["hidden_dim"], output_dim=10, num_hidden_layers=cfg["layers"])
        return stream, model, nn.CrossEntropyLoss(), accuracy
    stream = PermutedRegressionStream(
        input_dim=cfg["input_dim"], teacher_hidden=cfg["teacher_hidden"],
        num_tasks=cfg["num_tasks"], task_length=cfg["task_length"],
        batch_size=cfg["batch_size"], seed=seed,
        shared_teacher=cfg.get("shared_teacher", True),
    )
    model = MLP(cfg["input_dim"], cfg["hidden_dim"], num_hidden_layers=cfg["layers"])
    return stream, model, nn.MSELoss(), None


def _build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    if str(cfg.get("optimizer", "sgd")).lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"])


def _run_one(method: str, seed: int, cfg: Mapping[str, Any]) -> list[dict[str, float]]:
    set_global_seed(seed)
    stream, model, loss_fn, metric_fn = _build_run(cfg, seed)
    optimizer = _build_optimizer(model, cfg)
    mechanism = make_mechanism(method, cfg.get("mechanism_config"))
    return ContinualTrainer(
        model, stream, optimizer, mechanism=mechanism, loss_fn=loss_fn, metric_fn=metric_fn
    ).run()


def run_study(
    methods: Sequence[str], seeds: Sequence[int], config: Mapping[str, Any] | None = None, logger=None
) -> dict[str, Any]:
    """Train every (method, seed); return ``{results, config, methods, seeds}``."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    results: dict[str, list] = {}
    for method in methods:
        results[method] = []
        for seed in seeds:
            results[method].append(_run_one(method, seed, cfg))
            if logger:
                logger.info("done %s / seed %d", method, seed)
    return {"results": results, "config": cfg, "methods": list(methods), "seeds": list(seeds)}


def per_seed_scalars(history: Sequence[Mapping[str, float]], window: int = 25) -> dict[str, float]:
    """Reduce a per-task history to scalar summary metrics for one run."""
    late = np.array([r["train_loss_late"] for r in history], dtype=float)
    dormant = np.array([r["dormant_fraction"] for r in history], dtype=float)
    rank = np.array([r["effective_rank"] for r in history], dtype=float)
    wmag = np.array([r["weight_magnitude"] for r in history], dtype=float)
    w = min(window, len(late))
    early, final = float(late[:w].mean()), float(late[-w:].mean())
    out = {
        "final_late_loss": final,
        "early_late_loss": early,
        "plasticity_ratio": final / max(1e-9, early),   # <1 = improves; >1 = loses plasticity
        "auc_late_loss": float(late.mean()),
        "final_dormant_fraction": float(dormant[-w:].mean()),
        "final_effective_rank": float(rank[-w:].mean()),
        "final_weight_magnitude": float(wmag[-w:].mean()),
    }
    if "accuracy_late" in history[0]:                    # classification benchmarks
        acc = np.array([r["accuracy_late"] for r in history], dtype=float)
        out["final_accuracy"] = float(acc[-w:].mean())
        out["early_accuracy"] = float(acc[:w].mean())
    return out


def aggregate_study(results: Mapping[str, list], window: int = 25) -> dict[str, dict[str, dict[str, float]]]:
    """``summary[method][metric] = {mean, std, ci_low, ci_high, n}`` across seeds."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, histories in results.items():
        scalars = [per_seed_scalars(h, window) for h in histories]
        keys = list(scalars[0]) if scalars else []
        summary[method] = {k: summarise([s[k] for s in scalars]) for k in keys}
    return summary


def significance_study(
    results: Mapping[str, list],
    window: int = 25,
    baselines: Sequence[str] = ("vanilla", "continual_backprop"),
    metric: str = PRIMARY_METRIC,
) -> dict[str, dict[str, dict[str, float]]]:
    """Permutation tests of each method vs each baseline on ``metric``.

    Each test reports difference, Cohen's d and a permutation p-value; a
    Holm-Bonferroni ``p_value_holm`` is added per baseline family (across methods)
    to control the family-wise error rate over multiple comparisons.
    """
    values = {m: [per_seed_scalars(h, window).get(metric, float("nan")) for h in hs] for m, hs in results.items()}
    tests: dict[str, dict[str, dict[str, float]]] = {}
    for method in results:
        tests[method] = {}
        for base in baselines:
            if base in values and method != base:
                tests[method][f"vs_{base}"] = permutation_test(values[method], values[base])

    # Holm-Bonferroni within each baseline family.
    for base in baselines:
        key = f"vs_{base}"
        methods_tested = [m for m in tests if key in tests[m]]
        adjusted = holm_bonferroni([tests[m][key]["p_value"] for m in methods_tested])
        for method, adj in zip(methods_tested, adjusted):
            tests[method][key]["p_value_holm"] = float(adj)
    return tests


def task_curves(results: Mapping[str, list], key: str = "train_loss_late") -> dict[str, tuple]:
    """``{method: (mean[T], ci_low[T], ci_high[T])}`` for a per-task quantity."""
    curves: dict[str, tuple] = {}
    for method, histories in results.items():
        arrays = [np.array([r[key] for r in h], dtype=float) for h in histories]
        length = min(len(a) for a in arrays)
        stacked = np.stack([a[:length] for a in arrays])
        mean = stacked.mean(axis=0)
        n = stacked.shape[0]
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(length)
        curves[method] = (mean, mean - 1.96 * sem, mean + 1.96 * sem)
    return curves


__all__ = [
    "run_study",
    "per_seed_scalars",
    "aggregate_study",
    "significance_study",
    "task_curves",
    "DEFAULT_CONFIG",
    "METHODS",
    "PRIMARY_METRIC",
]
