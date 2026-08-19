"""Run the multi-seed mechanism study and reduce it to comparable scalars.

For each (mechanism, seed) it trains online across the task stream and records the
per-task history; per-seed scalar metrics are then extracted and aggregated across
seeds with CIs, plus permutation tests against the key baselines.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from analysis.statistics import permutation_test, summarise
from core.seeding import set_global_seed
from data.streams import PermutedRegressionStream
from mechanisms.base import make_mechanism
from models.mlp import MLP
from training.continual import ContinualTrainer

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dim": 16,
    "teacher_hidden": 64,
    "num_tasks": 250,
    "task_length": 200,
    "batch_size": 1,
    "hidden_dim": 32,
    "layers": 2,
    "lr": 0.01,
}
METHODS = ["vanilla", "l2", "shrink_perturb", "redo", "continual_backprop", "homeostatic", "structural", "combined"]
PRIMARY_METRIC = "final_late_loss"


def _run_one(method: str, seed: int, cfg: Mapping[str, Any]) -> list[dict[str, float]]:
    set_global_seed(seed)
    stream = PermutedRegressionStream(
        input_dim=cfg["input_dim"], teacher_hidden=cfg["teacher_hidden"],
        num_tasks=cfg["num_tasks"], task_length=cfg["task_length"],
        batch_size=cfg["batch_size"], seed=seed,
    )
    model = MLP(cfg["input_dim"], cfg["hidden_dim"], num_hidden_layers=cfg["layers"])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    return ContinualTrainer(model, stream, optimizer, mechanism=make_mechanism(method)).run()


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
    return {
        "final_late_loss": final,
        "early_late_loss": early,
        "plasticity_ratio": final / max(1e-9, early),   # <1 = improves; >1 = loses plasticity
        "auc_late_loss": float(late.mean()),
        "final_dormant_fraction": float(dormant[-w:].mean()),
        "final_effective_rank": float(rank[-w:].mean()),
        "final_weight_magnitude": float(wmag[-w:].mean()),
    }


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
    """Permutation tests of each method vs each baseline on ``metric``."""
    values = {m: [per_seed_scalars(h, window)[metric] for h in hs] for m, hs in results.items()}
    tests: dict[str, dict[str, dict[str, float]]] = {}
    for method in results:
        tests[method] = {}
        for base in baselines:
            if base in values and method != base:
                tests[method][f"vs_{base}"] = permutation_test(values[method], values[base])
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
