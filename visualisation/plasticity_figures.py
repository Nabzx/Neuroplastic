"""Matplotlib figures for the mechanism study.

* ``plasticity_curves.png`` — per-task late loss / dormant fraction / effective
  rank / weight magnitude over the task sequence (mean +/- 95% CI, all methods).
* ``final_late_loss.png`` — bar chart of retained plasticity per method.
* ``mechanism_attribution.png`` — final dormant fraction vs final late loss,
  showing which failure mode each mechanism repairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.study import task_curves

_ORDER = ["vanilla", "l2", "shrink_perturb", "redo", "continual_backprop", "homeostatic", "structural", "combined"]


def _ordered(methods) -> list[str]:
    present = list(methods)
    return [m for m in _ORDER if m in present] + [m for m in present if m not in _ORDER]


def _save(fig, path, saved):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def generate_study_figures(results: Mapping[str, list], summary: Mapping[str, Any], output_dir) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = _ordered(results)
    saved: list[Path] = []

    # 1. per-task curves (2x2): late loss, dormant, rank, weight
    panels = [
        ("train_loss_late", "per-task late loss (lower = plastic)"),
        ("dormant_fraction", "dormant unit fraction"),
        ("effective_rank", "effective rank"),
        ("weight_magnitude", "mean |weight|"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        curves = task_curves(results, key=key)
        for method in methods:
            mean, lo, hi = curves[method]
            x = np.arange(len(mean))
            (line,) = ax.plot(x, mean, label=method, linewidth=1.7)
            ax.fill_between(x, lo, hi, alpha=0.12, color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel("task")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Plasticity over a task sequence, by mechanism")
    _save(fig, out / "plasticity_curves.png", saved)

    # 2. final late loss bars (retained plasticity)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = [summary[m]["final_late_loss"]["mean"] for m in methods]
    errs = [summary[m]["final_late_loss"]["std"] for m in methods]
    colors = ["#888" if m in ("vanilla", "l2", "shrink_perturb", "redo", "continual_backprop") else "#c0392b" for m in methods]
    ax.bar(range(len(methods)), means, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("final per-task late loss")
    ax.set_title("Retained plasticity (lower is better; red = biological/ours)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out / "final_late_loss.png", saved)

    # 2b. retained accuracy bars (classification benchmarks)
    if any("final_accuracy" in summary[m] for m in methods):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        acc = [summary[m].get("final_accuracy", {}).get("mean", float("nan")) for m in methods]
        acc_err = [summary[m].get("final_accuracy", {}).get("std", float("nan")) for m in methods]
        ax.bar(range(len(methods)), acc, yerr=acc_err, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("final retained accuracy")
        ax.set_title("Retained accuracy (higher is better; red = biological/ours)")
        ax.grid(axis="y", alpha=0.3)
        _save(fig, out / "final_accuracy.png", saved)

    # 3. mechanism attribution: dormant fraction vs late loss
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        x = summary[method]["final_dormant_fraction"]["mean"]
        y = summary[method]["final_late_loss"]["mean"]
        ax.scatter(x, y, s=80)
        ax.annotate(method, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("final dormant unit fraction")
    ax.set_ylabel("final per-task late loss")
    ax.set_title("Mechanism attribution: dormant units vs retained plasticity")
    ax.grid(alpha=0.3)
    _save(fig, out / "mechanism_attribution.png", saved)

    return saved


__all__ = ["generate_study_figures"]
