#!/usr/bin/env python
"""A3 sensitivity: is a mechanism's advantage robust to its hyper-parameters?

Sweeps one hyper-parameter of one mechanism across a grid and reports the headline
metric (accuracy for MNIST, else late loss) at each value, with vanilla and
Continual Backprop reference levels. A flat curve well below (or above, for
accuracy) the SOTA reference across the range shows the result is not a lucky
setting.

    python scripts/run_sensitivity.py --method homeostatic --param rate \
        --values 0.02 0.05 0.1 0.2 0.4 --seeds 5
    python scripts/run_sensitivity.py --method structural --param replacement_rate \
        --values 1e-4 5e-4 1e-3 5e-3 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from analysis.statistics import summarise  # noqa: E402
from experiments.study import per_seed_scalars, run_study  # noqa: E402


def _summ(method, seeds, cfg, window, primary):
    bundle = run_study([method], list(range(seeds)), cfg)
    scalars = [per_seed_scalars(h, window).get(primary, float("nan")) for h in bundle["results"][method]]
    return summarise(scalars)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="homeostatic")
    parser.add_argument("--param", default="rate", help="mechanism hyper-parameter to sweep")
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--benchmark", default="permuted_regression", choices=["permuted_regression", "permuted_mnist"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=250)
    parser.add_argument("--task-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--output", default="results/sensitivity")
    args = parser.parse_args(argv)

    base = {
        "benchmark": args.benchmark, "num_tasks": args.num_tasks, "task_length": args.task_length,
        "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "lr": args.lr,
    }
    primary = "final_accuracy" if args.benchmark == "permuted_mnist" else "final_late_loss"
    higher_better = primary == "final_accuracy"

    swept = [_summ(args.method, args.seeds, {**base, "mechanism_config": {args.param: v}}, args.window, primary)
             for v in args.values]
    refs = {r: _summ(r, args.seeds, base, args.window, primary) for r in ("vanilla", "continual_backprop")}

    print(f"\nSensitivity of `{args.method}` to `{args.param}` ({primary}, {args.seeds} seeds):\n")
    print(f"{args.param:>12s} {'mean':>9s} {'std':>8s}")
    for v, s in zip(args.values, swept):
        print(f"{v:12g} {s['mean']:9.4f} {s['std']:8.4f}")
    print(f"\nreference  vanilla={refs['vanilla']['mean']:.4f}  continual_backprop={refs['continual_backprop']['mean']:.4f}")
    means = [s["mean"] for s in swept]
    beats = sum((m > refs["continual_backprop"]["mean"]) == higher_better for m in means)
    print(f"=> beats Continual Backprop at {beats}/{len(means)} settings "
          f"(spread {max(means) - min(means):.4f}); result is "
          f"{'robust' if beats == len(means) else 'sensitive'} to `{args.param}`.\n")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sensitivity_{args.method}_{args.param}.json").write_text(json.dumps(
        {"method": args.method, "param": args.param, "values": args.values,
         "swept": swept, "references": refs, "primary": primary}, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        lo = [s["mean"] - s["ci_low"] for s in swept]
        hi = [s["ci_high"] - s["mean"] for s in swept]
        ax.errorbar(args.values, means, yerr=[lo, hi], marker="o", capsize=4, linewidth=2, label=args.method)
        for r, c in (("vanilla", "#888"), ("continual_backprop", "#c0392b")):
            ax.axhline(refs[r]["mean"], color=c, linestyle="--", label=f"{r} (ref)")
        ax.set_xlabel(args.param)
        ax.set_ylabel(primary + (" (higher better)" if higher_better else " (lower better)"))
        ax.set_title(f"Sensitivity of {args.method} to {args.param}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(out / f"sensitivity_{args.method}_{args.param}.png", dpi=130, bbox_inches="tight")
        print(f"saved {out}/sensitivity_{args.method}_{args.param}.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
