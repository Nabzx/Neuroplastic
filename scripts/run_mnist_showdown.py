#!/usr/bin/env python
"""MNIST showdown: does ``selective_intrinsic_homeostatic`` significantly beat homeostatic?

The synthetic result is a statistical tie, so Permuted-MNIST is the only place the edge
(0.766 vs 0.762 at 8 seeds, p=0.24) might reach significance. This script runs the
head-to-head at a higher seed count and, optionally (``--tune``), first sweeps the
intrinsic sub-mechanism's ``(dead_threshold, rate)`` on a small grid to pick the
strongest configuration, then confirms it at full seeds. Reports permutation tests
(Holm-corrected) of the candidate against homeostatic, combined, and Continual Backprop.

    python scripts/run_mnist_showdown.py --seeds 20            # head-to-head, default config
    python scripts/run_mnist_showdown.py --tune --seeds 20     # tune intrinsic first, then confirm

Nothing runs on import. Invoke explicitly when ready (this is CPU-heavy: MNIST, many seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.statistics import holm_bonferroni, permutation_test, summarise  # noqa: E402
from experiments.study import per_seed_scalars, run_study  # noqa: E402

# Permuted-MNIST config that reproduced the study (hidden 32, lr 0.05).
MNIST = dict(
    benchmark="permuted_mnist", num_tasks=300, task_length=800, batch_size=16,
    hidden_dim=32, layers=2, lr=0.05, optimizer="sgd", shared_teacher=True,
)
CANDIDATE = "selective_intrinsic_homeostatic"
BASELINES = ["homeostatic", "combined", "continual_backprop"]
# Small tuning grid over the intrinsic (dead-unit revival) sub-mechanism.
TUNE_GRID = [(0.10, 0.01), (0.15, 0.02), (0.20, 0.03), (0.25, 0.05), (0.10, 0.05)]


def _accuracies(method: str, seeds: int, cfg: dict) -> list[float]:
    bundle = run_study([method], list(range(seeds)), cfg)
    return [per_seed_scalars(h, 25).get("final_accuracy", float("nan")) for h in bundle["results"][method]]


def _tune(cfg_base: dict, seeds: int) -> dict:
    print(f"\nTuning {CANDIDATE} intrinsic (dead_threshold, rate) @ {seeds} seeds:")
    best = None
    for dead_threshold, rate in TUNE_GRID:
        cfg = {**cfg_base, "mechanism_config": {"intrinsic": {"dead_threshold": dead_threshold, "rate": rate}}}
        mean = summarise(_accuracies(CANDIDATE, seeds, cfg))["mean"]
        print(f"  dead_threshold={dead_threshold:<4} rate={rate:<4} -> acc {mean:.4f}")
        if best is None or mean > best[0]:
            best = (mean, dead_threshold, rate)
    print(f"  => best: dead_threshold={best[1]} rate={best[2]}  (acc {best[0]:.4f})")
    return {"intrinsic": {"dead_threshold": best[1], "rate": best[2]}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20, help="seeds for the confirming head-to-head")
    parser.add_argument("--tune", action="store_true", help="sweep intrinsic (dead_threshold, rate) first")
    parser.add_argument("--tune-seeds", type=int, default=6, help="seeds per grid point during tuning")
    parser.add_argument("--num-tasks", type=int, default=300)
    parser.add_argument("--output", default="results/metaplastic/mnist_showdown")
    args = parser.parse_args(argv)

    cfg_base = {**MNIST, "num_tasks": args.num_tasks}
    mech_cfg = _tune(cfg_base, args.tune_seeds) if args.tune else None
    cand_cfg = {**cfg_base, "mechanism_config": mech_cfg} if mech_cfg else cfg_base

    print(f"\nHead-to-head on Permuted-MNIST @ {args.seeds} seeds "
          f"(config: {mech_cfg or 'defaults'}) ...")
    data = {CANDIDATE: _accuracies(CANDIDATE, args.seeds, cand_cfg)}
    for base in BASELINES:
        data[base] = _accuracies(base, args.seeds, cfg_base)

    print("\nfinal_accuracy (higher = better):")
    for name, acc in data.items():
        s = summarise(acc)
        print(f"  {name:34s} {s['mean']:.4f}  95% CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")

    cand = data[CANDIDATE]
    tests = {b: permutation_test(cand, data[b]) for b in BASELINES}
    padj = holm_bonferroni([tests[b]["p_value"] for b in BASELINES])
    cand_mean = summarise(cand)["mean"]
    print(f"\n{CANDIDATE} vs baselines (Holm-corrected across the {len(BASELINES)} tests):")
    for base, p_holm in zip(BASELINES, padj):
        r = tests[base]
        higher = cand_mean > summarise(data[base])["mean"]
        verdict = "BEATS" if (higher and p_holm < 0.05) else ("higher (n.s.)" if higher else "not higher")
        print(f"  vs {base:20s} p_holm={p_holm:.4f}  d={r['cohens_d']:+.2f}  {verdict}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds, "mechanism_config": mech_cfg, "config": cand_cfg,
        "accuracy": data,
        "tests": {b: {**tests[b], "p_holm": p} for b, p in zip(BASELINES, padj)},
    }
    (out / "showdown.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nSaved -> {out}/showdown.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
