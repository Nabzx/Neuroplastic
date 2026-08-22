#!/usr/bin/env python
"""A1 probe: is homeostatic's 'improvement over time' genuine plasticity?

Homeostatic scaling shows a plasticity ratio < 1 (per-task loss *falls* over the
sequence). This could be genuine retained plasticity, OR mere accumulation of the
structure shared across tasks (a single teacher / the same MNIST images). We
disentangle by comparing the ratio on the SHARED-teacher stream against an
INDEPENDENT-teacher stream (a fresh random teacher per task, so there is no shared
structure to accumulate):

* ratio stays < 1 with independent teachers  -> genuine plasticity retention.
* ratio rises toward / above 1               -> the apparent gain was shared-structure
  accumulation (an artefact of the benchmark), and should be reported as such.

    python scripts/probe_plasticity_gain.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from experiments.study import per_seed_scalars, run_study  # noqa: E402


def _ratios(methods, seeds, shared_teacher, args) -> dict[str, tuple[float, float]]:
    cfg = {
        "benchmark": "permuted_regression",
        "num_tasks": args.num_tasks, "task_length": args.task_length,
        "hidden_dim": args.hidden_dim, "lr": args.lr, "shared_teacher": shared_teacher,
    }
    bundle = run_study(methods, list(range(seeds)), cfg)
    out = {}
    for method, histories in bundle["results"].items():
        r = [per_seed_scalars(h, args.window)["plasticity_ratio"] for h in histories]
        out[method] = (float(np.mean(r)), float(np.std(r)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=250)
    parser.add_argument("--task-length", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--methods", nargs="+", default=["vanilla", "homeostatic", "combined"])
    args = parser.parse_args(argv)

    shared = _ratios(args.methods, args.seeds, True, args)
    indep = _ratios(args.methods, args.seeds, False, args)

    print(f"\nPlasticity ratio (final/early late loss; <1 = improves), {args.seeds} seeds:\n")
    print(f"{'method':18s} {'shared teacher':>16s} {'independent teachers':>22s}")
    print("-" * 58)
    for m in args.methods:
        sm, ss = shared[m]
        im, iss = indep[m]
        print(f"{m:18s} {sm:9.3f}±{ss:5.3f} {im:14.3f}±{iss:5.3f}")

    hm = indep.get("homeostatic", (float('nan'),))[0]
    print()
    if hm == hm and hm < 0.98:
        print("=> Homeostatic still improves with INDEPENDENT teachers: the plasticity")
        print("   gain is genuine, not shared-structure accumulation.")
    elif hm == hm:
        print("=> With independent teachers the improvement disappears (ratio ~>= 1):")
        print("   the apparent gain was accumulation of shared structure -- report honestly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
