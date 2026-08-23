#!/usr/bin/env python
"""A2 robustness sweep: re-run the mechanism study across optimisers x net sizes.

Checks whether homeostatic scaling's advantage over the SOTA remedy survives Adam
(whose momentum accumulation is where Continual Backprop's edge is strongest) and
larger networks. Each (optimiser, hidden-dim) combination is a full study written
to its own subdirectory, reusing scripts/run_study.py unchanged.

    python scripts/run_robustness.py --seeds 8 --optimizers sgd adam --hidden-dims 32 128
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="permuted_regression", choices=["permuted_regression", "permuted_mnist"])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=250)
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "adam"])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--methods", nargs="+", default=None, help="subset of methods for each study")
    parser.add_argument("--output", default="results/robustness")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra args forwarded verbatim to run_study.py (e.g. --lr 0.05 --batch-size 16)")
    args = parser.parse_args(argv)

    out = Path(args.output)
    combos = list(itertools.product(args.optimizers, args.hidden_dims))
    print(f"Robustness sweep: {len(combos)} studies ({args.benchmark}, {args.seeds} seeds each)\n")
    for optimizer, hidden in combos:
        sub = out / f"{args.benchmark}_{optimizer}_hid{hidden}"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_study.py"),
            "--benchmark", args.benchmark, "--seeds", str(args.seeds),
            "--num-tasks", str(args.num_tasks), "--optimizer", optimizer,
            "--hidden-dim", str(hidden), "--output", str(sub),
        ]
        if args.methods:
            cmd += ["--methods", *args.methods]
        cmd += list(args.extra)
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    print(f"\nSweep complete -> {out}/ (one study per optimiser x hidden-dim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
