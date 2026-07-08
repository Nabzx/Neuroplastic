#!/usr/bin/env python
"""Evaluate a single saved run directory (no training / torch required).

Reads ``metrics.csv`` and ``edge_weights.npz`` from a run directory and prints the
coordination, communication-graph and edge-weight-stability metrics.

Usage
-----
    python scripts/evaluate_run.py runs/plastic_communication
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.report import evaluate_run, flatten_report, load_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="a run directory containing metrics.csv (+ edge_weights.npz)")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true", help="print the raw JSON report instead of a table")
    args = parser.parse_args(argv)

    returns, steps, snapshots, agent_ids = load_run(args.run_dir)
    report = evaluate_run(returns, steps, snapshots, agent_ids, threshold=args.threshold)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Run: {args.run_dir}\n")
        for section, metrics in report.items():
            print(f"[{section}]")
            for name, value in metrics.items():
                print(f"  {name:<28} {float(value):.4g}")
            print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
