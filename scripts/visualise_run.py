#!/usr/bin/env python
"""Generate figures from saved experiment logs (matplotlib only).

Reads ``metrics.csv`` + ``edge_weights.npz`` from one or more run directories and
writes the standard figure set (reward / density / entropy curves, edge-weight
heatmaps, communication-graph snapshots, hub-centrality-over-time) to an output
directory.

Usage
-----
    # single run
    python scripts/visualise_run.py runs/plastic_communication

    # compare several runs on the shared curves, per-run graph figures for each
    python scripts/visualise_run.py runs/baseline_no_comm runs/baseline_fully_connected \
        runs/plastic_communication --output results/figures

No training or PyTorch required -- only matplotlib, NumPy and the saved logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualisation.run_figures import DEFAULT_OUTPUT, visualise_runs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="run directories containing metrics.csv (+ edge_weights.npz)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"figure output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--labels", nargs="+", default=None, help="labels for each run (default: directory names)")
    parser.add_argument("--threshold", type=float, default=1e-3, help="edge weight cutoff for graph drawing")
    parser.add_argument("--smooth", type=int, default=5, help="moving-average window for reward curves")
    args = parser.parse_args(argv)

    if args.labels and len(args.labels) != len(args.run_dirs):
        parser.error("--labels must have the same number of entries as run_dirs")

    saved = visualise_runs(
        args.run_dirs, output_dir=args.output, labels=args.labels,
        threshold=args.threshold, smooth=args.smooth,
    )
    print(f"Saved {len(saved)} figures to {args.output}:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
