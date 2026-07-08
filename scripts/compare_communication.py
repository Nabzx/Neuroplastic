#!/usr/bin/env python
"""Compare the three communication settings on a cooperative benchmark.

Trains, then evaluates, each of:

    * no communication            (configs/baselines/no_comm.yaml)
    * fixed communication         (configs/baselines/fully_connected.yaml)
    * adaptive neuroplastic comm. (configs/plastic.yaml)

with an identical agent/budget/seed, and prints + saves a side-by-side table of
coordination, communication-graph and edge-weight-stability metrics.

Usage
-----
    python scripts/compare_communication.py --steps 150000 --agents 5 --output runs/compare

Requires the RL extras (torch + PettingZoo). For evaluating an already-saved run
without training, see ``scripts/evaluate_run.py`` (no torch needed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make the project importable when run directly (python scripts/....py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.report import flatten_report, format_comparison  # noqa: E402


def _common_overrides(args: argparse.Namespace) -> list[str]:
    """Overrides applied to every setting so the comparison is fair."""
    return [
        f"seed={args.seed}",
        f"training.total_steps={args.steps}",
        f"training.rollout_length={args.rollout_length}",
        f"training.minibatch_size={args.rollout_length}",
        f"env.num_agents={args.agents}",
        f"env.max_cycles={args.max_cycles}",
        f"agent.hidden_dim={args.hidden_dim}",
        f"communication.message_dim={args.message_dim}",
    ]


def _save(reports: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(reports, indent=2))

    flats = {name: flatten_report(rep) for name, rep in reports.items()}
    metrics = list(next(iter(flats.values())).keys()) if flats else []
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *flats.keys()])
        for metric in metrics:
            writer.writerow([metric, *(flats[name].get(metric, "") for name in flats)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=150_000, help="env steps per setting")
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1e-3, help="edge weight cutoff for graph metrics")
    parser.add_argument("--output", default="runs/compare", help="directory for comparison.{json,csv}")
    args = parser.parse_args(argv)

    # Imported here so --help works without the RL stack installed.
    from training.compare import DEFAULT_SETTINGS, compare_settings

    overrides = _common_overrides(args)
    print(f"Comparing {list(DEFAULT_SETTINGS)} | {args.steps} steps, {args.agents} agents, seed {args.seed}\n")
    reports = compare_settings(DEFAULT_SETTINGS, overrides, threshold=args.threshold)

    print(format_comparison(reports))
    _save(reports, Path(args.output))
    print(f"\nSaved comparison to {args.output}/comparison.{{json,csv}}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
