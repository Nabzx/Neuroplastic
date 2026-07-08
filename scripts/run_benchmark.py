#!/usr/bin/env python
"""Run the full experimental benchmark suite and save all outputs.

Trains every (environment x communication method x seed) with an identical
budget, aggregates metrics across seeds with confidence intervals, runs
neuroplastic-vs-fixed significance tests, writes tables + JSON, generates
publication-style figures, and auto-writes an honest SUMMARY.md.

Usage
-----
    python scripts/run_benchmark.py --seeds 5 --steps 20000 --output results/benchmark

This is a *large* job (envs x methods x seeds runs); reduce --steps / --seeds to
calibrate on your hardware. Requires the RL extras (torch + PettingZoo) and
matplotlib for figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/inf and numpy scalars to JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):  # numpy scalar
        return _json_safe(obj.item())
    return obj


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"training.total_steps={args.steps}",
        f"training.rollout_length={args.rollout_length}",
        f"training.minibatch_size={args.rollout_length}",
        f"training.update_epochs={args.update_epochs}",
        f"training.log_every={args.steps}",  # quiet: one console line per run
        f"env.num_agents={args.agents}",
        f"env.max_cycles={args.max_cycles}",
        f"agent.hidden_dim={args.hidden_dim}",
        f"agent.obs_embedding_dim={args.obs_embedding_dim}",
        f"communication.message_dim={args.message_dim}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    parser.add_argument("--steps", type=int, default=20_000, help="env steps per run (identical for all)")
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--agents", type=int, default=4, help="agents for simple_spread")
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--obs-embedding-dim", type=int, default=32)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--spec-episodes", type=int, default=20)
    parser.add_argument("--envs", nargs="+", default=None, help="subset of environments")
    parser.add_argument("--methods", nargs="+", default=None, help="subset of method names")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from analysis.aggregate import aggregate, build_summary_tables, per_seed_rows, significance_tests, all_metric_names
    from analysis.interpret import generate_interpretation
    from training.benchmark import ENVIRONMENTS, METHODS, run_benchmark

    envs = args.envs or ENVIRONMENTS
    methods = {m: METHODS[m] for m in (args.methods or METHODS) if m in METHODS}
    seeds = list(range(args.seeds))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    bundle = run_benchmark(envs, methods, seeds, overrides=_overrides(args), spec_episodes=args.spec_episodes)
    results = bundle["results"]
    summary = aggregate(results)
    significance = significance_tests(results)
    meta = {"methods": list(methods), "seeds": args.seeds, "steps": args.steps, "environments": envs}

    # -- outputs ----------------------------------------------------------
    (output / "comparison.json").write_text(
        json.dumps(_json_safe({"meta": meta, "results": results, "summary": summary, "significance": significance}), indent=2)
    )
    (output / "summary_statistics.json").write_text(
        json.dumps(_json_safe({"meta": meta, "summary": summary, "significance": significance}), indent=2)
    )

    metric_names = all_metric_names(results)
    _write_csv(output / "comparison.csv", per_seed_rows(results), ["environment", "method", "seed_index", *metric_names])

    table_rows = build_summary_tables(summary)
    table_fields = ["environment", "metric", *list(methods)]
    _write_csv(output / "summary_tables.csv", table_rows, table_fields)

    if not args.no_figures:
        try:
            from visualisation.benchmark_figures import generate_benchmark_figures

            figures = generate_benchmark_figures(bundle, summary, output)
            print(f"Saved {len(figures)} figures.")
        except Exception as exc:  # pragma: no cover
            print(f"[warning] figure generation failed: {exc}")

    interpretation = generate_interpretation(summary, significance, meta)
    (output / "SUMMARY.md").write_text(interpretation + "\n")

    print(f"\nBenchmark outputs written to {output}/")
    print("  comparison.json  comparison.csv  summary_tables.csv  summary_statistics.json  SUMMARY.md")
    print("\n" + "=" * 70 + "\n" + interpretation.split("## Does")[0].strip())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
