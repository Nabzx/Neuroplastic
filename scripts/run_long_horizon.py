#!/usr/bin/env python
"""Long-horizon benchmark: does more training change the negative result?

Sweeps *only* the training budget (default 20k / 100k / 250k env steps) on
**simple_spread**, comparing fixed fully-connected, adaptive and neuroplastic
communication with multiple seeds. Nothing else changes -- same algorithm,
architecture, PPO settings, plasticity coefficients and communication
mechanisms. Writes budget-indexed tables, stats, figures and an honest SUMMARY.md
to a separate directory (does not touch the original benchmark).

Usage
-----
    python scripts/run_long_horizon.py --seeds 5 --budgets 20000 100000 250000 \
        --output results/long_horizon_benchmark
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

# Methods compared (a subset of the full benchmark -- exactly the three requested).
METHOD_CONFIGS = {
    "fully_connected": "configs/baselines/fully_connected.yaml",
    "adaptive": "configs/adaptive.yaml",
    "neuroplastic": "configs/plastic.yaml",
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item"):
        return _json_safe(obj.item())
    return obj


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fixed_overrides(args: argparse.Namespace) -> list[str]:
    """Everything except the budget -- identical to the original benchmark."""
    return [
        f"training.rollout_length={args.rollout_length}",
        f"training.minibatch_size={args.rollout_length}",
        f"training.update_epochs={args.update_epochs}",
        f"env.num_agents={args.agents}",
        f"env.max_cycles={args.max_cycles}",
        f"agent.hidden_dim={args.hidden_dim}",
        f"agent.obs_embedding_dim={args.obs_embedding_dim}",
        f"communication.message_dim={args.message_dim}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20000, 100000, 250000])
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--obs-embedding-dim", type=int, default=32)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--spec-episodes", type=int, default=20)
    parser.add_argument("--output", default="results/long_horizon_benchmark")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from analysis.aggregate import aggregate, all_metric_names, build_summary_tables, per_seed_rows, significance_tests
    from analysis.interpret import generate_long_horizon_interpretation
    from training.benchmark import run_benchmark

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    base = _fixed_overrides(args)

    results_by_budget: dict[str, Any] = {}
    curves_by_budget: dict[str, Any] = {}
    finals_by_budget: dict[str, Any] = {}
    for budget in args.budgets:
        overrides = base + [f"training.total_steps={budget}", f"training.log_every={budget}"]
        bundle = run_benchmark(["simple_spread"], METHOD_CONFIGS, seeds, overrides=overrides, spec_episodes=args.spec_episodes)
        results_by_budget[str(budget)] = bundle["results"]["simple_spread"]
        curves_by_budget[str(budget)] = bundle["curves"]["simple_spread"]
        finals_by_budget[str(budget)] = bundle["finals"]["simple_spread"]

    summary = aggregate(results_by_budget)
    sig_fixed = significance_tests(results_by_budget, "neuroplastic", "fully_connected")
    sig_adaptive = significance_tests(results_by_budget, "neuroplastic", "adaptive")
    significance = {
        b: {"vs_fixed": sig_fixed.get(b, {}), "vs_adaptive": sig_adaptive.get(b, {})}
        for b in results_by_budget
    }
    meta = {"methods": list(METHOD_CONFIGS), "seeds": args.seeds, "budgets": args.budgets, "environment": "simple_spread"}

    # -- outputs (relabel the "environment" axis as "budget") -------------
    (output / "comparison.json").write_text(
        json.dumps(_json_safe({"meta": meta, "results": results_by_budget, "summary": summary, "significance": significance}), indent=2)
    )
    (output / "summary_statistics.json").write_text(
        json.dumps(_json_safe({"meta": meta, "summary": summary, "significance": significance}), indent=2)
    )

    seed_rows = [{"budget": r.pop("environment"), **r} for r in per_seed_rows(results_by_budget)]
    _write_csv(output / "comparison.csv", seed_rows, ["budget", "method", "seed_index", *all_metric_names(results_by_budget)])

    table_rows = [{"budget": r.pop("environment"), **r} for r in build_summary_tables(summary)]
    _write_csv(output / "summary_tables.csv", table_rows, ["budget", "metric", *list(METHOD_CONFIGS)])

    if not args.no_figures:
        try:
            from visualisation.benchmark_figures import generate_long_horizon_figures

            figures = generate_long_horizon_figures(curves_by_budget, finals_by_budget, summary, args.budgets, output)
            print(f"Saved {len(figures)} figures.")
        except Exception as exc:  # pragma: no cover
            print(f"[warning] figure generation failed: {exc}")

    interpretation = generate_long_horizon_interpretation(summary, significance, meta)
    (output / "SUMMARY.md").write_text(interpretation + "\n")

    print(f"\nLong-horizon outputs written to {output}/")
    print("  comparison.json  comparison.csv  summary_tables.csv  summary_statistics.json  SUMMARY.md")
    print("\n" + "=" * 70 + "\n" + interpretation.split("## Limitations")[0].strip())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
