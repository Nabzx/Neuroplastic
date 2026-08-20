#!/usr/bin/env python
"""Run the full multi-seed mechanism study and save all outputs + figures.

    python scripts/run_study.py --seeds 8 --num-tasks 250 --output results/study
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

from core.logging_utils import get_logger  # noqa: E402


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


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="permuted_regression", choices=["permuted_regression", "permuted_mnist"])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=250)
    parser.add_argument("--task-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", default="results/study")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from analysis.interpret import generate_interpretation
    from experiments.study import METHODS, aggregate_study, per_seed_scalars, run_study, significance_study

    logger = get_logger("nci.study")
    methods = args.methods or METHODS
    seeds = list(range(args.seeds))
    config = {
        "benchmark": args.benchmark, "num_tasks": args.num_tasks, "task_length": args.task_length,
        "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "lr": args.lr,
    }

    bundle = run_study(methods, seeds, config, logger=logger)
    results = bundle["results"]
    summary = aggregate_study(results, window=args.window)
    significance = significance_study(results, window=args.window)
    meta = {"methods": methods, "seeds": args.seeds, "num_tasks": args.num_tasks, "benchmark": args.benchmark, "config": bundle["config"]}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_statistics.json").write_text(json.dumps(_json_safe({"meta": meta, "summary": summary, "significance": significance}), indent=2))

    # per-seed scalar rows
    seed_rows = []
    for method, histories in results.items():
        for si, h in enumerate(histories):
            seed_rows.append({"method": method, "seed": si, **per_seed_scalars(h, args.window)})
    metric_names = sorted({k for r in seed_rows for k in r if k not in ("method", "seed")})
    _write_csv(out / "comparison.csv", seed_rows, ["method", "seed", *metric_names])
    (out / "comparison.json").write_text(json.dumps(_json_safe({"meta": meta, "per_seed": seed_rows, "summary": summary, "significance": significance}), indent=2))

    # metric x method table (mean ± std)
    table = []
    for metric in metric_names:
        row = {"metric": metric}
        for method in methods:
            s = summary[method].get(metric, {})
            row[method] = f"{s.get('mean', float('nan')):.4g} ± {s.get('std', float('nan')):.3g}"
        table.append(row)
    _write_csv(out / "summary_tables.csv", table, ["metric", *methods])

    if not args.no_figures:
        try:
            from visualisation.plasticity_figures import generate_study_figures

            figs = generate_study_figures(results, summary, out)
            print(f"Saved {len(figs)} figures.")
        except Exception as exc:  # pragma: no cover
            print(f"[warning] figures failed: {exc}")

    interpretation = generate_interpretation(summary, significance, meta)
    (out / "SUMMARY.md").write_text(interpretation + "\n")
    print(f"\nStudy outputs written to {out}/\n" + "=" * 70)
    print(interpretation.split("## H1")[0].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
