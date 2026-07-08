"""Command-line entrypoint: ``nci``.

Examples
--------
Validate a config and print the resolved component wiring (no torch needed)::

    nci --config configs/default.yaml --dry-run
    nci --config configs/ablations/no_plasticity.yaml -o seed=1 --dry-run

Launch training (once implemented)::

    nci --config configs/default.yaml -o training.total_steps=2000000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from configs.loader import load_config
from training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nci",
        description="Neuroplastic Collective Intelligence experiment runner.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML experiment config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. -o training.lr=1e-4 (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the wiring without training (no torch required).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, overrides=args.override)
    trainer = Trainer(config)

    if args.dry_run:
        json.dump(trainer.describe(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    trainer.build()
    trainer.train()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
