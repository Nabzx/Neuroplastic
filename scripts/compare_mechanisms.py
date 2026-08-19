#!/usr/bin/env python
"""Compare plasticity mechanisms on the permuted-regression stream.

Runs each mechanism (vanilla + baselines + biological) over the task sequence for
several seeds and reports whether per-task fitting ability is preserved
(early-vs-late loss) plus the final dormant fraction and effective rank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.seeding import set_global_seed  # noqa: E402
from data.streams import PermutedRegressionStream  # noqa: E402
from mechanisms.base import make_mechanism  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402

METHODS = ["vanilla", "l2", "shrink_perturb", "redo", "continual_backprop", "homeostatic", "structural", "combined"]


def run(method: str, seed: int, args) -> list[dict]:
    set_global_seed(seed)
    stream = PermutedRegressionStream(
        input_dim=args.input_dim, teacher_hidden=args.teacher_hidden,
        num_tasks=args.num_tasks, task_length=args.task_length, seed=seed,
    )
    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_hidden_layers=args.layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    mech = make_mechanism(method)
    return ContinualTrainer(model, stream, optimizer, mechanism=mech).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=200)
    parser.add_argument("--task-length", type=int, default=200)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--teacher-hidden", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args(argv)

    w = args.window
    print(f"\n{'method':18s} {'early':>8s} {'late':>8s} {'ratio':>7s} {'dormant':>8s} {'rank':>7s}")
    print("-" * 60)
    for method in args.methods:
        hs = [run(method, s, args) for s in range(args.seeds)]
        late = np.array([[r["train_loss_late"] for r in h] for h in hs])
        dormant = np.array([[r["dormant_fraction"] for r in h] for h in hs])
        rank = np.array([[r["effective_rank"] for r in h] for h in hs])
        early, final = late[:, :w].mean(), late[:, -w:].mean()
        print(f"{method:18s} {early:8.4f} {final:8.4f} {final/max(1e-9,early):7.2f} "
              f"{dormant[:, -w:].mean():8.3f} {rank[:, -w:].mean():7.2f}")
    print("\n(ratio ~1 = plasticity preserved; >1 = plasticity lost. lower late loss is better.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
