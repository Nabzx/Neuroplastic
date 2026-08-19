#!/usr/bin/env python
"""Gate 1: reproduce loss of plasticity with a vanilla network.

Trains a plain MLP online over a long sequence of permuted-regression tasks and
checks whether its per-task fitting ability *degrades* over the sequence -- the
loss-of-plasticity phenomenon. Prints an early-vs-late comparison (averaged over
seeds) and, if matplotlib is available, saves diagnostic curves.

    python scripts/reproduce_plasticity_loss.py --seeds 3 --num-tasks 300
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
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402


def run_one(seed: int, args) -> list[dict]:
    set_global_seed(seed)
    stream = PermutedRegressionStream(
        input_dim=args.input_dim,
        teacher_hidden=args.teacher_hidden,
        num_tasks=args.num_tasks,
        task_length=args.task_length,
        batch_size=args.batch_size,
        seed=seed,
    )
    model = MLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_hidden_layers=args.layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    return ContinualTrainer(model, stream, optimizer).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-tasks", type=int, default=300)
    parser.add_argument("--task-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--teacher-hidden", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=20, help="tasks averaged at each end for the comparison")
    parser.add_argument("--output", default="results/gate1")
    args = parser.parse_args(argv)

    histories = [run_one(s, args) for s in range(args.seeds)]
    late = np.array([[r["train_loss_late"] for r in h] for h in histories])  # [seeds, tasks]
    dormant = np.array([[r["dormant_fraction"] for r in h] for h in histories])
    rank = np.array([[r["effective_rank"] for r in h] for h in histories])
    wmag = np.array([[r["weight_magnitude"] for r in h] for h in histories])

    w = args.window
    early = late[:, :w].mean()
    final = late[:, -w:].mean()
    ratio = final / max(1e-9, early)
    print(f"\nVanilla SGD over {args.num_tasks} permuted-regression tasks, {args.seeds} seeds:")
    print(f"  per-task late loss   first {w} tasks: {early:.4f}   last {w} tasks: {final:.4f}   ratio: {ratio:.2f}x")
    print(f"  dormant fraction     first: {dormant[:, :w].mean():.3f}   last: {dormant[:, -w:].mean():.3f}")
    print(f"  effective rank       first: {rank[:, :w].mean():.2f}    last: {rank[:, -w:].mean():.2f}")
    print(f"  weight magnitude     first: {wmag[:, :w].mean():.4f}   last: {wmag[:, -w:].mean():.4f}")
    verdict = "LOSS OF PLASTICITY reproduced" if ratio > 1.15 else "no clear plasticity loss"
    print(f"  => {verdict} (late-loss ratio {ratio:.2f}x)\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        x = np.arange(late.shape[1])
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, data, title in [
            (axes[0, 0], late, "per-task late loss (lower = plastic)"),
            (axes[0, 1], dormant, "dormant unit fraction"),
            (axes[1, 0], rank, "effective rank"),
            (axes[1, 1], wmag, "mean |weight|"),
        ]:
            m, sd = data.mean(0), data.std(0)
            ax.plot(x, m, color="#c44")
            ax.fill_between(x, m - sd, m + sd, alpha=0.2, color="#c44")
            ax.set_title(title)
            ax.set_xlabel("task")
            ax.grid(alpha=0.3)
        fig.suptitle("Loss of plasticity — vanilla SGD")
        fig.savefig(out / "plasticity_loss_vanilla.png", dpi=120, bbox_inches="tight")
        print(f"  saved {out}/plasticity_loss_vanilla.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
