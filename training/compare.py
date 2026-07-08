"""Train communication settings and evaluate them for comparison.

Orchestration only: it drives :class:`~training.trainer.Trainer` for each setting
and hands the results to :func:`evaluation.report.evaluate_run`. Kept in
``training`` (not ``evaluation``) so the metric library stays free of any
training/torch dependency.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from configs import load_config
from evaluation.report import evaluate_run
from training.trainer import Trainer

#: The three settings the project compares.
DEFAULT_SETTINGS: dict[str, str] = {
    "no_comm": "configs/baselines/no_comm.yaml",
    "fixed": "configs/baselines/fully_connected.yaml",
    "neuroplastic": "configs/plastic.yaml",
}


def evaluate_trainer(trainer: Trainer, threshold: float = 1e-3) -> dict[str, dict[str, float]]:
    """Extract a trained :class:`Trainer`'s history/graph snapshots into a report."""
    returns = [row["episode_return_mean"] for row in trainer.history]
    steps = [row["env_steps"] for row in trainer.history]
    snapshots = (
        np.stack([matrix for _, matrix in trainer.graph_history])
        if trainer.graph_history
        else None
    )
    return evaluate_run(returns, steps, snapshots, trainer.agent_ids, threshold=threshold)


def run_setting(
    config_path: str,
    overrides: Iterable[str] | None = None,
    threshold: float = 1e-3,
) -> dict[str, dict[str, float]]:
    """Train a single config and return its evaluation report."""
    trainer = Trainer(load_config(config_path, overrides=list(overrides or [])))
    trainer.train()
    return evaluate_trainer(trainer, threshold=threshold)


def compare_settings(
    settings: Mapping[str, str] | None = None,
    overrides: Iterable[str] | None = None,
    threshold: float = 1e-3,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate each setting; return ``{setting_name: report}``.

    The same ``overrides`` are applied to every setting so the comparison is fair
    (identical agents, budget and seed).
    """
    settings = dict(settings or DEFAULT_SETTINGS)
    overrides = list(overrides or [])
    return {name: run_setting(path, overrides, threshold) for name, path in settings.items()}


__all__ = ["DEFAULT_SETTINGS", "evaluate_trainer", "run_setting", "compare_settings"]
