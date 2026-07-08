"""Training callbacks: hook points around the (future) optimisation loop.

Callbacks keep cross-cutting concerns (logging, checkpointing, periodic snapshots
of the interaction graph for later analysis) out of the core loop. The base
class defines no-op hooks so concrete callbacks override only what they need.
"""

from __future__ import annotations

from typing import Any

from core.logging_utils import get_logger


class Callback:
    """Base callback with no-op hooks."""

    def on_train_start(self, trainer: Any) -> None: ...
    def on_step(self, trainer: Any, step: int, metrics: dict[str, float]) -> None: ...
    def on_update(self, trainer: Any, step: int, metrics: dict[str, float]) -> None: ...
    def on_episode_end(self, trainer: Any, episode: int) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...


class LoggingCallback(Callback):
    """Log scalar metrics every ``log_every`` steps."""

    def __init__(self, log_every: int = 1000) -> None:
        self.log_every = log_every
        self.logger = get_logger("nci.training")

    def on_update(self, trainer, step, metrics) -> None:
        if step % max(1, self.log_every) == 0:
            joined = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            self.logger.info("step=%d | %s", step, joined)


class TopologySnapshotCallback(Callback):  # pragma: no cover - deferred
    """Periodically snapshot the interaction graph for offline analysis."""

    def __init__(self, every: int = 1000) -> None:
        self.every = every

    def on_step(self, trainer, step, metrics) -> None:
        raise NotImplementedError(
            "TopologySnapshotCallback is a placeholder; snapshot storage is "
            "specified in docs/experiment_plan.md."
        )


class CheckpointCallback(Callback):  # pragma: no cover - deferred
    """Save model/optimiser state every ``every`` steps."""

    def __init__(self, every: int = 50000) -> None:
        self.every = every

    def on_update(self, trainer, step, metrics) -> None:
        raise NotImplementedError(
            "CheckpointCallback is a placeholder; checkpoint format is deferred "
            "to the training milestone."
        )


__all__ = [
    "Callback",
    "LoggingCallback",
    "TopologySnapshotCallback",
    "CheckpointCallback",
]
