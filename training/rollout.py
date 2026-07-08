"""Experience collection and storage (placeholder).

The :class:`RolloutBuffer` will store per-agent transitions plus the extra
signals this project needs for analysis: the interaction-graph snapshot and the
messages exchanged at each step, so that topology/information metrics can be
computed post-hoc from recorded rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutBuffer:
    """Fixed-size, per-agent transition store (placeholder).

    Only the intended shape is sketched here; storage/GAE logic arrives with the
    training milestone.
    """

    capacity: int
    _storage: list[Any] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:  # pragma: no cover - deferred
        raise NotImplementedError(
            "RolloutBuffer.add is a placeholder; the transition schema (incl. "
            "graph snapshots and messages) is specified in docs/experiment_plan.md."
        )

    def compute_returns(self, *args, **kwargs) -> None:  # pragma: no cover - deferred
        raise NotImplementedError("GAE/return computation is deferred.")

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:
        return len(self._storage)


class RolloutWorker:  # pragma: no cover - deferred
    """Runs an environment forward, collecting experience into a buffer."""

    def __init__(self, env: Any, agents: Any, channel: Any) -> None:
        self.env = env
        self.agents = agents
        self.channel = channel

    def collect(self, num_steps: int) -> RolloutBuffer:
        raise NotImplementedError(
            "RolloutWorker.collect is a placeholder; the env<->agent<->channel "
            "step loop is specified in docs/experiment_plan.md."
        )


__all__ = ["RolloutBuffer", "RolloutWorker"]
