"""MARL algorithm interface.

The training milestone will implement Independent PPO (IPPO) as the baseline
learner: each agent optimises its own clipped-surrogate objective while sharing
the communication substrate. Algorithms register under ``ALGORITHM_REGISTRY``
and are selected via ``config.training.algorithm``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.registry import Registry

ALGORITHM_REGISTRY: Registry[type["Algorithm"]] = Registry("algorithm")


class Algorithm(ABC):
    """Base class for multi-agent learning algorithms."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Run one optimisation step on ``batch``; return scalar metrics."""


@ALGORITHM_REGISTRY.register("ippo")
class IPPO(Algorithm):
    """Independent PPO baseline (placeholder)."""

    def update(self, batch: Any) -> dict[str, float]:  # pragma: no cover - deferred
        raise NotImplementedError(
            "IPPO.update is a placeholder; the PPO objective is specified in "
            "docs/experiment_plan.md and will be implemented in the training "
            "milestone."
        )


def make_algorithm(config: Any) -> Algorithm:
    """Construct the algorithm named by ``config.algorithm``."""
    return ALGORITHM_REGISTRY.get(config.algorithm)(config)


__all__ = ["Algorithm", "IPPO", "ALGORITHM_REGISTRY", "make_algorithm"]
