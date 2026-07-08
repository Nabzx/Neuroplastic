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
    """PPO baseline marker for config resolution / dry-run.

    The concrete, runnable implementation is
    :class:`training.learner.SharedPPOLearner` (parameter-shared PPO). This class
    is a torch-free registry entry so ``describe()`` / ``--dry-run`` can resolve
    ``training.algorithm: ippo`` without importing torch; the trainer builds the
    learner directly.
    """

    def update(self, batch: Any) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError(
            "Use training.learner.SharedPPOLearner (built by Trainer.build); this "
            "class is only a registry marker for config resolution."
        )


def make_algorithm(config: Any) -> Algorithm:
    """Construct the algorithm named by ``config.algorithm``."""
    return ALGORITHM_REGISTRY.get(config.algorithm)(config)


__all__ = ["Algorithm", "IPPO", "ALGORITHM_REGISTRY", "make_algorithm"]
