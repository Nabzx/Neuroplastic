"""Cooperative multi-agent environments (PettingZoo / SuperSuit).

* :mod:`environments.base`           -- the ``CooperativeEnv`` interface and the
  ``StepResult`` bundle.
* :mod:`environments.pettingzoo_env` -- the unified PettingZoo parallel adapter.
* :mod:`environments.registry`       -- name -> env factory registry.
* :mod:`environments.benchmarks`     -- registered cooperative benchmarks.
* :mod:`environments.wrappers`       -- SuperSuit-based preprocessing.
* :mod:`environments.random_rollout` -- random-agent rollouts (smoke testing).
"""

from environments.base import CooperativeEnv, StepResult
from environments.pettingzoo_env import PettingZooParallelEnv
from environments.random_rollout import EpisodeStats, run_random_episodes
from environments.registry import ENV_REGISTRY, make_env

__all__ = [
    "CooperativeEnv",
    "StepResult",
    "PettingZooParallelEnv",
    "ENV_REGISTRY",
    "make_env",
    "run_random_episodes",
    "EpisodeStats",
]
