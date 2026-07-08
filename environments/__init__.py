"""Cooperative multi-agent environments (PettingZoo-based).

* :mod:`environments.base`       -- the ``CooperativeEnv`` interface + a thin
  PettingZoo ``ParallelEnv`` adapter.
* :mod:`environments.registry`   -- name -> env factory registry.
* :mod:`environments.benchmarks` -- registered cooperative benchmarks.
* :mod:`environments.wrappers`   -- optional observation/reward wrappers.
"""

from environments.base import CooperativeEnv
from environments.registry import ENV_REGISTRY, make_env

__all__ = ["CooperativeEnv", "ENV_REGISTRY", "make_env"]
