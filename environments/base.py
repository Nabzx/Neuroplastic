"""The cooperative-environment interface used by the training loop.

We target PettingZoo's *parallel* API (all agents step simultaneously), which
matches the simultaneous-communication setting of this project. :class:`CooperativeEnv`
states the contract; :class:`PettingZooParallelEnv` adapts any PettingZoo
``ParallelEnv`` to it.

PettingZoo is imported lazily inside the adapter so that config validation and
graph/metric analysis do not require the RL dependencies to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import Action, AgentID, Observation


class CooperativeEnv(ABC):
    """Abstract cooperative multi-agent environment (parallel API)."""

    @property
    @abstractmethod
    def agents(self) -> list[AgentID]:
        """Currently active agent ids."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[AgentID, Observation]:
        """Reset and return the initial per-agent observations."""

    @abstractmethod
    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """Advance the environment by one joint action.

        Returns ``(observations, rewards, terminations, truncations, infos)``,
        each keyed by agent id (PettingZoo parallel convention).
        """

    @abstractmethod
    def observation_space(self, agent: AgentID):
        ...

    @abstractmethod
    def action_space(self, agent: AgentID):
        ...

    def close(self) -> None:
        """Release any resources. No-op by default."""


class PettingZooParallelEnv(CooperativeEnv):
    """Adapt a PettingZoo ``ParallelEnv`` to :class:`CooperativeEnv`.

    Parameters
    ----------
    env_factory:
        A zero-arg callable returning a PettingZoo ``ParallelEnv``. Deferred so
        the (heavy) import happens only when an env is actually constructed.
    """

    def __init__(self, env_factory) -> None:
        self._pz = env_factory()

    @property
    def agents(self) -> list[AgentID]:
        return list(self._pz.agents)

    def reset(self, seed: int | None = None) -> dict[AgentID, Observation]:
        obs, _infos = self._pz.reset(seed=seed)
        return obs

    def step(self, actions):
        return self._pz.step(actions)

    def observation_space(self, agent: AgentID):
        return self._pz.observation_space(agent)

    def action_space(self, agent: AgentID):
        return self._pz.action_space(agent)

    def close(self) -> None:
        self._pz.close()


__all__ = ["CooperativeEnv", "PettingZooParallelEnv"]
