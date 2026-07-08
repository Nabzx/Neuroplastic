"""The cooperative-environment interface used by the training loop.

We target PettingZoo's *parallel* API (all agents step simultaneously), which
matches the simultaneous-communication setting of this project.

* :class:`CooperativeEnv` states the contract every environment must satisfy and
  provides shared conveniences (spaces dicts, reproducible action-space seeding,
  random action sampling).
* :class:`StepResult` bundles one step's ``(observations, rewards, terminations,
  truncations, infos)`` behind named fields while still unpacking like the
  PettingZoo 5-tuple, so downstream code can use whichever it prefers.

The concrete PettingZoo adapter lives in :mod:`environments.pettingzoo_env`; the
RL dependencies are imported lazily there so config/graph tooling never needs
them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

from core.types import Action, AgentID, Observation


@dataclass
class StepResult:
    """One environment step, keyed by agent id (PettingZoo parallel convention).

    Iterating a :class:`StepResult` yields the classic 5-tuple, so both styles
    work::

        result = env.step(actions)
        obs, rewards, terms, truncs, infos = env.step(actions)   # unpacking
        if result.all_done: ...                                   # named access
    """

    observations: dict[AgentID, Observation]
    rewards: dict[AgentID, float]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    infos: dict[AgentID, dict[str, Any]]

    def __iter__(self) -> Iterator[dict]:
        yield self.observations
        yield self.rewards
        yield self.terminations
        yield self.truncations
        yield self.infos

    @property
    def agents(self) -> list[AgentID]:
        """Agents present in this step's observations."""
        return list(self.observations)

    @property
    def dones(self) -> dict[AgentID, bool]:
        """Per-agent done = termination OR truncation."""
        keys = set(self.terminations) | set(self.truncations)
        return {
            a: bool(self.terminations.get(a, False) or self.truncations.get(a, False))
            for a in keys
        }

    @property
    def all_done(self) -> bool:
        """True when every agent in this step is done (episode boundary)."""
        dones = self.dones
        return bool(dones) and all(dones.values())


class CooperativeEnv(ABC):
    """Abstract cooperative multi-agent environment (parallel API)."""

    # -- agent sets --------------------------------------------------------
    @property
    @abstractmethod
    def possible_agents(self) -> list[AgentID]:
        """All agent ids that can ever appear (stable across the episode)."""

    @property
    @abstractmethod
    def agents(self) -> list[AgentID]:
        """Currently active agent ids (may shrink as agents finish)."""

    @property
    def agent_ids(self) -> list[AgentID]:
        """Stable, construction-time agent ids (alias of :pyattr:`possible_agents`).

        Prefer this over :pyattr:`agents` when you need the full roster before a
        reset (e.g. to size networks), since ``agents`` is only populated after
        ``reset``.
        """
        return self.possible_agents

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    # -- core loop ---------------------------------------------------------
    @abstractmethod
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict[str, Any]]]:
        """Reset and return ``(observations, infos)``.

        Passing ``seed`` must make the episode reproducible, including the random
        action stream drawn via :meth:`sample_actions`.
        """

    @abstractmethod
    def step(self, actions: dict[AgentID, Action]) -> StepResult:
        """Advance the environment by one joint action."""

    @abstractmethod
    def observation_space(self, agent: AgentID):
        ...

    @abstractmethod
    def action_space(self, agent: AgentID):
        ...

    # -- shared conveniences ----------------------------------------------
    @property
    def observation_spaces(self) -> dict[AgentID, Any]:
        return {a: self.observation_space(a) for a in self.possible_agents}

    @property
    def action_spaces(self) -> dict[AgentID, Any]:
        return {a: self.action_space(a) for a in self.possible_agents}

    def seed_action_spaces(self, seed: int) -> None:
        """Seed each agent's action space with a distinct, derived seed.

        This makes :meth:`sample_actions` reproducible. Agent ``i`` gets
        ``seed + i + 1`` so its stream differs from both the env seed and the
        other agents.
        """
        for i, agent in enumerate(self.possible_agents):
            self.action_space(agent).seed(seed + i + 1)

    def sample_actions(self, only_active: bool = True) -> dict[AgentID, Action]:
        """Sample a random action per agent from its (seeded) action space."""
        agents = self.agents if only_active else self.possible_agents
        return {a: self.action_space(a).sample() for a in agents}

    def render(self) -> Any:
        """Render the environment (no-op by default)."""
        return None

    def close(self) -> None:
        """Release any resources. No-op by default."""


__all__ = ["CooperativeEnv", "StepResult"]
