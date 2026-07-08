"""Run random agents in a cooperative environment.

A dependency-light sanity check and smoke-test helper: it drives any
:class:`~environments.base.CooperativeEnv` with uniformly random actions and
reports per-episode statistics. Because action sampling is seeded through the
environment (:meth:`CooperativeEnv.reset` seeds the action spaces), rollouts are
fully reproducible from the ``seed`` argument.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from core.types import AgentID
from environments.base import CooperativeEnv


@dataclass
class EpisodeStats:
    """Summary of one random-agent episode."""

    length: int
    returns: dict[AgentID, float] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        """Team return (sum over agents)."""
        return float(sum(self.returns.values()))


def run_random_episodes(
    env: CooperativeEnv,
    num_episodes: int = 3,
    seed: int = 0,
    max_steps: int | None = None,
) -> list[EpisodeStats]:
    """Roll out ``num_episodes`` of random actions and return per-episode stats.

    Parameters
    ----------
    env:
        The environment to drive.
    num_episodes:
        Number of episodes to run.
    seed:
        Base seed. Episode ``e`` uses ``seed + e`` for full reproducibility.
    max_steps:
        Optional hard cap on steps per episode (safety net for envs that might
        not terminate).
    """
    stats: list[EpisodeStats] = []
    for episode in range(num_episodes):
        env.reset(seed=seed + episode)
        returns: dict[AgentID, float] = defaultdict(float)
        length = 0
        while env.agents:
            result = env.step(env.sample_actions())
            for agent, reward in result.rewards.items():
                returns[agent] += float(reward)
            length += 1
            if max_steps is not None and length >= max_steps:
                break
        stats.append(EpisodeStats(length=length, returns=dict(returns)))
    return stats


__all__ = ["EpisodeStats", "run_random_episodes"]
