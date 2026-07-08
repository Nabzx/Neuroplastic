"""Unified PettingZoo ``ParallelEnv`` adapter.

:class:`PettingZooParallelEnv` wraps *any* PettingZoo parallel environment (raw,
or preprocessed with SuperSuit) behind the project's
:class:`~environments.base.CooperativeEnv` interface. It:

* exposes observations / actions / rewards / done flags / agent ids cleanly
  (via :class:`~environments.base.StepResult`),
* makes episodes reproducible from a single ``seed`` (env RNG *and* the random
  action stream), and
* accepts an ordered list of ``preprocessors`` (e.g. SuperSuit wrappers) so
  heterogeneous benchmarks can be homogenised without touching this class.

PettingZoo itself is only needed by the factories that build concrete envs; this
adapter is agnostic to which parallel env it is handed.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from core.types import Action, AgentID, Observation
from environments.base import CooperativeEnv, StepResult

#: A preprocessor takes a PettingZoo parallel env and returns a wrapped one.
Preprocessor = Callable[[Any], Any]


class PettingZooParallelEnv(CooperativeEnv):
    """Adapt a (possibly preprocessed) PettingZoo ``ParallelEnv``.

    Parameters
    ----------
    env_factory:
        Zero-arg callable returning a PettingZoo ``ParallelEnv``. Deferred so the
        heavy import happens only when an env is actually constructed.
    preprocessors:
        Optional ordered callables applied to the raw env (e.g. SuperSuit
        ``pad_observations`` / ``pad_action_space``). Applied left-to-right.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        preprocessors: Sequence[Preprocessor] | None = None,
    ) -> None:
        env = env_factory()
        for preprocess in preprocessors or ():
            env = preprocess(env)
        self._pz = env
        self._last_infos: dict[AgentID, dict[str, Any]] = {}

    # -- agent sets --------------------------------------------------------
    @property
    def possible_agents(self) -> list[AgentID]:
        return list(self._pz.possible_agents)

    @property
    def agents(self) -> list[AgentID]:
        return list(self._pz.agents)

    # -- core loop ---------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict[str, Any]]]:
        observations, infos = self._pz.reset(seed=seed, options=options)
        if seed is not None:
            # Make the random action stream reproducible alongside env dynamics.
            self.seed_action_spaces(seed)
        self._last_infos = infos
        return observations, infos

    def step(self, actions: dict[AgentID, Action]) -> StepResult:
        observations, rewards, terminations, truncations, infos = self._pz.step(actions)
        self._last_infos = infos
        return StepResult(observations, rewards, terminations, truncations, infos)

    # -- spaces ------------------------------------------------------------
    def observation_space(self, agent: AgentID):
        return self._pz.observation_space(agent)

    def action_space(self, agent: AgentID):
        return self._pz.action_space(agent)

    # -- misc --------------------------------------------------------------
    @property
    def unwrapped(self) -> Any:
        """The underlying PettingZoo env (after any preprocessing)."""
        return self._pz

    def render(self) -> Any:
        return self._pz.render()

    def close(self) -> None:
        self._pz.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        name = type(self._pz).__name__
        return f"PettingZooParallelEnv({name}, num_agents={self.num_agents})"


__all__ = ["PettingZooParallelEnv", "Preprocessor"]
