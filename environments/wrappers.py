"""Optional environment wrappers.

Placeholders for common preprocessing that keeps the training loop clean:

* observation flattening / normalisation,
* reward shaping or centralised-reward broadcasting,
* recording of ground-truth interaction structure (where the benchmark exposes
  it) for validating discovered topologies.

These are intentionally unimplemented until the training milestone; the base
:class:`~environments.base.CooperativeEnv` API is sufficient to start.
"""

from __future__ import annotations

from environments.base import CooperativeEnv


class ObservationNormaliser(CooperativeEnv):  # pragma: no cover - deferred
    """Running-mean/variance observation normalisation wrapper (placeholder)."""

    def __init__(self, env: CooperativeEnv) -> None:
        self.env = env
        raise NotImplementedError(
            "ObservationNormaliser is a placeholder; see docs/experiment_plan.md."
        )

    @property
    def agents(self):
        return self.env.agents

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)


__all__ = ["ObservationNormaliser"]
