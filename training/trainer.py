"""The Trainer: composes every subsystem into a runnable experiment.

Responsibilities:

* seed everything reproducibly from the config,
* construct the environment, agents, communication channel (topology + protocol
  + plasticity) and learning algorithm from their registries, and
* expose :meth:`describe` for a dependency-light "dry run" that verifies the
  whole config resolves to real components *without* importing torch/PettingZoo.

The optimisation loop (:meth:`train`) is deliberately a placeholder; this file
is about getting the wiring and interfaces right first.
"""

from __future__ import annotations

from typing import Any

from configs.schema import ExperimentConfig
from core.logging_utils import get_logger
from core.seeding import set_global_seed


class Trainer:
    """Owns the full experiment object graph."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = get_logger("nci.trainer")
        set_global_seed(config.seed)

        self.env = None
        self.agents: dict[str, Any] = {}
        self.channel = None
        self.algorithm = None
        self._built = False

    # -- construction ------------------------------------------------------
    def build(self) -> "Trainer":
        """Instantiate every subsystem. Requires the RL dependencies."""
        from agents.base import make_agent
        from communication.channel import CommunicationChannel
        from environments.registry import make_env
        from plasticity.base import make_plasticity
        from training.algorithms import make_algorithm

        cfg = self.config
        self.logger.info("Building experiment %r on device=%s", cfg.name, cfg.device)

        self.env = make_env(cfg.env)
        agent_ids = self.env.agents

        plasticity_rule = make_plasticity(cfg.plasticity)
        self.channel = CommunicationChannel(agent_ids, cfg.communication, plasticity_rule)

        self.agents = {
            aid: make_agent(
                agent_id=aid,
                obs_dim=_space_size(self.env.observation_space(aid)),
                action_dim=_space_size(self.env.action_space(aid)),
                config=cfg.agent,
            )
            for aid in agent_ids
        }
        self.algorithm = make_algorithm(cfg.training)
        self._built = True
        return self

    # -- dry run -----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Resolve every configured component by name and report the wiring.

        Does not construct the environment or any torch modules, so it runs with
        only the light dependencies installed -- useful for validating a config.
        """
        from agents.base import AGENT_REGISTRY
        from communication.protocol import PROTOCOL_REGISTRY
        from communication.topology import TOPOLOGY_REGISTRY
        from environments.registry import ENV_REGISTRY
        from plasticity.base import PLASTICITY_REGISTRY
        from training.algorithms import ALGORITHM_REGISTRY

        import agents.recurrent_policy  # noqa: F401  (registration)
        import environments.benchmarks  # noqa: F401
        import plasticity.hebbian  # noqa: F401

        cfg = self.config
        plasticity_name = "none" if not cfg.plasticity.enabled else cfg.plasticity.rule
        return {
            "experiment": cfg.name,
            "seed": cfg.seed,
            "device": cfg.device,
            "environment": _resolved(ENV_REGISTRY, cfg.env.name),
            "agent": _resolved(AGENT_REGISTRY, cfg.agent.type),
            "communication": {
                "enabled": cfg.communication.enabled,
                "topology": _resolved(TOPOLOGY_REGISTRY, cfg.communication.topology),
                "protocol": _resolved(PROTOCOL_REGISTRY, cfg.communication.protocol),
            },
            "plasticity": {
                "enabled": cfg.plasticity.enabled,
                "rule": _resolved(PLASTICITY_REGISTRY, plasticity_name),
                "modulation": cfg.plasticity.modulation,
                "homeostasis": cfg.plasticity.homeostasis,
            },
            "algorithm": _resolved(ALGORITHM_REGISTRY, cfg.training.algorithm),
        }

    # -- (deferred) optimisation ------------------------------------------
    def train(self) -> None:  # pragma: no cover - deferred
        raise NotImplementedError(
            "Trainer.train is a placeholder. The rollout/update loop is "
            "specified in docs/experiment_plan.md and will be implemented in the "
            "training milestone. Use `describe()` / `--dry-run` to validate "
            "wiring in the meantime."
        )

    def evaluate(self) -> dict[str, Any]:  # pragma: no cover - deferred
        raise NotImplementedError(
            "Trainer.evaluate is a placeholder; see evaluation.evaluator.Evaluator."
        )


def _resolved(registry, key: str) -> dict[str, str]:
    """Return a small ``{key, class}`` record, verifying the key is registered."""
    obj = registry.get(key)
    return {"name": key, "impl": getattr(obj, "__name__", type(obj).__name__)}


def _space_size(space: Any) -> int:
    """Best-effort flat size of a gym/gymnasium space (deferred to build time)."""
    n = getattr(space, "n", None)
    if n is not None:
        return int(n)
    shape = getattr(space, "shape", None)
    if shape:
        size = 1
        for dim in shape:
            size *= int(dim)
        return size
    raise TypeError(f"Cannot infer size of space {space!r}")


__all__ = ["Trainer"]
