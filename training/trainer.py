"""The Trainer: composes every subsystem into a runnable experiment.

Responsibilities:

* seed everything reproducibly from the config,
* build the environment, the shared-parameter PPO learner and the fixed
  communication mask, and run the on-policy training loop, logging reward and
  episode length, and
* expose :meth:`describe` for a dependency-light "dry run" that verifies the
  whole config resolves to real components *without* importing torch/PettingZoo.

This implements the first **learning baseline**: parameter-shared PPO over a
*fixed* communication graph (no plasticity). The three communication settings
(none / fully-connected / sparse) differ only in the mask; see
:mod:`training.utils` and ``configs/baselines/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.schema import ExperimentConfig
from core.logging_utils import get_logger
from core.seeding import set_global_seed


class Trainer:
    """Owns the full experiment object graph and runs the training loop."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = get_logger("nci.trainer")
        set_global_seed(config.seed)

        self.env = None
        self.learner = None
        self.buffer = None
        self.mask = None
        self.comm_mode = "none"
        self.agent_ids: list[str] = []
        self.device = "cpu"
        self.history: list[dict[str, Any]] = []
        self.graph_history: list[tuple[int, Any]] = []
        self._built = False

    # -- construction ------------------------------------------------------
    def build(self) -> "Trainer":
        """Instantiate the environment, learner and rollout buffer (needs torch)."""
        import numpy as np

        from environments.registry import make_env
        from training.learner import SharedPPOLearner
        from training.rollout import RolloutBuffer
        from training.utils import build_comm, resolve_device

        cfg = self.config
        self.device = resolve_device(cfg.device)
        self.logger.info("Building experiment %r on device=%s", cfg.name, self.device)

        self.env = make_env(cfg.env)
        self.agent_ids = self.env.possible_agents
        _assert_homogeneous(self.env, self.agent_ids)

        obs_space = self.env.observation_space(self.agent_ids[0])
        action_space = self.env.action_space(self.agent_ids[0])
        obs_dim = int(np.prod(obs_space.shape))
        if not hasattr(action_space, "n"):
            raise NotImplementedError(
                "The baseline supports discrete action spaces only. Set "
                "env.continuous_actions=false (or add a continuous head later)."
            )
        action_dim = int(action_space.n)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.comm_mode, self.mask = build_comm(self.agent_ids, cfg, self.device)
        self.learner = SharedPPOLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=len(self.agent_ids),
            comm_mode=self.comm_mode,
            adjacency=self.mask,
            config=cfg,
            device=self.device,
        )
        self.buffer = RolloutBuffer(
            cfg.training.rollout_length, len(self.agent_ids), obs_dim, self.device
        )
        self._built = True
        return self

    # -- training ----------------------------------------------------------
    def train(self) -> list[dict[str, Any]]:
        """Run the PPO training loop; return per-iteration metric history."""
        import torch

        from training.logger import MetricLogger
        from training.utils import stack_by_agent

        if not self._built:
            self.build()

        cfg = self.config
        t = cfg.training
        n = len(self.agent_ids)
        comm = self.comm_mode if self.comm_mode == "none" else f"{self.comm_mode}/{cfg.communication.topology}"
        self.logger.info(
            "Training %r: %d agents, communication=%s, %d env steps",
            cfg.name, n, comm, t.total_steps,
        )

        metric_logger = MetricLogger(cfg.output_dir, cfg.name, self.logger)
        num_iterations = max(1, t.total_steps // t.rollout_length)
        console_every = max(1, t.log_every // t.rollout_length)

        obs_dict, _ = self.env.reset(seed=cfg.seed)
        obs = stack_by_agent(obs_dict, self.agent_ids, self.device)
        episode_return = torch.zeros(n, device=self.device)
        episode_length = 0
        env_steps = 0
        last_return_mean = float("nan")
        last_length_mean = float("nan")

        for iteration in range(1, num_iterations + 1):
            ep_returns: list[float] = []
            ep_lengths: list[int] = []

            for _ in range(t.rollout_length):
                action, log_prob, value = self.learner.act(obs)
                actions = {aid: int(action[i].item()) for i, aid in enumerate(self.agent_ids)}
                result = self.env.step(actions)
                reward = stack_by_agent(result.rewards, self.agent_ids, self.device)
                done = float(result.all_done or not self.env.agents)

                self.buffer.add(obs, action, log_prob, value, reward, done)
                episode_return += reward
                episode_length += 1
                env_steps += 1

                if done:
                    ep_returns.append(episode_return.mean().item())
                    ep_lengths.append(episode_length)
                    episode_return = torch.zeros(n, device=self.device)
                    episode_length = 0
                    obs_dict, _ = self.env.reset()
                    obs = stack_by_agent(obs_dict, self.agent_ids, self.device)
                else:
                    obs = stack_by_agent(result.observations, self.agent_ids, self.device)

            # Bootstrap + advantage estimation, then a PPO update.
            last_value = self.learner.value(obs)
            self.buffer.compute_gae(last_value, t.gamma, t.gae_lambda)
            metrics = self.learner.update(self.buffer)
            # Hebbian plasticity update (plastic mode only), then snapshot the
            # (possibly just-updated) communication graph for stats/logging.
            plast_stats = self.learner.plasticity_update(self.buffer, iteration)
            snapshot = self.learner.graph_snapshot(self.buffer.obs)
            comm_stats = self.learner.communication_statistics(snapshot)
            self.buffer.reset()

            if snapshot is not None:
                self.graph_history.append((env_steps, snapshot.detach().cpu().numpy()))
            if ep_returns:
                last_return_mean = float(sum(ep_returns) / len(ep_returns))
                last_length_mean = float(sum(ep_lengths) / len(ep_lengths))

            row = {
                "iteration": iteration,
                "env_steps": env_steps,
                "episode_return_mean": last_return_mean,
                "episode_length_mean": last_length_mean,
                "num_episodes": len(ep_returns),
                **{k: round(v, 6) for k, v in metrics.items()},
                **{k: round(v, 6) for k, v in comm_stats.items()},
                **{k: round(v, 6) for k, v in plast_stats.items()},
            }
            self.history.append(row)
            metric_logger.log(row, verbose=(iteration % console_every == 0 or iteration == num_iterations))

        metric_logger.close()
        self._save_graph_history()
        self.logger.info("Training complete: %d iterations, %d env steps.", num_iterations, env_steps)
        return self.history

    def _save_graph_history(self) -> None:
        """Persist the recorded edge-weight snapshots to ``edge_weights.npz``."""
        if not self.graph_history:
            return
        import numpy as np

        out = Path(self.config.output_dir) / self.config.name / "edge_weights.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        steps = np.array([step for step, _ in self.graph_history])
        weights = np.stack([w for _, w in self.graph_history])  # [num_snapshots, N, N]
        np.savez(out, steps=steps, weights=weights, agents=np.array(self.agent_ids))
        self.logger.info("Saved edge-weight evolution: %s (%d snapshots)", out, len(steps))

    # -- evaluation --------------------------------------------------------
    def evaluate(self, episodes: int | None = None, seed: int | None = None) -> dict[str, float]:
        """Roll out the greedy policy and return mean return / episode length."""
        import torch

        from training.utils import stack_by_agent

        if not self._built:
            self.build()
        cfg = self.config
        episodes = episodes or cfg.evaluation.episodes
        n = len(self.agent_ids)

        returns: list[float] = []
        lengths: list[int] = []
        for episode in range(episodes):
            obs_dict, _ = self.env.reset(seed=(seed + episode) if seed is not None else None)
            obs = stack_by_agent(obs_dict, self.agent_ids, self.device)
            ep_return = torch.zeros(n, device=self.device)
            length = 0
            while self.env.agents:
                action, _, _ = self.learner.act(obs, deterministic=True)
                actions = {aid: int(action[i].item()) for i, aid in enumerate(self.agent_ids)}
                result = self.env.step(actions)
                ep_return += stack_by_agent(result.rewards, self.agent_ids, self.device)
                length += 1
                if not self.env.agents:
                    break
                obs = stack_by_agent(result.observations, self.agent_ids, self.device)
            returns.append(ep_return.mean().item())
            lengths.append(length)

        return {
            "eval_return_mean": float(sum(returns) / len(returns)),
            "eval_length_mean": float(sum(lengths) / len(lengths)),
            "eval_episodes": float(episodes),
        }

    # -- behavioural profiles (functional specialisation) ------------------
    def behavioural_profiles(self, episodes: int = 20, seed: int = 0) -> Any:
        """Greedy-rollout action distribution per agent, shape ``[N, action_dim]``.

        Each row is an agent's normalised histogram over discrete actions -- the
        behavioural descriptor used to measure functional specialisation.
        """
        import numpy as np

        from training.utils import stack_by_agent

        if not self._built:
            self.build()
        n = len(self.agent_ids)
        counts = np.zeros((n, self.action_dim), dtype=float)
        for episode in range(episodes):
            obs_dict, _ = self.env.reset(seed=seed + episode)
            obs = stack_by_agent(obs_dict, self.agent_ids, self.device)
            while self.env.agents:
                action, _, _ = self.learner.act(obs, deterministic=True)
                for i, aid in enumerate(self.agent_ids):
                    if aid in self.env.agents:
                        counts[i, int(action[i].item())] += 1.0
                result = self.env.step(
                    {aid: int(action[i].item()) for i, aid in enumerate(self.agent_ids)}
                )
                if not self.env.agents:
                    break
                obs = stack_by_agent(result.observations, self.agent_ids, self.device)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums

    # -- communication graph export ---------------------------------------
    def export_communication_graph(self, obs: Any | None = None):
        """Export the current communication graph as an ``InteractionGraph``.

        Uses the most recent rollout's observations by default (so call after
        :meth:`train`). Returns ``None`` when communication is disabled; otherwise
        the returned graph carries the (adaptive or fixed) edge weights and can be
        handed to NetworkX via ``result.to_networkx()``.
        """
        from communication.graph import InteractionGraph

        if not self._built:
            self.build()
        obs_batch = obs if obs is not None else self.buffer.obs
        snapshot = self.learner.graph_snapshot(obs_batch)
        if snapshot is None:
            return None
        return InteractionGraph.from_weight_matrix(
            self.agent_ids, snapshot.detach().cpu().numpy()
        )

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


def _resolved(registry, key: str) -> dict[str, str]:
    """Return a small ``{key, class}`` record, verifying the key is registered."""
    obj = registry.get(key)
    return {"name": key, "impl": getattr(obj, "__name__", type(obj).__name__)}


def _assert_homogeneous(env: Any, agent_ids: list[str]) -> None:
    """Ensure all agents share observation/action shapes (parameter sharing)."""
    obs_shapes = {tuple(env.observation_space(a).shape) for a in agent_ids}
    action_sizes = {getattr(env.action_space(a), "n", None) for a in agent_ids}
    if len(obs_shapes) != 1 or len(action_sizes) != 1:
        raise ValueError(
            "The shared-parameter baseline requires homogeneous agents "
            f"(got obs shapes {obs_shapes}, action sizes {action_sizes}). "
            "Add SuperSuit padding preprocessors (env.preprocessors: "
            "[pad_observations, pad_action_space]) to homogenise the env."
        )


__all__ = ["Trainer"]
