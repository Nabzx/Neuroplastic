"""Run the benchmark grid: environments x communication methods x seeds.

Trains every combination with an identical budget and collects, per run, the
flattened evaluation metrics, the training history (for curves) and the final
communication-graph matrix (for figures). Orchestration only -- the metric
computation lives in ``evaluation`` / ``analysis``.

This adds no new learning algorithm; it re-uses the existing Trainer to evaluate
the five communication modes rigorously.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Sequence

from core.logging_utils import get_logger
from evaluation.report import flatten_report

logger = get_logger("nci.benchmark")

#: Cooperative PettingZoo benchmarks (heterogeneous ones are padded by SuperSuit).
ENVIRONMENTS = ["simple_spread", "simple_reference", "simple_speaker_listener"]

#: The five communication modes, in comparison order.
METHODS: dict[str, str] = {
    "no_comm": "configs/baselines/no_comm.yaml",
    "fully_connected": "configs/baselines/fully_connected.yaml",
    "sparse": "configs/baselines/sparse.yaml",
    "adaptive": "configs/adaptive.yaml",
    "neuroplastic": "configs/plastic.yaml",
}


def run_single(
    env_name: str,
    config_path: str,
    seed: int,
    overrides: Iterable[str] = (),
    threshold: float = 1e-3,
    spec_episodes: int = 20,
) -> dict[str, Any]:
    """Train one (env, method, seed) and return its metrics + curves + final graph."""
    from analysis.specialisation_analysis import functional_specialisation
    from configs import load_config
    from training.compare import evaluate_trainer
    from training.trainer import Trainer

    all_overrides = list(overrides) + [f"env.name={env_name}", f"seed={seed}"]
    trainer = Trainer(load_config(config_path, overrides=all_overrides))
    trainer.train()

    report = evaluate_trainer(trainer, threshold=threshold)
    try:
        profiles = trainer.behavioural_profiles(episodes=spec_episodes, seed=1000 + seed)
        report["specialisation"] = functional_specialisation(profiles)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("specialisation failed for %s/%s: %s", env_name, config_path, exc)
        report["specialisation"] = {
            "role_diversity": float("nan"),
            "role_cluster_count": float("nan"),
            "role_entropy": float("nan"),
        }

    final_matrix = trainer.graph_history[-1][1] if trainer.graph_history else None
    return {
        "metrics": flatten_report(report),
        "history": trainer.history,
        "final_matrix": final_matrix,
        "agent_ids": list(trainer.agent_ids),
    }


def run_benchmark(
    environments: Sequence[str] | None = None,
    methods: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    overrides: Iterable[str] = (),
    threshold: float = 1e-3,
    spec_episodes: int = 20,
) -> dict[str, Any]:
    """Run the full grid; return ``{results, curves, finals}``.

    * ``results[env][method]`` -- list of per-seed flat metric dicts.
    * ``curves[env][method]``  -- list of per-seed training histories.
    * ``finals[env][method]``  -- ``(final_matrix, agent_ids)`` from the first seed.

    Runs that fail (e.g. an incompatible environment) are logged and skipped, so
    the rest of the grid still completes.
    """
    environments = list(environments or ENVIRONMENTS)
    methods = dict(methods or METHODS)
    overrides = list(overrides)

    results: dict[str, dict[str, list]] = {}
    curves: dict[str, dict[str, list]] = {}
    finals: dict[str, dict[str, tuple]] = {}

    total = len(environments) * len(methods) * len(seeds)
    done = 0
    start = time.time()
    logger.info("Benchmark: %d envs x %d methods x %d seeds = %d runs", len(environments), len(methods), len(seeds), total)

    for env in environments:
        results[env], curves[env], finals[env] = {}, {}, {}
        for method, config_path in methods.items():
            results[env][method], curves[env][method] = [], []
            for seed in seeds:
                done += 1
                tag = f"{env}/{method}/seed{seed} [{done}/{total}]"
                try:
                    out = run_single(env, config_path, seed, overrides, threshold, spec_episodes)
                except Exception as exc:  # noqa: BLE001 - keep the grid going
                    logger.warning("SKIP %s: %s", tag, exc)
                    continue
                results[env][method].append(out["metrics"])
                curves[env][method].append(out["history"])
                if method not in finals[env] and out["final_matrix"] is not None:
                    finals[env][method] = (out["final_matrix"], out["agent_ids"])
                fr = out["metrics"].get("coordination.final_reward", float("nan"))
                logger.info("done %s | final_reward=%.2f", tag, fr)
            if not results[env][method]:
                logger.warning("no successful runs for %s/%s", env, method)

    logger.info("Benchmark complete in %.1f min", (time.time() - start) / 60.0)
    return {"results": results, "curves": curves, "finals": finals}


__all__ = ["ENVIRONMENTS", "METHODS", "run_single", "run_benchmark"]
