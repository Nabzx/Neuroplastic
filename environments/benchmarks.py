"""Registered cooperative benchmarks.

Each factory returns a :class:`~environments.pettingzoo_env.PettingZooParallelEnv`.
The initial suite targets the MPE (Multi-agent Particle Environment) cooperative
tasks, which are standard for emergent-communication and coordination research:

* ``simple_spread``           -- N agents cover N landmarks (pure coordination;
  homogeneous agents).
* ``simple_reference``        -- paired agents must communicate goal landmarks
  (homogeneous agents, communication required).
* ``simple_speaker_listener`` -- asymmetric speaker/listener communication
  (*heterogeneous* agents -> homogenised with SuperSuit padding).

Adding a benchmark:
    1. write a factory ``(EnvConfig) -> PettingZooParallelEnv``,
    2. decorate it with ``@ENV_REGISTRY.register("<name>")``,
    3. reference ``<name>`` from ``config.env.name``.
No other code changes are needed -- the trainer and evaluator resolve envs by
name.

MPE moved packages across PettingZoo versions: the standalone ``mpe2`` package
is the current home, ``pettingzoo.mpe`` the (deprecated) legacy one.
:func:`_load_mpe_env` prefers ``mpe2`` and falls back to ``pettingzoo.mpe`` so
the same code runs regardless. All heavy imports happen inside the factories.
"""

from __future__ import annotations

import importlib
from typing import Any

from environments.pettingzoo_env import PettingZooParallelEnv
from environments.registry import ENV_REGISTRY
from environments.wrappers import resolve_preprocessors


def _load_mpe_env(env_module: str) -> Any:
    """Return an MPE env module (exposing ``parallel_env``) by name.

    Preference order: the current ``mpe2`` package (as a submodule, then as an
    attribute), then the deprecated ``pettingzoo.mpe`` submodule. Raises an
    actionable error if none is installed.
    """
    candidates = (
        lambda: importlib.import_module(f"mpe2.{env_module}"),
        lambda: getattr(importlib.import_module("mpe2"), env_module),
        lambda: importlib.import_module(f"pettingzoo.mpe.{env_module}"),
    )
    for load in candidates:
        try:
            return load()
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f"Could not import MPE env {env_module!r} from 'mpe2' or 'pettingzoo.mpe'. "
        "Install the RL extras: `pip install -e .[rl]` "
        "(or `pip install mpe2` / `pip install 'pettingzoo[mpe]'`)."
    )


def _preprocessors(config: Any, defaults: tuple[str, ...] = ()) -> list:
    """Combine a benchmark's default preprocessors with any from the config."""
    names = list(defaults) + list(getattr(config, "preprocessors", []))
    return resolve_preprocessors(names)


@ENV_REGISTRY.register("simple_spread")
def make_simple_spread(config: Any) -> PettingZooParallelEnv:
    mod = _load_mpe_env("simple_spread_v3")
    return PettingZooParallelEnv(
        lambda: mod.parallel_env(
            N=config.num_agents,
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        ),
        preprocessors=_preprocessors(config),  # homogeneous: none by default
    )


@ENV_REGISTRY.register("simple_reference")
def make_simple_reference(config: Any) -> PettingZooParallelEnv:
    mod = _load_mpe_env("simple_reference_v3")
    return PettingZooParallelEnv(
        lambda: mod.parallel_env(
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        ),
        preprocessors=_preprocessors(config),
    )


@ENV_REGISTRY.register("simple_speaker_listener")
def make_simple_speaker_listener(config: Any) -> PettingZooParallelEnv:
    mod = _load_mpe_env("simple_speaker_listener_v4")
    # Heterogeneous agents -> pad observations and actions to a common shape so a
    # shared communication/message space is well-defined.
    return PettingZooParallelEnv(
        lambda: mod.parallel_env(
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        ),
        preprocessors=_preprocessors(config, defaults=("pad_observations", "pad_action_space")),
    )


__all__ = [
    "make_simple_spread",
    "make_simple_reference",
    "make_simple_speaker_listener",
]
