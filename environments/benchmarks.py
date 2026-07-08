"""Registered cooperative benchmarks.

Each factory returns a :class:`~environments.base.CooperativeEnv`. The initial
suite targets the MPE (Multi-agent Particle Environment) cooperative tasks from
PettingZoo, which are standard for emergent-communication and coordination
research:

* ``simple_spread``    -- N agents cover N landmarks (pure coordination).
* ``simple_reference`` -- agents must communicate goals to cover landmarks.
* ``simple_speaker_listener`` -- asymmetric speaker/listener communication.

PettingZoo is imported *inside* each factory so importing this module (e.g. for
registration discovery) never requires the RL stack. A missing dependency
produces an actionable error only when an env is actually built.
"""

from __future__ import annotations

from typing import Any

from environments.base import PettingZooParallelEnv
from environments.registry import ENV_REGISTRY


def _require_mpe():
    try:
        from pettingzoo.mpe import (  # type: ignore
            simple_reference_v3,
            simple_speaker_listener_v4,
            simple_spread_v3,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "PettingZoo MPE environments are required for this benchmark. "
            "Install the RL extras: `pip install -e .[rl]` "
            "(or `pip install 'pettingzoo[mpe]'`)."
        ) from exc
    return simple_spread_v3, simple_reference_v3, simple_speaker_listener_v4


@ENV_REGISTRY.register("simple_spread")
def _make_simple_spread(config: Any) -> PettingZooParallelEnv:
    simple_spread_v3, _, _ = _require_mpe()
    return PettingZooParallelEnv(
        lambda: simple_spread_v3.parallel_env(
            N=config.num_agents,
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        )
    )


@ENV_REGISTRY.register("simple_reference")
def _make_simple_reference(config: Any) -> PettingZooParallelEnv:
    _, simple_reference_v3, _ = _require_mpe()
    return PettingZooParallelEnv(
        lambda: simple_reference_v3.parallel_env(
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        )
    )


@ENV_REGISTRY.register("simple_speaker_listener")
def _make_simple_speaker_listener(config: Any) -> PettingZooParallelEnv:
    _, _, simple_speaker_listener_v4 = _require_mpe()
    return PettingZooParallelEnv(
        lambda: simple_speaker_listener_v4.parallel_env(
            max_cycles=config.max_cycles,
            continuous_actions=config.continuous_actions,
            **config.kwargs,
        )
    )


__all__ = [
    "_make_simple_spread",
    "_make_simple_reference",
    "_make_simple_speaker_listener",
]
