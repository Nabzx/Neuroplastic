"""Optional environment preprocessing, built on SuperSuit.

Preprocessors are ordered callables ``parallel_env -> parallel_env`` applied by
:class:`~environments.pettingzoo_env.PettingZooParallelEnv` when it constructs an
environment. They are selected *by name* from ``PREPROCESSOR_REGISTRY`` (so a
config can list them) and each is a thin, lazily-imported SuperSuit wrapper.

Why these matter here: benchmarks with *heterogeneous* agents (e.g.
``simple_speaker_listener``, where the speaker and listener have different
observation/action shapes) cannot be handled by a single shared policy or a
uniform message space. SuperSuit's ``pad_observations`` / ``pad_action_space``
homogenise them, which is exactly what the graph-based communication layer needs.

SuperSuit is an optional dependency (part of the ``[rl]`` extra); importing this
module never requires it -- only *calling* a preprocessor does.
"""

from __future__ import annotations

from typing import Any

from core.registry import Registry

#: name -> preprocessor callable. Config-driven and extensible.
PREPROCESSOR_REGISTRY: Registry = Registry("env_preprocessor")


def _supersuit() -> Any:
    try:
        import supersuit as ss  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "This environment preprocessor requires SuperSuit. Install the RL "
            "extras: `pip install -e .[rl]` (or `pip install supersuit`)."
        ) from exc
    return ss


@PREPROCESSOR_REGISTRY.register("pad_observations")
def pad_observations(env: Any) -> Any:
    """Pad all agents' observations to a common shape (heterogeneous agents)."""
    return _supersuit().pad_observations_v0(env)


@PREPROCESSOR_REGISTRY.register("pad_action_space")
def pad_action_space(env: Any) -> Any:
    """Pad all agents' action spaces to a common shape (heterogeneous agents)."""
    return _supersuit().pad_action_space_v0(env)


@PREPROCESSOR_REGISTRY.register("flatten")
def flatten(env: Any) -> Any:
    """Flatten structured observations to 1-D vectors."""
    return _supersuit().flatten_v0(env)


def resolve_preprocessors(names) -> list:
    """Resolve preprocessor *names* to their callables, preserving order."""
    return [PREPROCESSOR_REGISTRY.get(name) for name in names]


__all__ = [
    "PREPROCESSOR_REGISTRY",
    "pad_observations",
    "pad_action_space",
    "flatten",
    "resolve_preprocessors",
]
