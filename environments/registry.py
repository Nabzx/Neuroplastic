"""Environment registry: build a :class:`CooperativeEnv` from a config by name."""

from __future__ import annotations

from typing import Any

from core.registry import Registry
from environments.base import CooperativeEnv

#: Maps ``config.env.name`` -> a factory ``(EnvConfig) -> CooperativeEnv``.
ENV_REGISTRY: Registry = Registry("environment")


def make_env(config: Any) -> CooperativeEnv:
    """Construct the environment named by ``config.name``."""
    # Import for registration side-effects without an import cycle at load time.
    import environments.benchmarks  # noqa: F401

    factory = ENV_REGISTRY.get(config.name)
    return factory(config)


__all__ = ["ENV_REGISTRY", "make_env"]
