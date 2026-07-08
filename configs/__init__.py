"""Configuration system: typed schema + YAML loader.

Typical usage::

    from configs import load_config
    cfg = load_config("configs/default.yaml", overrides=["training.lr=1e-4"])
    print(cfg.communication.topology)   # -> "adaptive"
"""

from configs.loader import apply_overrides, deep_merge, load_config, load_yaml
from configs.schema import (
    AgentConfig,
    CommunicationConfig,
    EnvConfig,
    EvaluationConfig,
    ExperimentConfig,
    PlasticityConfig,
    TrainingConfig,
)

__all__ = [
    "load_config",
    "load_yaml",
    "deep_merge",
    "apply_overrides",
    "ExperimentConfig",
    "EnvConfig",
    "AgentConfig",
    "CommunicationConfig",
    "PlasticityConfig",
    "TrainingConfig",
    "EvaluationConfig",
]
