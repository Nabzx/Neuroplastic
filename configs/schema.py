"""Typed configuration schema.

Every experiment is fully described by an :class:`ExperimentConfig`. The schema
is expressed as nested dataclasses so that:

* configuration is validated (unknown keys raise immediately),
* defaults live in one place,
* editors/type-checkers can autocomplete config access, and
* a config can be round-tripped to/from plain dicts (and therefore YAML/JSON).

NB: this module deliberately does *not* use ``from __future__ import
annotations`` because :func:`_build` introspects real field types at runtime to
recurse into nested dataclasses.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Cooperative environment / benchmark selection."""

    name: str = "simple_spread"          # key into environments.registry
    num_agents: int = 3
    max_cycles: int = 25
    continuous_actions: bool = False
    preprocessors: list[str] = field(default_factory=list)  # extra SuperSuit wrappers by name
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Per-agent policy architecture (shapes only; no weights here)."""

    type: str = "recurrent_policy"       # key into agents' registry
    hidden_dim: int = 128
    obs_embedding_dim: int = 64
    recurrent: bool = True
    share_parameters: bool = True        # homogeneous vs. heterogeneous agents


@dataclass
class CommunicationConfig:
    """Adaptive graph-based communication settings."""

    enabled: bool = True
    message_dim: int = 16
    topology: str = "adaptive"           # fully_connected | k_nearest | ring | adaptive | static
    protocol: str = "attention"          # mean (fixed uniform) | attention (adaptive) | gnn
    attention_dim: int = 32              # query/key dim for adaptive (attention) weighting
    max_neighbours: int = 4              # bandwidth cap for sparse topologies
    num_rounds: int = 1                  # message-passing rounds per step
    bandwidth: int | None = None         # optional bits/step channel constraint


@dataclass
class PlasticityConfig:
    """Hebbian-inspired plasticity for the communication weights."""

    enabled: bool = True
    rule: str = "hebbian"                # hebbian | oja | none
    learning_rate: float = 1e-3
    decay: float = 1e-4                  # weight decay / forgetting
    modulation: str = "reward_gated"     # none | reward_gated | three_factor
    homeostasis: bool = True             # synaptic scaling to keep weights bounded
    update_every: int = 1                # env steps between plasticity updates


@dataclass
class TrainingConfig:
    """Multi-agent RL training hyper-parameters (loop not yet implemented)."""

    algorithm: str = "ippo"              # independent PPO baseline
    total_steps: int = 1_000_000
    rollout_length: int = 128
    num_envs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2               # PPO clipped-surrogate epsilon
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 256
    update_epochs: int = 4
    log_every: int = 1_000
    checkpoint_every: int = 50_000


@dataclass
class EvaluationConfig:
    """Which emergent-behaviour measures to compute, and how often."""

    episodes: int = 100
    deterministic: bool = True
    graph_metrics: list[str] = field(
        default_factory=lambda: [
            "density",
            "modularity",
            "global_clustering",
            "characteristic_path_length",
            "degree_centralisation",
        ]
    )
    information_metrics: list[str] = field(
        default_factory=lambda: [
            "message_entropy",
            "mutual_information",
            "transfer_entropy",
        ]
    )
    coordination_metrics: list[str] = field(
        default_factory=lambda: [
            "episode_return",
            "success_rate",
            "coordination_index",
        ]
    )
    specialisation_metrics: list[str] = field(
        default_factory=lambda: [
            "role_entropy",
            "role_cluster_count",
        ]
    )
    log_topology_every: int = 1_000


@dataclass
class ExperimentConfig:
    """Top-level, fully-specified experiment description."""

    name: str = "nci_default"
    seed: int = 0
    device: str = "auto"                 # auto | cpu | cuda
    output_dir: str = "runs"
    notes: str = ""

    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # -- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Build a validated config from a (possibly nested) plain dict."""
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct dataclass ``cls`` from ``data``.

    Unknown keys raise ``KeyError`` so typos in a YAML file are caught early
    rather than silently ignored.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping to build {cls.__name__}, got {type(data).__name__}")

    valid = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(valid)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in valid.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


__all__ = [
    "EnvConfig",
    "AgentConfig",
    "CommunicationConfig",
    "PlasticityConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
