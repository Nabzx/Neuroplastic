"""Config system: schema validation, YAML inheritance and overrides."""

import pytest

from configs import ExperimentConfig, load_config
from configs.loader import apply_overrides, deep_merge
from configs.schema import _build

CONFIG_DIR = "configs"


def test_default_config_loads():
    cfg = load_config(f"{CONFIG_DIR}/default.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.communication.topology == "adaptive"
    assert cfg.plasticity.rule == "hebbian"
    assert cfg.plasticity.enabled is True


def test_ablation_extends_base():
    cfg = load_config(f"{CONFIG_DIR}/ablations/no_plasticity.yaml")
    # Inherited from default:
    assert cfg.communication.topology == "adaptive"
    # Overridden by the ablation:
    assert cfg.plasticity.enabled is False
    assert cfg.plasticity.rule == "none"


def test_static_topology_ablation():
    cfg = load_config(f"{CONFIG_DIR}/ablations/static_topology.yaml")
    assert cfg.communication.topology == "static"
    assert cfg.plasticity.enabled is True  # plasticity kept


def test_cli_style_overrides():
    cfg = load_config(
        f"{CONFIG_DIR}/default.yaml",
        overrides=["seed=7", "training.lr=1e-4", "communication.topology=k_nearest"],
    )
    assert cfg.seed == 7
    assert cfg.training.lr == pytest.approx(1e-4)
    assert cfg.communication.topology == "k_nearest"


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        _build(ExperimentConfig, {"not_a_real_key": 1})


def test_deep_merge_is_recursive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}
    # original untouched
    assert base["a"]["y"] == 2


def test_apply_overrides_parses_types():
    data = apply_overrides({}, ["a.b=true", "a.c=3", "d=1.5"])
    assert data == {"a": {"b": True, "c": 3}, "d": 1.5}
