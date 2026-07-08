"""Registry mechanics and built-in registrations."""

import pytest

from core.registry import Registry


def test_register_and_get():
    reg: Registry = Registry("thing")
    reg.register("a", 1)

    @reg.register("b")
    class B:
        pass

    assert reg.get("a") == 1
    assert reg.get("b") is B
    assert "a" in reg and "b" in reg
    assert set(reg.keys()) == {"a", "b"}
    assert len(reg) == 2


def test_duplicate_registration_raises():
    reg: Registry = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(KeyError):
        reg.register("a", 2)


def test_unknown_key_raises_with_options():
    reg: Registry = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(KeyError, match="Registered options"):
        reg.get("missing")


def test_builtin_topologies_registered():
    from communication.topology import TOPOLOGY_REGISTRY

    for name in ("fully_connected", "static", "k_nearest", "adaptive"):
        assert name in TOPOLOGY_REGISTRY


def test_builtin_protocols_registered():
    from communication.protocol import PROTOCOL_REGISTRY

    for name in ("mean", "attention", "gnn"):
        assert name in PROTOCOL_REGISTRY


def test_builtin_plasticity_rules_registered():
    import plasticity.hebbian  # noqa: F401  (registration side-effect)
    from plasticity.base import PLASTICITY_REGISTRY

    for name in ("none", "hebbian", "oja"):
        assert name in PLASTICITY_REGISTRY
