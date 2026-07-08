"""End-to-end message routing with a functional topology/protocol/plasticity.

Exercises the fully-connected topology + mean-pool protocol + reference Hebbian
rule on synthetic messages, before any policy network exists.
"""

import numpy as np

from communication.channel import CommunicationChannel
from communication.message import Message
from configs import load_config
from plasticity.base import make_plasticity

AGENTS = ["agent_0", "agent_1", "agent_2"]


def _channel():
    cfg = load_config(
        "configs/default.yaml",
        overrides=[
            "communication.topology=fully_connected",
            "communication.protocol=mean",
            "plasticity.rule=hebbian",
        ],
    )
    rule = make_plasticity(cfg.plasticity)
    return CommunicationChannel(AGENTS, cfg.communication, rule), cfg


def _messages(step=0):
    return {
        a: Message(sender=a, content=np.full(4, float(i + 1)), step=step)
        for i, a in enumerate(AGENTS)
    }


def test_mean_pool_produces_contexts():
    channel, _ = _channel()
    contexts = channel.step(_messages(step=1), step=1)
    assert set(contexts) == set(AGENTS)
    for a in AGENTS:
        ctx = contexts[a]
        assert ctx is not None
        assert np.asarray(ctx).shape == (4,)


def test_hebbian_updates_edge_weights():
    channel, _ = _channel()
    channel.step(_messages(step=1), step=1)
    # After a Hebbian update + homeostatic scaling, each receiver's two incoming
    # weights should be rescaled below their initial value of 1.0.
    w = channel.graph.weight("agent_1", "agent_0")
    assert w != 1.0
    # homeostasis: incoming weights per receiver sum to <= 1.0
    incoming = sum(channel.graph.weight(s, "agent_0") for s in channel.graph.neighbours("agent_0"))
    assert incoming <= 1.0 + 1e-9


def test_disabled_communication_returns_none_contexts():
    cfg = load_config("configs/default.yaml", overrides=["communication.enabled=false"])
    channel = CommunicationChannel(AGENTS, cfg.communication, None)
    contexts = channel.step(_messages(), step=0)
    assert all(v is None for v in contexts.values())
