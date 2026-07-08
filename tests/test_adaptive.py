"""Adaptive graph-based communication.

Three tiers:
* Torch-free: weighted-graph statistics and NetworkX export.
* Torch: the attention edge-weight module + adaptive policy forward.
* End-to-end (PettingZoo): adaptive training exposes graph stats and exports.
"""

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Torch-free: statistics + NetworkX export
# --------------------------------------------------------------------------- #
UNIFORM_TRIANGLE = np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)  # receiver x sender, 3 fully-connected agents with uniform weights


def test_weight_statistics_uniform():
    from communication.statistics import weight_matrix_statistics

    stats = weight_matrix_statistics(UNIFORM_TRIANGLE)
    assert stats["comm_edge_density"] == pytest.approx(1.0)      # all off-diag active
    assert stats["comm_mean_weight"] == pytest.approx(0.5)
    assert stats["comm_max_weight"] == pytest.approx(0.5)
    assert stats["comm_weight_entropy"] == pytest.approx(1.0)    # 2 equal senders -> 1 bit
    assert stats["comm_effective_degree"] == pytest.approx(2.0)  # 2^1


def test_weight_statistics_concentrated():
    from communication.statistics import weight_matrix_statistics

    # each receiver puts all weight on a single sender -> entropy 0, degree 1
    concentrated = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    stats = weight_matrix_statistics(concentrated)
    assert stats["comm_weight_entropy"] == pytest.approx(0.0)
    assert stats["comm_effective_degree"] == pytest.approx(1.0)
    assert stats["comm_edge_density"] == pytest.approx(3 / 6)    # 3 of 6 possible edges


def test_export_to_networkx():
    import networkx as nx

    from communication.graph import InteractionGraph

    agents = ["agent_0", "agent_1", "agent_2"]
    graph = InteractionGraph.from_weight_matrix(agents, UNIFORM_TRIANGLE)
    dg = graph.to_networkx()

    assert isinstance(dg, nx.DiGraph)
    assert dg.number_of_nodes() == 3
    assert dg.number_of_edges() == 6                              # complete digraph
    # matrix[0, 1] = weight from sender agent_1 to receiver agent_0
    assert dg["agent_1"]["agent_0"]["weight"] == pytest.approx(0.5)
    assert not dg.has_edge("agent_0", "agent_0")                  # no self-loops


def test_export_thresholds_weak_edges():
    from communication.graph import InteractionGraph

    matrix = np.array([[0.0, 0.9, 0.001], [0.5, 0.0, 0.5], [0.2, 0.8, 0.0]])
    graph = InteractionGraph.from_weight_matrix(["a", "b", "c"], matrix, threshold=0.01)
    dg = graph.to_networkx()
    # the 0.001 edge (sender c -> receiver a) is below threshold and dropped
    assert not dg.has_edge("c", "a")
    assert dg.has_edge("b", "a")


# --------------------------------------------------------------------------- #
# Torch: attention edge-weights + adaptive policy
# --------------------------------------------------------------------------- #
def test_adaptive_weights_normalised_and_no_self_loops():
    pytest.importorskip("torch")
    import torch

    from communication.adaptive import AdaptiveCommunication

    module = AdaptiveCommunication(embedding_dim=8, message_dim=4, attention_dim=8)
    embeddings = torch.randn(2, 5, 8)
    messages = torch.randn(2, 5, 4)
    context, weights = module(embeddings, messages)

    assert context.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)
    # rows (over senders) sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    # no self-loops
    diag = weights.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros(2, 5), atol=1e-6)


def test_adaptive_weights_respect_structural_mask():
    pytest.importorskip("torch")
    import torch

    from communication.adaptive import AdaptiveCommunication
    from configs.schema import CommunicationConfig
    from training.utils import build_structural_mask

    agents = ["a0", "a1", "a2", "a3"]
    mask = build_structural_mask(agents, CommunicationConfig(topology="ring", max_neighbours=2), "cpu")
    module = AdaptiveCommunication(embedding_dim=8, message_dim=4, attention_dim=8)
    embeddings = torch.randn(3, 4, 8)
    weights = module.edge_weights(embeddings, mask)

    # weight must be zero wherever the ring structural mask forbids an edge
    forbidden = mask.unsqueeze(0).expand_as(weights) == 0
    assert torch.all(weights[forbidden] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4), atol=1e-5)


def test_build_structural_mask_binary_no_self_loops():
    pytest.importorskip("torch")
    import torch

    from configs.schema import CommunicationConfig
    from training.utils import build_structural_mask

    agents = ["a0", "a1", "a2", "a3"]
    full = build_structural_mask(agents, CommunicationConfig(topology="fully_connected"), "cpu")
    assert set(torch.unique(full).tolist()) <= {0.0, 1.0}
    assert torch.equal(full.diagonal(), torch.zeros(4))
    assert full.sum().item() == 4 * 3  # every off-diagonal edge is a candidate

    ring = build_structural_mask(agents, CommunicationConfig(topology="ring", max_neighbours=2), "cpu")
    assert ring.sum().item() == 4 * 2  # each agent has two ring neighbours


def test_adaptive_policy_forward_and_weights():
    pytest.importorskip("torch")
    import torch

    from agents.policy import PolicyNetwork
    from configs.schema import CommunicationConfig
    from training.utils import build_structural_mask

    agents = ["a0", "a1", "a2", "a3"]
    mask = build_structural_mask(agents, CommunicationConfig(topology="fully_connected"), "cpu")
    net = PolicyNetwork(
        obs_dim=6, action_dim=5, hidden_dim=16, obs_embedding_dim=8,
        message_dim=4, comm_mode="adaptive", attention_dim=8,
    )
    obs = torch.randn(3, 4, 6)
    logits, value, messages = net(obs, mask)
    assert logits.shape == (3, 4, 5)
    assert value.shape == (3, 4)
    assert net._last_weights.shape == (3, 4, 4)  # maintained edge-weight matrix

    snapshot = net.edge_weights(obs, mask)
    assert snapshot.shape == (3, 4, 4)
    assert torch.allclose(snapshot.sum(dim=-1), torch.ones(3, 4), atol=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end (requires PettingZoo + MPE)
# --------------------------------------------------------------------------- #
TINY = [
    "training.total_steps=160",
    "training.rollout_length=40",
    "training.minibatch_size=40",
    "training.update_epochs=2",
    "training.log_every=40",
    "env.num_agents=4",
    "env.max_cycles=10",
    "agent.hidden_dim=32",
    "agent.obs_embedding_dim=16",
    "communication.message_dim=8",
    "communication.attention_dim=16",
]


def test_adaptive_training_exposes_stats_and_exports(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")

    import networkx as nx

    from configs import load_config
    from training.trainer import Trainer

    cfg = load_config("configs/adaptive.yaml", overrides=TINY)
    cfg.output_dir = str(tmp_path)

    try:
        trainer = Trainer(cfg)
        history = trainer.train()
    except ImportError as exc:  # MPE unavailable
        pytest.skip(f"environment unavailable: {exc}")

    assert trainer.comm_mode == "adaptive"
    final = history[-1]
    # graph statistics were logged during training
    for key in ("comm_edge_density", "comm_effective_degree", "comm_weight_entropy"):
        assert key in final and np.isfinite(final[key])
    # effective degree is a plausible number of attended senders (<= N-1)
    assert 0.0 < final["comm_effective_degree"] <= 3.0 + 1e-6

    # stats are persisted to the CSV
    header = (tmp_path / cfg.name / "metrics.csv").read_text().splitlines()[0]
    assert "comm_edge_density" in header

    # export the learned communication graph to NetworkX
    graph = trainer.export_communication_graph()
    assert graph is not None
    dg = graph.to_networkx()
    assert isinstance(dg, nx.DiGraph)
    assert dg.number_of_nodes() == 4
    assert dg.number_of_edges() > 0
