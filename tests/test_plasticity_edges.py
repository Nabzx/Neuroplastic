"""Hebbian-inspired plastic communication edges.

Unit tests for the reward-gated Hebbian rule (strengthen / decay / clamp /
modulation / configurable coefficients) plus an end-to-end plastic training run
that checks the edge weights evolve and the evolution is logged.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from configs.schema import PlasticityConfig  # noqa: E402
from plasticity.plastic_edges import PlasticEdges  # noqa: E402


def _mask(n: int) -> torch.Tensor:
    m = torch.ones(n, n)
    m.fill_diagonal_(0.0)
    return m


def _config(**kw) -> PlasticityConfig:
    base = dict(learning_rate=0.5, decay=0.0, modulation="none", homeostasis=False, max_weight=10.0)
    base.update(kw)
    return PlasticityConfig(**base)


# --------------------------------------------------------------------------- #
# Aggregation / initialisation
# --------------------------------------------------------------------------- #
def test_init_uniform_and_normalised():
    edges = PlasticEdges(3, _mask(3), _config())
    # starts at the candidate mask (uniform), no self-loops
    assert torch.equal(edges.edge_weights, _mask(3))
    weights = edges.aggregation_weights()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
    assert torch.allclose(weights.diagonal(), torch.zeros(3))


def test_aggregation_is_weighted_mean():
    edges = PlasticEdges(3, _mask(3), _config())
    messages = torch.tensor([[[1.0], [2.0], [3.0]]])  # [1, 3, 1]
    context = edges(messages)
    # receiver 0 averages senders 1,2 -> (2+3)/2 = 2.5
    assert context[0, 0, 0].item() == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Hebbian rule
# --------------------------------------------------------------------------- #
def test_correlated_edge_strengthens():
    edges = PlasticEdges(3, _mask(3), _config(learning_rate=0.5))
    coactivity = torch.zeros(3, 3)
    coactivity[0, 1] = coactivity[1, 0] = 1.0     # agents 0 and 1 communicate in sync
    edges.hebbian_update(coactivity, reward=1.0)  # modulation "none" -> m = 1
    # the co-active edge grew; an idle edge (0 -> 2) did not
    assert edges.edge_weights[0, 1].item() == pytest.approx(1.5)
    assert edges.edge_weights[0, 2].item() == pytest.approx(1.0)


def test_decay_weakens_unused_edges():
    edges = PlasticEdges(3, _mask(3), _config(learning_rate=0.0, decay=0.5))
    edges.hebbian_update(torch.zeros(3, 3), reward=1.0)
    # no coactivity, no reward drive -> pure decay: 1.0 -> 0.5
    active = _mask(3) > 0
    assert torch.allclose(edges.edge_weights[active], torch.full((6,), 0.5))


def test_weights_are_clamped():
    edges = PlasticEdges(3, _mask(3), _config(learning_rate=100.0, max_weight=1.0))
    edges.hebbian_update(torch.ones(3, 3), reward=1.0)
    assert edges.edge_weights.max().item() <= 1.0 + 1e-6
    assert edges.edge_weights.min().item() >= 0.0


def test_reward_modulation_sign():
    edges = PlasticEdges(3, _mask(3), _config(modulation="reward_gated"))
    first = edges.hebbian_update(torch.zeros(3, 3), reward=1.0)   # sets baseline -> m = 0
    assert first["plast_modulation"] == pytest.approx(0.0, abs=1e-6)
    better = edges.hebbian_update(torch.zeros(3, 3), reward=5.0)  # above baseline -> m > 0
    assert better["plast_modulation"] > 0.0
    worse = edges.hebbian_update(torch.zeros(3, 3), reward=-5.0)  # below baseline -> m < 0
    assert worse["plast_modulation"] < 0.0


def test_learning_rate_is_configurable():
    coactivity = torch.zeros(3, 3)
    coactivity[0, 1] = 1.0
    slow = PlasticEdges(3, _mask(3), _config(learning_rate=0.1))
    fast = PlasticEdges(3, _mask(3), _config(learning_rate=1.0))
    slow.hebbian_update(coactivity, reward=1.0)
    fast.hebbian_update(coactivity, reward=1.0)
    slow_delta = slow.edge_weights[0, 1].item() - 1.0
    fast_delta = fast.edge_weights[0, 1].item() - 1.0
    assert fast_delta == pytest.approx(10 * slow_delta, rel=1e-3)


def test_homeostasis_caps_incoming_weight():
    edges = PlasticEdges(3, _mask(3), _config(learning_rate=5.0, homeostasis=True, max_weight=1.0))
    edges.hebbian_update(torch.ones(3, 3), reward=1.0)
    # each receiver's incoming weights are rescaled to sum <= max_weight
    assert torch.all(edges.edge_weights.sum(dim=-1) <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- #
# Network integration
# --------------------------------------------------------------------------- #
def test_policy_plastic_forward():
    from agents.policy import PolicyNetwork

    net = PolicyNetwork(
        obs_dim=6, action_dim=5, hidden_dim=16, obs_embedding_dim=8, message_dim=4,
        comm_mode="plastic", num_agents=4, structural_mask=_mask(4), plasticity_config=_config(),
    )
    obs = torch.randn(3, 4, 6)
    logits, value, messages = net(obs, None)
    assert logits.shape == (3, 4, 5)
    assert value.shape == (3, 4)
    assert net.plastic is not None
    assert net.messages_only(obs).shape == (3, 4, 4)


# --------------------------------------------------------------------------- #
# End-to-end (requires PettingZoo + MPE)
# --------------------------------------------------------------------------- #
TINY = [
    "training.total_steps=200",
    "training.rollout_length=50",
    "training.minibatch_size=50",
    "training.update_epochs=2",
    "training.log_every=50",
    "env.num_agents=4",
    "env.max_cycles=10",
    "agent.hidden_dim=32",
    "agent.obs_embedding_dim=16",
    "communication.message_dim=8",
]


def test_plastic_training_evolves_and_logs(tmp_path):
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")

    from configs import load_config
    from training.trainer import Trainer

    cfg = load_config("configs/plastic.yaml", overrides=TINY)
    cfg.output_dir = str(tmp_path)

    try:
        trainer = Trainer(cfg)
        history = trainer.train()
    except ImportError as exc:  # MPE unavailable
        pytest.skip(f"environment unavailable: {exc}")

    assert trainer.comm_mode == "plastic"
    final = history[-1]
    for key in ("plast_modulation", "plast_mean_weight", "plast_update_norm", "comm_edge_density"):
        assert key in final and np.isfinite(final[key])

    # the plastic edge matrix moved away from its uniform initialisation
    final_matrix = trainer.learner.net.plastic.current_matrix()
    initial_mask = _mask(4)
    assert not torch.allclose(final_matrix, initial_mask)
    assert final["plast_update_norm"] > 0.0

    # evolution logged to CSV and saved as a weight-history archive
    header = (tmp_path / cfg.name / "metrics.csv").read_text().splitlines()[0]
    assert "plast_modulation" in header
    archive = tmp_path / cfg.name / "edge_weights.npz"
    assert archive.exists()
    data = np.load(archive)
    assert data["weights"].shape[0] == len(history)   # one snapshot per iteration
    assert data["weights"].shape[1:] == (4, 4)
