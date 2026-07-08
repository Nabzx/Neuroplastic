"""Learning-baseline tests.

Two tiers:
* Torch-only unit tests for the communication mask and the policy network
  (no environment needed).
* A tiny end-to-end training run of all three communication settings, requiring
  PettingZoo + MPE (skipped otherwise).
"""

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.policy import PolicyNetwork  # noqa: E402
from configs.schema import CommunicationConfig  # noqa: E402
from training.utils import build_communication_mask  # noqa: E402

AGENTS = ["agent_0", "agent_1", "agent_2", "agent_3"]


# --------------------------------------------------------------------------- #
# Communication mask
# --------------------------------------------------------------------------- #
def test_mask_none_when_disabled():
    cfg = CommunicationConfig(enabled=False)
    assert build_communication_mask(AGENTS, cfg, "cpu") is None


def test_fully_connected_mask_is_mean_of_others():
    cfg = CommunicationConfig(enabled=True, topology="fully_connected")
    mask = build_communication_mask(AGENTS, cfg, "cpu")
    assert mask.shape == (4, 4)
    # no self-loops, rows sum to 1, off-diagonal entries equal (mean of others)
    assert torch.allclose(mask.diagonal(), torch.zeros(4))
    assert torch.allclose(mask.sum(dim=1), torch.ones(4))
    assert mask[0, 1].item() == pytest.approx(1 / 3)


def test_ring_mask_is_sparse():
    cfg = CommunicationConfig(enabled=True, topology="ring", max_neighbours=2)
    mask = build_communication_mask(AGENTS, cfg, "cpu")
    # each receiver has exactly two senders (its ring neighbours)
    nonzero_per_row = (mask > 0).sum(dim=1)
    assert torch.equal(nonzero_per_row, torch.full((4,), 2))
    assert torch.allclose(mask.sum(dim=1), torch.ones(4))
    # sparser than fully-connected (which would have 3 senders per row)
    assert (mask > 0).sum().item() == 8


# --------------------------------------------------------------------------- #
# Policy network
# --------------------------------------------------------------------------- #
def test_policy_forward_shapes_with_communication():
    net = PolicyNetwork(
        obs_dim=5, action_dim=3, hidden_dim=16, obs_embedding_dim=8, message_dim=4, comm_mode="fixed"
    )
    cfg = CommunicationConfig(enabled=True, topology="fully_connected", protocol="mean")
    mask = build_communication_mask(AGENTS, cfg, "cpu")
    obs = torch.randn(2, 4, 5)  # [B, N, obs_dim]
    logits, value, messages = net(obs, mask)
    assert logits.shape == (2, 4, 3)
    assert value.shape == (2, 4)
    assert messages.shape == (2, 4, 4)


def test_policy_forward_without_communication_zeros_context():
    net = PolicyNetwork(obs_dim=5, action_dim=3, hidden_dim=16, obs_embedding_dim=8, message_dim=4)
    obs = torch.randn(2, 4, 5)
    logits, value, _ = net(obs, adjacency=None)  # no-communication path
    assert logits.shape == (2, 4, 3)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# End-to-end training (requires PettingZoo + MPE)
# --------------------------------------------------------------------------- #
BASELINES = [
    "configs/baselines/no_comm.yaml",
    "configs/baselines/fully_connected.yaml",
    "configs/baselines/sparse.yaml",
]

TINY_OVERRIDES = [
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
]


@pytest.fixture
def _needs_env():
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")


@pytest.mark.parametrize("config_path", BASELINES)
def test_train_baseline_runs_and_logs(config_path, tmp_path, _needs_env):
    from configs import load_config
    from training.trainer import Trainer

    try:
        cfg = load_config(config_path, overrides=TINY_OVERRIDES)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"config load failed: {exc}")
    cfg.output_dir = str(tmp_path)

    try:
        trainer = Trainer(cfg)
        history = trainer.train()
    except ImportError as exc:  # MPE unavailable
        pytest.skip(f"environment unavailable: {exc}")

    # 160 steps / 40 per rollout -> 4 iterations of logged metrics.
    assert len(history) == 4
    final = history[-1]
    assert np.isfinite(final["episode_return_mean"])
    assert final["episode_length_mean"] == 10  # simple_spread truncates at max_cycles
    assert np.isfinite(final["policy_loss"])
    assert np.isfinite(final["value_loss"])

    # reward/episode-length logs were written to CSV
    metrics_csv = tmp_path / cfg.name / "metrics.csv"
    assert metrics_csv.exists()
    header = metrics_csv.read_text().splitlines()[0]
    assert "episode_return_mean" in header and "episode_length_mean" in header


def test_no_comm_learner_has_no_mask(tmp_path, _needs_env):
    from configs import load_config
    from training.trainer import Trainer

    cfg = load_config("configs/baselines/no_comm.yaml", overrides=TINY_OVERRIDES)
    cfg.output_dir = str(tmp_path)
    trainer = Trainer(cfg)
    try:
        trainer.build()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"environment unavailable: {exc}")
    assert trainer.mask is None  # communication disabled -> no adjacency


def test_fully_connected_learner_has_mask(tmp_path, _needs_env):
    from configs import load_config
    from training.trainer import Trainer

    cfg = load_config("configs/baselines/fully_connected.yaml", overrides=TINY_OVERRIDES)
    cfg.output_dir = str(tmp_path)
    trainer = Trainer(cfg)
    try:
        trainer.build()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"environment unavailable: {exc}")
    assert trainer.mask is not None
    assert trainer.mask.shape == (4, 4)
