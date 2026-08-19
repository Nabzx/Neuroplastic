"""Tests for the continual-plasticity core: data, model, diagnostics, training."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from data.streams import PermutedRegressionStream  # noqa: E402
from diagnostics.plasticity import dormant_fraction, effective_rank, weight_magnitude  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402


# --------------------------------------------------------------------------- #
# Data stream
# --------------------------------------------------------------------------- #
def test_stream_shapes_and_boundaries():
    stream = PermutedRegressionStream(input_dim=8, num_tasks=3, task_length=10, batch_size=2, seed=0)
    steps = list(stream.steps())
    assert len(steps) == 3 * (10 // 2)
    assert steps[0].x.shape == (2, 8) and steps[0].y.shape == (2,)
    boundaries = [s for s in steps if s.boundary]
    assert len(boundaries) == 3                       # one per task
    assert sorted({s.task_id for s in steps}) == [0, 1, 2]


def test_stream_reproducible():
    a = list(PermutedRegressionStream(num_tasks=2, task_length=4, seed=1).steps())
    b = list(PermutedRegressionStream(num_tasks=2, task_length=4, seed=1).steps())
    assert torch.allclose(a[0].x, b[0].x) and torch.allclose(a[0].y, b[0].y)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_forward_and_activations():
    net = MLP(input_dim=8, hidden_dim=16, num_hidden_layers=2)
    x = torch.randn(5, 8)
    out = net(x)
    assert out.shape == (5,)
    out2, acts = net(x, return_activations=True)
    assert len(acts) == 2 and acts[0].shape == (5, 16)
    assert net.hidden_units() == 32


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_dormant_fraction_detects_dead_units():
    acts = torch.ones(4, 10)
    acts[:, :5] = 0.0                                  # half the units are dead
    assert dormant_fraction([acts], tau=0.0) == pytest.approx(0.5)


def test_effective_rank_orders_correctly():
    rank1 = torch.ones(20, 8) * torch.randn(8)         # rank-1 features
    full = torch.randn(20, 8)
    assert effective_rank(rank1) < effective_rank(full)


def test_weight_magnitude_positive():
    net = MLP(input_dim=4, hidden_dim=8, num_hidden_layers=1)
    assert weight_magnitude(net) > 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_continual_training_records_per_task():
    stream = PermutedRegressionStream(input_dim=8, num_tasks=4, task_length=20, seed=0)
    net = MLP(input_dim=8, hidden_dim=16, num_hidden_layers=2)
    trainer = ContinualTrainer(net, stream, torch.optim.SGD(net.parameters(), lr=0.01))
    history = trainer.run()
    assert len(history) == 4
    row = history[0]
    for key in ("task", "train_loss_late", "dormant_fraction", "effective_rank", "weight_magnitude"):
        assert key in row
    assert all(r["train_loss_late"] == r["train_loss_late"] for r in history)  # finite (not NaN)
