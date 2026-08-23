"""Tests for the metaplasticity mechanisms (S1: code only, no experiments)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from data.streams import PermutedRegressionStream  # noqa: E402
from mechanisms.base import MECHANISM_REGISTRY, make_mechanism  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402


def test_metaplastic_methods_registered():
    for name in ["metaplastic", "metaplastic_consolidation", "metaplastic_homeostatic"]:
        mech = make_mechanism(name)
        assert mech is not None
        assert name in MECHANISM_REGISTRY
    # activity-driven variants need the forward activations; the weight-history foil does not
    assert make_mechanism("metaplastic").requires_activations is True
    assert make_mechanism("metaplastic_homeostatic").requires_activations is True
    assert make_mechanism("metaplastic_consolidation").requires_activations is False


def _primed_metaplastic():
    """A MetaplasticLR whose activity EMA marks unit 0 dormant and unit 3 hyperactive."""
    mech = make_mechanism("metaplastic", {"beta": 1.0, "g_min": 0.1, "g_max": 10.0, "ema_decay": 0.0})
    base = torch.ones(8, 4)
    mech.observe([base])              # set-point a* = mean activity at first sight = 1.0
    acts = base.clone()
    acts[:, 0] = 0.0                  # unit 0 dormant
    acts[:, 3] = 5.0                  # unit 3 hyperactive
    mech.observe([acts])             # ema_decay 0 => EMA tracks the latest batch: [0, 1, 1, 5]
    return mech


def test_metaplastic_gain_boosts_dormant_damps_hyperactive():
    g = _primed_metaplastic().layer_gain(0)
    assert g[0] == pytest.approx(10.0)          # dormant unit: boosted (hits g_max)
    assert g[1] == pytest.approx(1.0)           # at set-point: unmodulated
    assert g[3] == pytest.approx(0.2)           # hyperactive unit: damped
    assert float(g.min()) >= 0.1 and float(g.max()) <= 10.0   # bounded by g_min / g_max


def test_metaplastic_rescales_realised_update_as_exact_per_unit_lr():
    mech = _primed_metaplastic()
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    g = mech.layer_gain(0)

    orig = net.hidden_layers[0].weight.clone()
    mech.before_optimizer_step(net, 0)                      # snapshot (== orig)
    with torch.no_grad():
        net.hidden_layers[0].weight.add_(0.1)              # a uniform "optimiser step"
    mech.after_optimizer_step(net, 0)

    applied = net.hidden_layers[0].weight - orig
    assert torch.allclose(applied, g.unsqueeze(1) * 0.1, atol=1e-6)   # exact per-unit LR
    # dormant unit moved most, hyperactive least
    assert applied[0].abs().mean() > applied[1].abs().mean() > applied[3].abs().mean()


def test_consolidation_foil_protects_large_settled_weights():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    mech = make_mechanism("metaplastic_consolidation", {"lambda": 10.0, "ema_decay": 0.0})
    with torch.no_grad():
        net.hidden_layers[0].weight[0] = 5.0               # large -> will consolidate
        net.hidden_layers[0].weight[1] = 0.01              # small -> stays plastic

    mech.before_optimizer_step(net, 0)                     # step A builds the consolidation state
    with torch.no_grad():
        net.hidden_layers[0].weight.add_(0.1)
    mech.after_optimizer_step(net, 0)

    orig = net.hidden_layers[0].weight.clone()
    mech.before_optimizer_step(net, 1)                     # step B: now the large row is protected
    with torch.no_grad():
        net.hidden_layers[0].weight.add_(0.1)
    mech.after_optimizer_step(net, 1)

    applied = net.hidden_layers[0].weight - orig
    assert applied[0].abs().mean() < applied[1].abs().mean()   # consolidated row moved less


@pytest.mark.parametrize("name", ["metaplastic", "metaplastic_homeostatic", "metaplastic_consolidation"])
def test_metaplastic_runs_in_trainer(name):
    stream = PermutedRegressionStream(input_dim=8, num_tasks=3, task_length=20, seed=0)
    net = MLP(input_dim=8, hidden_dim=16, num_hidden_layers=2)
    trainer = ContinualTrainer(
        net, stream, torch.optim.SGD(net.parameters(), lr=0.01), mechanism=make_mechanism(name)
    )
    history = trainer.run()
    assert len(history) == 3
    for rec in history:
        assert torch.isfinite(torch.tensor(rec["train_loss_late"]))


def test_before_optimizer_step_is_noop_for_existing_mechanisms():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    homeo = make_mechanism("homeostatic")                  # does not implement the new hook
    before = net.hidden_layers[0].weight.clone()
    homeo.before_optimizer_step(net, 0)                    # inherited no-op: must not touch weights
    assert torch.allclose(net.hidden_layers[0].weight, before)
