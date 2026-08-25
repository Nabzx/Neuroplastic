"""Tests for the metaplasticity mechanisms (S1: code only, no experiments)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from data.streams import PermutedRegressionStream  # noqa: E402
from mechanisms.base import MECHANISM_REGISTRY, make_mechanism  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402


def test_metaplastic_methods_registered():
    for name in ["metaplastic", "intrinsic", "selective_intrinsic", "btsp", "metaplastic_consolidation",
                 "metaplastic_homeostatic", "metaplastic_reactivation", "metaplastic_structural",
                 "selective_intrinsic_homeostatic", "metaplastic_combined", "btsp_homeostatic",
                 "metaplastic_triad"]:
        mech = make_mechanism(name)
        assert mech is not None
        assert name in MECHANISM_REGISTRY
    # BTSP-bearing mechanisms need the step context (input + loss); others do not
    assert make_mechanism("btsp").requires_context is True
    assert make_mechanism("metaplastic_triad").requires_context is True
    assert make_mechanism("homeostatic").requires_context is False
    # activity-driven variants need the forward activations; the weight-history foil does not
    assert make_mechanism("metaplastic").requires_activations is True
    assert make_mechanism("intrinsic").requires_activations is True
    assert make_mechanism("metaplastic_reactivation").requires_activations is True
    assert make_mechanism("metaplastic_structural").requires_activations is True
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


def test_intrinsic_plasticity_raises_bias_of_dormant_units():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    mech = make_mechanism("intrinsic", {"rate": 0.5, "ema_decay": 0.0})
    base = torch.ones(8, 4)
    mech.observe([base])                      # set-point a* = 1.0, activity = 1
    acts = base.clone()
    acts[:, 0] = 0.0                          # unit 0 dormant
    mech.observe([acts])                      # ema_decay 0 => activity = [0, 1, 1, 1]

    before = net.hidden_layers[0].bias.detach().clone()
    mech.after_optimizer_step(net, 0)
    delta = (net.hidden_layers[0].bias.detach() - before)
    assert delta[0] > 0                       # dormant unit: bias raised (gradient-free revival)
    assert float(delta[1]) == pytest.approx(0.0, abs=1e-6)   # at set-point: bias unchanged


def test_selective_intrinsic_revives_only_dead_units():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    mech = make_mechanism("selective_intrinsic", {"rate": 0.5, "ema_decay": 0.0, "dead_threshold": 0.15})
    base = torch.ones(8, 4)
    mech.observe([base])                      # set-point a* = 1.0
    acts = base.clone()
    acts[:, 0] = 0.0                          # unit 0 dead (0 < 0.15)
    acts[:, 1] = 0.5                          # unit 1 low but alive (0.5 > 0.15) -> must be left alone
    mech.observe([acts])                      # activity = [0, 0.5, 1, 1]

    before = net.hidden_layers[0].bias.detach().clone()
    mech.after_optimizer_step(net, 0)
    delta = net.hidden_layers[0].bias.detach() - before
    assert delta[0] > 0                                        # dead unit revived
    assert float(delta[1]) == pytest.approx(0.0, abs=1e-6)     # low-but-alive unit preserved
    assert float(delta[2]) == pytest.approx(0.0, abs=1e-6)     # healthy unit untouched


def test_btsp_imprints_dead_unit_toward_input_on_high_loss():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    mech = make_mechanism("btsp", {"rate": 1.0, "dead_threshold": 0.15, "loss_factor": 0.0,
                                    "ema_decay": 0.0, "refractory": 0})
    healthy = torch.ones(8, 4)
    mech.observe([healthy])                   # set-point a* = 1.0
    acts = healthy.clone()
    acts[:, 0] = 0.0                          # unit 0 dead
    mech.observe([acts])                      # activity = [0, 1, 1, 1]
    x = torch.zeros(8, 4)
    x[:, 1] = 1.0                             # input pattern: feature 1 active
    mech.observe_context(x, [acts], loss=0.5)  # loss_factor 0 => any positive loss triggers

    before = net.hidden_layers[0].weight.detach().clone()
    mech.after_optimizer_step(net, 0)
    dW = net.hidden_layers[0].weight.detach() - before
    assert float(dW[0, 1]) == pytest.approx(1.0, abs=1e-5)   # dead unit imprinted toward feature 1
    assert float(dW[0, 0]) == pytest.approx(0.0, abs=1e-5)   # ...only along the input direction
    assert torch.allclose(dW[1], torch.zeros(4), atol=1e-6)  # active unit untouched


def test_btsp_silent_without_instructive_signal():
    net = MLP(input_dim=4, hidden_dim=4, num_hidden_layers=1)
    mech = make_mechanism("btsp", {"rate": 1.0, "dead_threshold": 0.15, "loss_factor": 100.0,
                                    "ema_decay": 0.0})
    healthy = torch.ones(8, 4)
    mech.observe([healthy])
    acts = healthy.clone(); acts[:, 0] = 0.0
    mech.observe([acts])
    x = torch.zeros(8, 4); x[:, 1] = 1.0
    mech.observe_context(x, [acts], loss=0.5)   # 0.5 is not > 100 x loss-EMA => no plateau
    before = net.hidden_layers[0].weight.detach().clone()
    mech.after_optimizer_step(net, 0)
    assert torch.allclose(net.hidden_layers[0].weight.detach(), before)   # no imprint


@pytest.mark.parametrize("name", ["metaplastic", "intrinsic", "selective_intrinsic", "btsp",
                                   "metaplastic_homeostatic", "metaplastic_reactivation",
                                   "metaplastic_structural", "selective_intrinsic_homeostatic",
                                   "metaplastic_combined", "btsp_homeostatic", "metaplastic_triad",
                                   "metaplastic_consolidation"])
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
