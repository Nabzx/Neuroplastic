"""Tests for the plasticity mechanisms and the study aggregation."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from data.streams import PermutedRegressionStream  # noqa: E402
from experiments.study import aggregate_study, per_seed_scalars, significance_study  # noqa: E402
from mechanisms.base import MECHANISM_REGISTRY, make_mechanism  # noqa: E402
from mechanisms.biological import HomeostaticScaling, StructuralPlasticity  # noqa: E402
from models.mlp import MLP  # noqa: E402
from training.continual import ContinualTrainer  # noqa: E402

ALL = ["l2", "shrink_perturb", "redo", "continual_backprop", "homeostatic", "structural", "combined",
       "metaplastic", "metaplastic_consolidation", "metaplastic_homeostatic"]


def test_make_mechanism_all_registered():
    assert make_mechanism("vanilla") is None
    for name in ALL:
        mech = make_mechanism(name)
        assert mech is not None
        assert name in MECHANISM_REGISTRY or name in ("continual_backprop", "redo", "l2", "shrink_perturb")


def test_homeostatic_scaling_pulls_toward_setpoint():
    net = MLP(input_dim=8, hidden_dim=16, num_hidden_layers=1)
    mech = HomeostaticScaling({"rate": 1.0})       # full pull to set-point
    mech.after_optimizer_step(net, 0)              # records set-point norms
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(3.0)             # blow weights up
    mech.after_optimizer_step(net, 1)              # should rescale back
    norms = net.hidden_layers[0].weight.norm(dim=1)
    target = mech._targets[0].squeeze(1)
    assert torch.allclose(norms, target, atol=1e-4)


def test_structural_plasticity_resets_dormant_unit():
    net = MLP(input_dim=8, hidden_dim=8, num_hidden_layers=1)
    mech = StructuralPlasticity({"selection": "utility", "replacement_rate": 0.5, "period": 1, "maturity": 0})
    # unit 0 is dead (zero utility), others active
    acts = torch.ones(4, 8)
    acts[:, 0] = 0.0
    original = net.hidden_layers[0].weight[0].clone()
    for _ in range(3):
        mech.observe([acts])
    mech.after_optimizer_step(net, 0)              # step 1 -> period boundary
    # the dead unit's incoming weights were re-initialised (changed)
    assert not torch.allclose(net.hidden_layers[0].weight[0], original)


def test_mechanism_runs_in_trainer():
    stream = PermutedRegressionStream(input_dim=8, num_tasks=3, task_length=20, seed=0)
    net = MLP(input_dim=8, hidden_dim=16, num_hidden_layers=2)
    trainer = ContinualTrainer(net, stream, torch.optim.SGD(net.parameters(), lr=0.01), mechanism=make_mechanism("combined"))
    history = trainer.run()
    assert len(history) == 3


def test_study_aggregation_and_significance():
    # tiny synthetic histories: two methods, two seeds
    def hist(base):
        return [{"train_loss_late": base + 0.01 * t, "dormant_fraction": 0.1,
                 "effective_rank": 10.0, "weight_magnitude": 0.1} for t in range(40)]

    results = {"vanilla": [hist(0.8), hist(0.82)], "homeostatic": [hist(0.5), hist(0.52)]}
    summary = aggregate_study(results, window=10)
    assert summary["homeostatic"]["final_late_loss"]["mean"] < summary["vanilla"]["final_late_loss"]["mean"]
    scal = per_seed_scalars(hist(0.8), window=10)
    assert "plasticity_ratio" in scal and "final_dormant_fraction" in scal
    sig = significance_study(results, window=10, baselines=("vanilla",))
    assert "vs_vanilla" in sig["homeostatic"]
