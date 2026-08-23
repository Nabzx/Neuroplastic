"""Tests for the A1 (independent teachers) and A2 (optimiser/scale) infrastructure."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from analysis.statistics import holm_bonferroni, permutation_test  # noqa: E402
from data.streams import PermutedRegressionStream  # noqa: E402
from experiments.study import _build_optimizer, aggregate_study, run_study, significance_study  # noqa: E402
from mechanisms.base import make_mechanism  # noqa: E402
from models.mlp import MLP  # noqa: E402


def test_build_optimizer_selects_adam_or_sgd():
    net = MLP(4, 8, num_hidden_layers=1)
    assert isinstance(_build_optimizer(net, {"optimizer": "adam", "lr": 0.01}), torch.optim.Adam)
    assert isinstance(_build_optimizer(net, {"optimizer": "sgd", "lr": 0.01}), torch.optim.SGD)
    assert isinstance(_build_optimizer(net, {"lr": 0.01}), torch.optim.SGD)   # default


def test_independent_vs_shared_teacher_stream():
    ind = PermutedRegressionStream(input_dim=8, num_tasks=4, task_length=4, seed=0, shared_teacher=False)
    assert len(ind._w1) == 4 and len(ind._y_std) == 4          # a fresh teacher per task
    sh = PermutedRegressionStream(input_dim=8, num_tasks=4, task_length=4, seed=0, shared_teacher=True)
    assert len(sh._w1) == 1                                    # one shared teacher
    # independent teachers differ from each other
    assert not torch.allclose(ind._w1[0], ind._w1[1])


def test_shared_teacher_reproduces_previous_behaviour():
    # same seed => identical first batch (shared-teacher path is unchanged)
    a = list(PermutedRegressionStream(num_tasks=2, task_length=4, seed=3).steps())
    b = list(PermutedRegressionStream(num_tasks=2, task_length=4, seed=3, shared_teacher=True).steps())
    assert torch.allclose(a[0].x, b[0].x) and torch.allclose(a[0].y, b[0].y)


def test_study_runs_with_adam_and_independent_teachers():
    cfg = {
        "benchmark": "permuted_regression", "num_tasks": 3, "task_length": 20,
        "hidden_dim": 16, "lr": 0.01, "optimizer": "adam", "shared_teacher": False,
    }
    bundle = run_study(["vanilla", "homeostatic"], [0], cfg)
    summary = aggregate_study(bundle["results"], window=1)
    assert set(summary) == {"vanilla", "homeostatic"}
    assert len(bundle["results"]["vanilla"][0]) == 3          # one record per task


# --------------------------------------------------------------------------- #
# A3/A4: sensitivity hyper-parameters + statistics rigor
# --------------------------------------------------------------------------- #
def test_mechanism_config_override():
    mech = make_mechanism("homeostatic", {"rate": 0.5})
    assert mech.rate == 0.5                                   # sensitivity sweep can override
    struct = make_mechanism("structural", {"replacement_rate": 5e-3})
    assert struct.replacement_rate == 5e-3


def test_permutation_test_reports_cohens_d():
    result = permutation_test([10, 11, 12, 13, 14], [0, 1, 2, 3, 4])
    assert "cohens_d" in result and result["cohens_d"] > 2.0  # large effect


def test_holm_bonferroni_monotone():
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)                 # 3 * 0.01
    assert all(0.0 <= a <= 1.0 for a in adjusted)
    assert adjusted[0] <= adjusted[2]                         # most significant stays smallest


def test_significance_study_adds_holm_correction():
    def hist(base):
        return [{"train_loss_late": base, "dormant_fraction": 0.1, "effective_rank": 10.0, "weight_magnitude": 0.1} for _ in range(30)]

    results = {"vanilla": [hist(0.8), hist(0.82)], "homeostatic": [hist(0.5), hist(0.52)]}
    sig = significance_study(results, window=10, baselines=("vanilla",))
    entry = sig["homeostatic"]["vs_vanilla"]
    assert "p_value_holm" in entry and "cohens_d" in entry
