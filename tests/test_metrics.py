"""Information-theoretic, coordination and analysis helpers (functional)."""

import numpy as np
import pytest

from analysis.statistics import bootstrap_ci
from analysis.specialisation_analysis import role_cluster_count, role_entropy
from evaluation.coordination import episode_return
from evaluation.information_metrics import mutual_information, shannon_entropy


def test_entropy_of_constant_is_zero():
    assert shannon_entropy(np.ones(100)) == pytest.approx(0.0)


def test_entropy_positive_for_spread():
    rng = np.random.default_rng(0)
    assert shannon_entropy(rng.uniform(size=1000), bins=16) > 0.0


def test_mutual_information_identical_vs_independent():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    mi_self = mutual_information(x, x, bins=16)
    mi_indep = mutual_information(x, rng.normal(size=2000), bins=16)
    assert mi_self > mi_indep


def test_episode_return_sums_team_rewards():
    stream = [{"a": 1.0, "b": 2.0}, {"a": 0.5, "b": -0.5}]
    assert episode_return(stream) == pytest.approx(3.0)


def test_role_entropy_and_count():
    assert role_cluster_count([0, 0, 1, 1, 2]) == 3
    assert role_entropy([0, 0, 0, 0]) == pytest.approx(0.0)   # one role
    assert role_entropy([0, 1]) == pytest.approx(1.0)          # two equal roles -> 1 bit


def test_bootstrap_ci_brackets_mean():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lo, hi = bootstrap_ci(samples, seed=0)
    assert mean == pytest.approx(3.0)
    assert lo <= mean <= hi
