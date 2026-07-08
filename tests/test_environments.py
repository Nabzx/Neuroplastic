"""Environment layer: registry wiring (light) + PettingZoo smoke test (heavy).

The registry/StepResult tests run with only the core deps. The smoke test that
actually steps an environment is skipped unless PettingZoo (with MPE) is
installed, so the suite stays green on a core-only install.
"""

import pytest

from configs.schema import EnvConfig
from environments.base import StepResult


# --------------------------------------------------------------------------- #
# Light tests (no PettingZoo required)
# --------------------------------------------------------------------------- #
def test_benchmarks_are_registered():
    import environments.benchmarks  # noqa: F401  (registration side-effect)
    from environments.registry import ENV_REGISTRY

    for name in ("simple_spread", "simple_reference", "simple_speaker_listener"):
        assert name in ENV_REGISTRY


def test_step_result_unpacks_like_tuple():
    result = StepResult(
        observations={"a": 0, "b": 1},
        rewards={"a": 1.0, "b": -1.0},
        terminations={"a": False, "b": False},
        truncations={"a": True, "b": True},
        infos={"a": {}, "b": {}},
    )
    obs, rew, term, trunc, info = result  # tuple-style unpacking
    assert obs == {"a": 0, "b": 1}
    assert rew["a"] == 1.0
    assert result.agents == ["a", "b"]
    assert result.dones == {"a": True, "b": True}  # truncation counts as done
    assert result.all_done is True


def test_preprocessor_registry_has_supersuit_wrappers():
    from environments.wrappers import PREPROCESSOR_REGISTRY

    for name in ("pad_observations", "pad_action_space", "flatten"):
        assert name in PREPROCESSOR_REGISTRY


# --------------------------------------------------------------------------- #
# Smoke test (requires PettingZoo + MPE)
# --------------------------------------------------------------------------- #
def _make_spread_env():
    from environments.registry import make_env

    cfg = EnvConfig(name="simple_spread", num_agents=3, max_cycles=10)
    return make_env(cfg)


@pytest.fixture
def spread_env():
    pytest.importorskip("pettingzoo", reason="PettingZoo not installed ([rl] extra)")
    try:
        env = _make_spread_env()
    except ImportError as exc:  # MPE package not available
        pytest.skip(f"MPE environments unavailable: {exc}")
    yield env
    env.close()


def test_smoke_random_agents_several_episodes(spread_env):
    """Run random agents for several episodes and validate the interface."""
    from environments.random_rollout import run_random_episodes

    env = spread_env
    assert env.num_agents == 3
    assert len(env.possible_agents) == 3

    stats = run_random_episodes(env, num_episodes=3, seed=0)
    assert len(stats) == 3
    for episode in stats:
        # simple_spread truncates at max_cycles; every episode should run to it.
        assert episode.length == 10
        # a reward was recorded for every agent
        assert set(episode.returns) == set(env.possible_agents)
        assert isinstance(episode.total_return, float)


def test_smoke_observations_match_spaces(spread_env):
    """Observations/rewards/dones are keyed by agent id with correct shapes."""
    env = spread_env
    observations, infos = env.reset(seed=0)

    assert set(observations) == set(env.agents)
    assert set(infos) == set(env.agents)
    for agent, obs in observations.items():
        assert env.observation_space(agent).contains(obs)

    actions = env.sample_actions()
    assert set(actions) == set(env.agents)

    result = env.step(actions)
    assert set(result.rewards) == set(env.agents)
    assert set(result.terminations) == set(env.agents)
    assert set(result.truncations) == set(env.agents)


def test_smoke_seeding_is_reproducible(spread_env):
    """Same seed -> identical initial observations and identical rollout stats."""
    import numpy as np

    env = spread_env

    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    for agent in env.agents:
        np.testing.assert_array_equal(obs_a[agent], obs_b[agent])

    from environments.random_rollout import run_random_episodes

    run_a = run_random_episodes(env, num_episodes=2, seed=7)
    run_b = run_random_episodes(env, num_episodes=2, seed=7)
    assert [e.length for e in run_a] == [e.length for e in run_b]
    assert [e.returns for e in run_a] == [e.returns for e in run_b]


def test_smoke_different_seeds_differ(spread_env):
    """Different seeds should generally produce different initial states."""
    import numpy as np

    env = spread_env
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    agent = env.agents[0]
    assert not np.array_equal(obs_a[agent], obs_b[agent])
