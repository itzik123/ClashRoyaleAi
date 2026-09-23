"""`compute_shaping` zeroes the auto-reset step itself.

Every cumulative counter restarts at 0 when a vector env auto-resets. The delta
terms clamp negative differences, but the potential terms compare raw values,
so a reset would evaluate `gamma * Phi(new episode) - Phi(finished episode)`: a
large negative reward on a step where nothing happened. A monotone counter that
went down is a reset and nothing else, so the guard lives here rather than in
each caller.
"""
import numpy as np
import pytest

from python_ai.rewards.shaping import compute_shaping

#: A fixed discount rather than PPOConfig.gamma: these tests assert exact
#: numbers, and a test whose expected answer moves with the config pins
#: nothing. Whether the trainer passes its real gamma is pinned in
#: test_rl_config.py and test_reward_horizon_invariant.py.
SHAPING_TEST_GAMMA = 0.99



def _stats(n=1, **over):
    """A stats dict in `extract_engine_stats`' exact shape."""
    z = np.zeros(n, dtype=np.float32)
    s = {
        "team0_troop_damage": z.copy(), "team1_troop_damage": z.copy(),
        "team0_building_damage": z.copy(), "team1_building_damage": z.copy(),
        "team0_tower_damage": z.copy(), "team1_tower_damage": z.copy(),
        "team0_wincon_damage": z.copy(),
        "team0_elixir_spent": z.copy(), "team1_elixir_spent": z.copy(),
        "spell_in_hand": z.copy(), "spell_value_killed": z.copy(),
        "spell_elixir_spent": z.copy(),
        # A fixture Fireball, as the envs publish it (see
        # helpers.shaping_stats).
        "spell_damage": np.full(n, 689.0, dtype=np.float32),
        "spell_cost": np.full(n, 4.0, dtype=np.float32),
        "enemy_tower_hp": np.zeros((n, 3), dtype=np.float32),
        "team0_elixir_current": np.full(n, 5.0, dtype=np.float32),
        "team0_towers_alive": np.full(n, 3, dtype=np.int64),
        "team1_towers_alive": np.full(n, 3, dtype=np.int64),
    }
    for k, v in over.items():
        s[k] = np.asarray(v).reshape(np.shape(s[k])) if k in s else v
    return s


def test_an_auto_reset_step_contributes_exactly_zero_shaping():
    """The finished episode had dealt real tower damage and the new one none;
    unguarded, the whole differential reads as lost in one step.
    """
    prev = _stats(team0_tower_damage=[3000.0])   # we were winning
    cur = _stats()                                # counters restarted at 0

    out = compute_shaping(cur, prev, SHAPING_TEST_GAMMA)

    assert out.shape == (1,)
    assert float(out[0]) == 0.0, (
        f"auto-reset step emitted {float(out[0]):+.4f} of spurious shaping")


def test_the_reset_spike_would_otherwise_be_large_and_negative():
    """Pins the size of what the guard suppresses."""
    from python_ai.rewards.shaping import tower_potential
    prev = _stats(team0_tower_damage=[3000.0])
    spike = 0.99 * float(tower_potential(_stats())[0]) \
        - float(tower_potential(prev)[0])
    assert spike < -0.1, spike


def test_the_reset_guard_is_per_environment_not_all_or_nothing():
    """A vector env resets one worker at a time; zeroing the whole batch would
    discard the live envs' reward.
    """
    prev = _stats(2, team0_tower_damage=[3000.0, 1000.0])
    cur = _stats(2, team0_tower_damage=[0.0, 1200.0])  # env 0 reset, env 1 live

    out = compute_shaping(cur, prev, SHAPING_TEST_GAMMA)

    assert float(out[0]) == 0.0, "the reset env leaked a spike"
    assert float(out[1]) != 0.0, "the LIVE env was zeroed along with it"


def test_a_towers_alive_count_going_UP_is_a_reset():
    """`towers_alive` is monotone downward, so for it a reset looks like an
    increase.
    """
    # Every other counter is held equal, so `towers_alive` is the only witness
    # to the reset. Elixir sits above the overflow threshold so there is a real
    # term to suppress.
    prev = _stats(team0_towers_alive=[1], team1_towers_alive=[2],
                  team0_elixir_current=[9.9])
    cur = _stats(team0_towers_alive=[3], team1_towers_alive=[3],
                 team0_elixir_current=[9.9])
    assert float(compute_shaping(cur, _stats(team0_elixir_current=[9.9]), SHAPING_TEST_GAMMA)[0]) < 0.0,         "control: with no reset this configuration must earn a real penalty"
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) == 0.0


def test_an_ordinary_step_is_untouched_by_the_guard():
    """The guard must not fire on live play, or it deletes the dense reward."""
    prev = _stats(team0_tower_damage=[1000.0])
    cur = _stats(team0_tower_damage=[1300.0])
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) > 0.0


def test_the_first_step_of_an_episode_is_not_mistaken_for_a_reset():
    """All-zero to all-zero is not a decrease, so a fresh episode still earns
    shaping on its first real step.
    """
    out = compute_shaping(_stats(team0_elixir_current=[9.9]), _stats(), SHAPING_TEST_GAMMA)
    assert np.isfinite(out).all()
    assert float(out[0]) < 0.0, "the overflow penalty should still apply"


def test_every_cumulative_engine_counter_is_covered_by_the_reset_detector():
    """`_MONOTONE_UP` is a second copy of a list `rl.engine_stats` keeps; a
    counter added there and not here is a reset this guard would miss.
    """
    from python_ai.rewards.shaping import _MONOTONE_DOWN, _MONOTONE_UP
    from python_ai.rl import engine_stats as es

    cumulative = set(es._INT_KEYS) | set(es._FLOAT_KEYS)
    missing = cumulative - set(_MONOTONE_UP)
    assert not missing, f"cumulative counters the reset detector cannot see: {missing}"

    # ...and the two directions must not overlap.
    assert not (set(_MONOTONE_UP) & set(_MONOTONE_DOWN))


def test_the_reset_guard_survives_a_stats_dict_missing_optional_keys():
    """Keys can be absent (`extract_engine_stats` omits keys the info dict never
    carried); the detector degrades rather than raising.
    """
    prev = _stats(team0_tower_damage=[3000.0])
    cur = _stats()
    for d in (prev, cur):
        d.pop("team0_wincon_damage")
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) == 0.0
