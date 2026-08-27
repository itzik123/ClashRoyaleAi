"""`compute_shaping`: the auto-reset spike, and that it cannot be re-lost.

WHY THIS FILE EXISTS. The shaping terms are pure functions of the engine's
cumulative counters, and every one of those counters RESTARTS AT 0 when a
vector env auto-resets. The delta-based terms defend themselves -- `delta()`
clamps a negative difference to 0 -- but the two POTENTIAL-based terms
(`tower_potential`, `lethal_spell_potential`) are computed on the raw values,
not on deltas, so at a reset they evaluate

    gamma * Phi(brand new episode) - Phi(finished episode)

which is a large NEGATIVE reward landing exactly once per episode, on a step
where nothing happened.

The guard for that lived in the CALLERS -- `base_trainer` and `exploiter` each
multiply the result by `(1 - prev_dones)`. Two copies of one invariant, in a
codebase whose own rule is "don't keep a second copy": the exploiter shipped
WITHOUT it through burst #0, which is this exact bug reaching production once
already. `compute_shaping` can detect the reset unaided -- a monotone counter
that went DOWN is a reset, and nothing else -- so the invariant belongs here,
where a third caller cannot forget it.

This is a SAFETY NET, not a behaviour change: both live callers already zero
these steps, so for them the result is 0 either way and no win rate moves.
"""
import numpy as np
import pytest

from python_ai.rewards.shaping import compute_shaping

#: The discount these arithmetic tests are written against. Deliberately a
#: FIXED fixture value and NOT `PPOConfig.gamma`: these cases assert exact
#: numbers out of `gamma*Phi(s') - Phi(s)`, so reading the live config would
#: make their expected values move every time someone tunes the discount --
#: a test that changes its own answer cannot pin anything. The separate
#: question of whether the TRAINER passes its real gamma is pinned by
#: tests/test_rl_config.py and tests/test_reward_horizon_invariant.py.
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
        "fireball_in_hand": z.copy(), "fireball_value_killed": z.copy(),
        "fireball_elixir_spent": z.copy(),
        "enemy_tower_hp": np.zeros((n, 3), dtype=np.float32),
        "team0_elixir_current": np.full(n, 5.0, dtype=np.float32),
        "team0_towers_alive": np.full(n, 3, dtype=np.int64),
        "team1_towers_alive": np.full(n, 3, dtype=np.int64),
    }
    for k, v in over.items():
        s[k] = np.asarray(v).reshape(np.shape(s[k])) if k in s else v
    return s


def test_an_auto_reset_step_contributes_exactly_zero_shaping():
    """The finished episode had dealt real tower damage; the new one has dealt
    none. Without the guard this reads as the whole differential being lost in
    a single step."""
    prev = _stats(team0_tower_damage=[3000.0])   # we were winning
    cur = _stats()                                # counters restarted at 0

    out = compute_shaping(cur, prev, SHAPING_TEST_GAMMA)

    assert out.shape == (1,)
    assert float(out[0]) == 0.0, (
        f"auto-reset step emitted {float(out[0]):+.4f} of spurious shaping")


def test_the_reset_spike_would_otherwise_be_large_and_negative():
    """Pins the SIZE of what the guard suppresses, so nobody later reads this
    as defending against a rounding error."""
    from python_ai.rewards.shaping import tower_potential
    prev = _stats(team0_tower_damage=[3000.0])
    spike = 0.99 * float(tower_potential(_stats())[0]) \
        - float(tower_potential(prev)[0])
    assert spike < -0.1, spike


def test_the_reset_guard_is_per_environment_not_all_or_nothing():
    """A vector env resets one worker at a time. Zeroing the whole batch
    because ONE env reset would discard real gradient from the other seven."""
    prev = _stats(2, team0_tower_damage=[3000.0, 1000.0])
    cur = _stats(2, team0_tower_damage=[0.0, 1200.0])  # env 0 reset, env 1 live

    out = compute_shaping(cur, prev, SHAPING_TEST_GAMMA)

    assert float(out[0]) == 0.0, "the reset env leaked a spike"
    assert float(out[1]) != 0.0, "the LIVE env was zeroed along with it"


def test_a_towers_alive_count_going_UP_is_a_reset():
    """`towers_alive` is the one counter that is monotone DOWNWARD, so for it a
    reset looks like an increase. Detecting only 'went down' would miss an
    episode that ended with towers already lost."""
    # Every OTHER counter is held equal, so `towers_alive` is the only witness
    # to the reset -- otherwise this passes off the damage detector and proves
    # nothing. Elixir is parked above the overflow threshold so there is a real
    # non-zero term for the guard to suppress.
    prev = _stats(team0_towers_alive=[1], team1_towers_alive=[2],
                  team0_elixir_current=[9.9])
    cur = _stats(team0_towers_alive=[3], team1_towers_alive=[3],
                 team0_elixir_current=[9.9])
    assert float(compute_shaping(cur, _stats(team0_elixir_current=[9.9]), SHAPING_TEST_GAMMA)[0]) < 0.0,         "control: with no reset this configuration must earn a real penalty"
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) == 0.0


def test_an_ordinary_step_is_untouched_by_the_guard():
    """The guard must not fire on live play -- otherwise it silently deletes
    the entire dense reward, which is the no-learning failure."""
    prev = _stats(team0_tower_damage=[1000.0])
    cur = _stats(team0_tower_damage=[1300.0])
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) > 0.0


def test_the_first_step_of_an_episode_is_not_mistaken_for_a_reset():
    """All-zero to all-zero is not a decrease, so a genuinely fresh episode
    still earns shaping on its first real step."""
    out = compute_shaping(_stats(team0_elixir_current=[9.9]), _stats(), SHAPING_TEST_GAMMA)
    assert np.isfinite(out).all()
    assert float(out[0]) < 0.0, "the overflow penalty should still apply"


def test_every_cumulative_engine_counter_is_covered_by_the_reset_detector():
    """`_MONOTONE_UP` is a second copy of a list `rl.engine_stats` also keeps,
    and this repo's rule is that a second copy is a scheduled defect unless
    something pins it. Pinned here: a counter added to the extractor and not to
    the detector is a reset this guard would silently miss.
    """
    from python_ai.rewards.shaping import _MONOTONE_DOWN, _MONOTONE_UP
    from python_ai.rl import engine_stats as es

    cumulative = set(es._INT_KEYS) | set(es._FLOAT_KEYS)
    missing = cumulative - set(_MONOTONE_UP)
    assert not missing, f"cumulative counters the reset detector cannot see: {missing}"

    # ...and the two directions must not overlap, or one would cancel the other.
    assert not (set(_MONOTONE_UP) & set(_MONOTONE_DOWN))


def test_the_reset_guard_survives_a_stats_dict_missing_optional_keys():
    """`team0_wincon_damage` is absent for a deck with no building-targeter,
    and `extract_engine_stats` omits keys the info dict never carried. The
    detector must degrade, not raise."""
    prev = _stats(team0_tower_damage=[3000.0])
    cur = _stats()
    for d in (prev, cur):
        d.pop("team0_wincon_damage")
    assert float(compute_shaping(cur, prev, SHAPING_TEST_GAMMA)[0]) == 0.0
