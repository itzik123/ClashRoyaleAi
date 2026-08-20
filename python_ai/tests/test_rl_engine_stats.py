"""extract_engine_stats: every default must contribute EXACTLY zero reward.

This dictionary was written out inline three times (two trainers plus the
exploiter) and the copies had already drifted -- the exploiter's was missing
`team0_wincon_damage`, so it alone trained without the win-condition term.

The property that makes the defaults safe is not "reasonable value"; it is that
a MISSING key produces a zero contribution to `compute_shaping`, never a
spurious spike. That is checked here against the real shaping function.
"""
import numpy as np
import pytest

from python_ai.rewards.shaping import compute_shaping
from python_ai.rl.engine_stats import extract_engine_stats, opponent_elixir_target

N = 3


def test_an_entirely_empty_infos_dict_yields_a_complete_stats_dict():
    """The rare-but-real step where EVERY env auto-resets at once: gymnasium
    omits the key entirely rather than supplying defaults."""
    stats = extract_engine_stats({}, N)
    for key in ("team0_troop_damage", "team1_troop_damage",
                "team0_building_damage", "team1_building_damage",
                "team0_tower_damage", "team1_tower_damage",
                "team0_elixir_spent", "team1_elixir_spent",
                "team0_elixir_current", "team0_towers_alive",
                "team1_towers_alive", "enemy_tower_hp", "fireball_in_hand",
                "fireball_value_killed", "fireball_elixir_spent",
                "team0_wincon_damage"):
        assert key in stats, f"{key} missing -- compute_shaping would KeyError"


def test_the_win_condition_key_is_present_which_the_exploiter_copy_was_missing():
    assert "team0_wincon_damage" in extract_engine_stats({}, N)


def test_every_array_has_one_entry_per_env():
    stats = extract_engine_stats({}, N)
    for key, value in stats.items():
        arr = np.asarray(value)
        assert arr.shape[0] == N, f"{key} has shape {arr.shape}"
    assert np.asarray(stats["enemy_tower_hp"]).shape == (N, 3)


def test_towers_alive_defaults_to_a_FULL_set_not_zero():
    """Defaulting to 0 would read as a three-crown swing on the one step where
    the key is absent -- a phantom +1.8 of reward from nothing happening."""
    stats = extract_engine_stats({}, N)
    assert list(stats["team0_towers_alive"]) == [3] * N
    assert list(stats["team1_towers_alive"]) == [3] * N


def test_two_all_default_steps_contribute_nothing_beyond_the_PBRS_residual():
    """The end-to-end version of the claim, stated exactly.

    Every DELTA-driven term must be zero -- that is what the defaults buy. The
    POTENTIAL-based terms are a different matter and the distinction is worth
    being precise about: for an unchanged state they emit
    `gamma*Phi(s) - Phi(s) = (gamma - 1) * Phi(s)`, which is not zero wherever
    Phi is not. Here the default elixir reading of 0 is a genuinely BROKE state,
    so the solvency potential sits at its floor of -W_SOLVENCY and the residual
    is +0.001.

    That is correct, not a leak: PBRS telescopes to `gamma^T*Phi(s_T) - Phi(s_0)`
    over an episode, so the residual cannot be farmed. The test pins the exact
    value so that a term which is NOT potential-based can never hide inside it.
    """
    from python_ai.rewards.elixir_shaping import solvency_potential
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    gamma = 0.99
    shaping = compute_shaping(cur, prev, gamma=gamma)
    expected = (gamma - 1.0) * solvency_potential(cur["team0_elixir_current"])
    assert np.allclose(shaping, expected, atol=1e-7), (shaping, expected)


def test_a_solvent_all_default_pair_produces_exactly_zero_shaping():
    """With the one non-zero potential neutralized, the defaults really are
    inert -- which is the property the missing-key handling depends on.

    5.0 elixir specifically: at or above SOLVENCY_RESERVE (4.0) so the solvency
    potential is 0, and below ELIXIR_OVERFLOW_THRESHOLD (9.0) so the per-step
    overflow penalty is 0. A full bar would fail this test correctly -- capping
    out is meant to cost something on every step it is true.
    """
    from python_ai.rewards.weights import (
        ELIXIR_OVERFLOW_THRESHOLD, SPELL_SOLVENCY_RESERVE,
    )
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    solvent = np.full(N, 5.0, dtype=np.float32)
    assert SPELL_SOLVENCY_RESERVE <= 5.0 <= ELIXIR_OVERFLOW_THRESHOLD
    prev["team0_elixir_current"] = solvent
    cur["team0_elixir_current"] = solvent.copy()
    assert np.allclose(compute_shaping(cur, prev, gamma=0.99), 0.0)


def test_a_full_elixir_bar_costs_something_on_every_step_it_is_true():
    """The overflow term is continuous, not one-time: a real player never
    intentionally caps out, because it wastes ongoing regeneration."""
    from python_ai.rewards.weights import W_ELIXIR_OVERFLOW
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    full = np.full(N, 10.0, dtype=np.float32)
    prev["team0_elixir_current"] = full
    cur["team0_elixir_current"] = full.copy()
    assert np.allclose(compute_shaping(cur, prev, gamma=0.99),
                       -W_ELIXIR_OVERFLOW, atol=1e-6)


def test_a_missing_key_never_fabricates_a_lethal_spell_opportunity():
    """`enemy_tower_hp` and `fireball_in_hand` default to the NO-OPPORTUNITY
    state, so an absent key can only ever zero the potential."""
    from python_ai.rewards.shaping import lethal_spell_potential
    stats = extract_engine_stats({}, N)
    assert np.allclose(lethal_spell_potential(stats), 0.0)


def test_supplied_values_pass_through_unchanged():
    infos = {
        "team0_troop_damage": np.array([10, 20, 30]),
        "elixir": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "team1_towers_alive": np.array([2, 3, 1]),
    }
    stats = extract_engine_stats(infos, N)
    assert list(stats["team0_troop_damage"]) == [10, 20, 30]
    assert list(stats["team0_elixir_current"]) == pytest.approx([1.5, 2.5, 3.5])
    assert list(stats["team1_towers_alive"]) == [2, 3, 1]


def test_the_elixir_key_is_read_from_infos_elixir_not_from_a_team_key():
    """`gym_wrapper` publishes the instantaneous reading under "elixir"; a
    rename here would silently zero the overflow penalty."""
    stats = extract_engine_stats({"elixir": np.array([9.5, 9.5, 9.5],
                                                    dtype=np.float32)}, N)
    assert list(stats["team0_elixir_current"]) == pytest.approx([9.5] * 3)


def test_the_aux_target_defaults_to_zero_and_is_float():
    target = opponent_elixir_target({}, N)
    assert target.shape == (N,)
    assert target.dtype == np.float32
    assert np.allclose(target, 0.0)
