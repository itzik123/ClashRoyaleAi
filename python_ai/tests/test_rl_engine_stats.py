"""extract_engine_stats: every default must contribute exactly zero reward.

A missing key must yield a zero contribution to `compute_shaping`, never a
spurious spike; checked here against the real shaping function.
"""
import numpy as np
import pytest

from python_ai.rewards.shaping import compute_shaping
from python_ai.rl.engine_stats import (extract_engine_stats,
                                       next_card_labels,
                                       opponent_played_card)

N = 3


def test_an_entirely_empty_infos_dict_yields_a_complete_stats_dict():
    """The step where every env auto-resets at once: gymnasium omits the key
    entirely.
    """
    stats = extract_engine_stats({}, N)
    for key in ("team0_troop_damage", "team1_troop_damage",
                "team0_building_damage", "team1_building_damage",
                "team0_tower_damage", "team1_tower_damage",
                "team0_elixir_spent", "team1_elixir_spent",
                "team0_elixir_current", "team0_towers_alive",
                "team1_towers_alive", "enemy_tower_hp", "spell_in_hand",
                "spell_value_killed", "spell_elixir_spent", "spell_damage",
                "spell_cost",
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
    """Defaulting to 0 would read as a three-crown swing on the step the key is
    absent.
    """
    stats = extract_engine_stats({}, N)
    assert list(stats["team0_towers_alive"]) == [3] * N
    assert list(stats["team1_towers_alive"]) == [3] * N


def test_two_all_default_steps_contribute_nothing_beyond_the_PBRS_residual():
    """Every delta-driven term is zero. A potential-based term emits `(gamma - 1)
    * Phi(s)` for an unchanged state, non-zero wherever Phi is: at the default
    elixir of 0 the solvency potential sits at -W_SOLVENCY. That telescopes
    over an episode and cannot be farmed; the exact value is pinned so a
    non-potential term cannot hide inside it.
    """
    from python_ai.rewards.elixir_shaping import solvency_potential
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    gamma = 0.99
    shaping = compute_shaping(cur, prev, gamma=gamma)
    expected = (gamma - 1.0) * solvency_potential(cur["team0_elixir_current"])
    assert np.allclose(shaping, expected, atol=1e-7), (shaping, expected)


def test_a_solvent_all_default_pair_produces_exactly_zero_shaping():
    """With the non-zero potential neutralised, the defaults are inert. 5.0 elixir
    is at or above SOLVENCY_RESERVE (potential 0) and below the overflow
    threshold (no penalty).
    """
    # SOLVENCY_RESERVE is the solvency potential's reserve, which the docstring
    # means.
    from python_ai.rewards.elixir_shaping import SOLVENCY_RESERVE
    from python_ai.rewards.weights import ELIXIR_OVERFLOW_THRESHOLD
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    solvent = np.full(N, 5.0, dtype=np.float32)
    assert SOLVENCY_RESERVE <= 5.0 <= ELIXIR_OVERFLOW_THRESHOLD
    prev["team0_elixir_current"] = solvent
    cur["team0_elixir_current"] = solvent.copy()
    assert np.allclose(compute_shaping(cur, prev, gamma=0.99), 0.0)


def test_a_full_elixir_bar_costs_something_on_every_step_it_is_true():
    """The overflow term is per step, not one-time: capping out wastes ongoing
    regeneration.
    """
    from python_ai.rewards.weights import W_ELIXIR_OVERFLOW
    prev = extract_engine_stats({}, N)
    cur = extract_engine_stats({}, N)
    full = np.full(N, 10.0, dtype=np.float32)
    prev["team0_elixir_current"] = full
    cur["team0_elixir_current"] = full.copy()
    assert np.allclose(compute_shaping(cur, prev, gamma=0.99),
                       -W_ELIXIR_OVERFLOW, atol=1e-6)


def test_a_missing_key_never_fabricates_a_lethal_spell_opportunity():
    """`enemy_tower_hp` and the `spell_*` keys default to the no-opportunity
    state, so an absent key can only zero the potential.
    """
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
    """`gym_wrapper` publishes the reading under "elixir"; a rename would silently
    zero the overflow penalty.
    """
    stats = extract_engine_stats({"elixir": np.array([9.5, 9.5, 9.5],
                                                    dtype=np.float32)}, N)
    assert list(stats["team0_elixir_current"]) == pytest.approx([9.5] * 3)


def test_the_aux_target_defaults_to_no_play_and_is_an_integer_card_id():
    """-1 ("played nothing"), not 0, which is a real card id: a sentinel
    `next_card_labels` drops, not a class the head would be trained toward on
    every all-envs-auto-reset step.
    """
    target = opponent_played_card({}, N)
    assert target.shape == (N,)
    assert target.dtype == np.int64
    assert np.all(target == -1)
