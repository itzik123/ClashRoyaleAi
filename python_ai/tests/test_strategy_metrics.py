"""The live strategy read-out, and the two definitions in it that are load-bearing.

ROI and Fwd are meant to be read as a PAIR. Raising placement entropy spreads
the marginal toward the middle of the legal region, which lifts the forward rate
whether or not the policy improved -- ROI is the half that noise pushes the
other way. That only works if ROI is defined the way the console claims.
"""
import numpy as np
import pytest

from python_ai.trainers.strategy_metrics import (
    FORWARD_ROW_Y, ScenarioMetrics, StrategyMetrics,
)

DECK = [15, 6, 25, 40, 24, 72, 33, 7]
BOARD_W = 18


def test_roi_is_a_ratio_of_SUMS_not_a_mean_of_per_episode_ratios():
    """An episode where a card went unplayed contributes 0/0, and averaging
    those moves the number for reasons that have nothing to do with how well the
    card was used.

    Constructed so the two definitions disagree: one big efficient episode and
    one tiny inefficient one.
    """
    m = StrategyMetrics(num_envs=1, deck=DECK)
    killed_big = np.zeros(8); killed_big[0] = 100.0
    spent_big = np.zeros(8); spent_big[0] = 50.0          # ratio 2.0
    killed_small = np.zeros(8); killed_small[0] = 1.0
    spent_small = np.zeros(8); spent_small[0] = 2.0       # ratio 0.5
    m.finish_episode(0, killed_big, spent_big, tower_damage=0, steps=1)
    m.finish_episode(0, killed_small, spent_small, tower_damage=0, steps=1)

    roi, _worst_i, _worst, _per = m.roi()
    assert roi == pytest.approx(101.0 / 52.0)             # ratio of sums
    assert roi != pytest.approx((2.0 + 0.5) / 2)          # NOT a mean of ratios


def test_an_episode_where_nothing_was_spent_is_excluded_entirely():
    """0/0 must not enter the window at all."""
    m = StrategyMetrics(num_envs=1, deck=DECK)
    m.finish_episode(0, np.zeros(8), np.zeros(8), tower_damage=0, steps=1)
    assert len(m.econ_spent) == 0
    assert np.isnan(m.roi()[0])


def test_the_worst_card_is_the_lowest_ROI_among_cards_ACTUALLY_played():
    """A card nobody played has no ROI. Treating its 0/0 as 0.0 would make the
    worst-card readout always name whatever the policy has abandoned -- which is
    a different (already-monitored) fact."""
    m = StrategyMetrics(num_envs=1, deck=DECK)
    killed = np.zeros(8); spent = np.zeros(8)
    killed[0], spent[0] = 10.0, 5.0        # ROI 2.0
    killed[3], spent[3] = 1.0, 5.0         # ROI 0.2  <- worst
    # slot 5 never played at all
    m.finish_episode(0, killed, spent, tower_damage=0, steps=1)
    roi, worst_i, worst, per_card = m.roi()
    assert worst_i == 3 and worst == pytest.approx(0.2)
    assert np.isnan(per_card[5]), "an unplayed card contributes no ROI"


def test_a_forward_placement_is_one_at_or_past_the_bridge_band():
    m = StrategyMetrics(num_envs=1, deck=DECK)
    back_cell = (FORWARD_ROW_Y - 1) * BOARD_W + 3
    fwd_cell = FORWARD_ROW_Y * BOARD_W + 3
    m.record_play(0, DECK[0], 5.0, back_cell, BOARD_W)
    m.record_play(0, DECK[1], 5.0, fwd_cell, BOARD_W)
    m.finish_episode(0, np.ones(8), np.ones(8), tower_damage=0, steps=10)
    assert m.summary()["forward_rate"] == pytest.approx(0.5)


def test_tower_damage_is_normalized_per_1000_TICKS_not_per_episode():
    """`AvgTicks` moved 24% in one hour after the placement-mask fix, and an
    unnormalized total would have read as more pressure when it was only longer
    matches."""
    m = StrategyMetrics(num_envs=1, deck=DECK)
    # 100 decisions = 1000 engine ticks, so 500 damage is a rate of 500.
    m.finish_episode(0, np.ones(8), np.ones(8), tower_damage=500, steps=100)
    assert m.summary()["tower_damage_rate"] == pytest.approx(500.0)
    # Twice as long a match with the same total is HALF the rate.
    m2 = StrategyMetrics(num_envs=1, deck=DECK)
    m2.finish_episode(0, np.ones(8), np.ones(8), tower_damage=500, steps=200)
    assert m2.summary()["tower_damage_rate"] == pytest.approx(250.0)


def test_distinct_cards_per_game_counts_identities_not_plays():
    """Cards/Game is the headline: 8.0 means the whole deck is in use, ~6.0 was
    the phase-1 failure mode (win condition and spell abandoned)."""
    m = StrategyMetrics(num_envs=1, deck=DECK)
    for _ in range(5):
        m.record_play(0, DECK[0], 4.0, 0, BOARD_W)
    m.record_play(0, DECK[1], 6.0, 0, BOARD_W)
    m.finish_episode(0, np.ones(8), np.ones(8), tower_damage=0, steps=10)
    s = m.summary()
    assert s["cards_per_game"] == 2
    assert s["plays_per_game"] == 6
    assert s["elixir_at_play"] == pytest.approx((4.0 * 5 + 6.0) / 6)


def test_a_play_with_an_unknown_card_still_counts_as_a_play():
    """A card id of -1 means the slot could not be resolved; the elixir and the
    placement are still real information."""
    m = StrategyMetrics(num_envs=1, deck=DECK)
    m.record_play(0, -1, 3.0, 0, BOARD_W)
    m.finish_episode(0, np.ones(8), np.ones(8), tower_damage=0, steps=1)
    s = m.summary()
    assert s["cards_per_game"] == 0 and s["plays_per_game"] == 1


def test_reset_env_drops_a_scenario_episode_without_recording_it():
    m = StrategyMetrics(num_envs=1, deck=DECK)
    m.record_play(0, DECK[0], 4.0, 0, BOARD_W)
    m.reset_env(0)
    m.finish_episode(0, np.ones(8), np.ones(8), tower_damage=0, steps=1)
    assert m.summary()["plays_per_game"] == 0


def test_empty_windows_report_nan():
    s = StrategyMetrics(num_envs=1, deck=DECK).summary()
    for key in ("cards_per_game", "plays_per_game", "elixir_at_play",
                "forward_rate", "tower_damage_rate", "roi"):
        assert np.isnan(s[key]), key


# ---------------------------------------------------------------- scenarios --
def test_only_DEFENSIVE_scenarios_reach_the_defence_success_rate():
    """`fireball_tower_value` spawns at the ENEMY tower, so "did not take a big
    hit" is true no matter what the agent does -- including nothing. Counting it
    pushed ScenDef toward 1.0 and would have masked a genuine collapse in the
    reflex the metric exists to watch."""
    sm = ScenarioMetrics()
    sm.record(is_defensive=True, raw_reward=-1.0)     # lost the defence
    sm.record(is_defensive=False, raw_reward=0.0)     # vacuously "survived"
    s = sm.summary()
    assert s["defensive_rate"] == 0.0 and s["defensive_n"] == 1
    assert s["other_rate"] == 1.0 and s["other_n"] == 1


def test_success_means_the_episode_did_not_end_in_a_loss():
    sm = ScenarioMetrics()
    sm.record(True, 1.0)        # win
    sm.record(True, 0.0)        # truncated out of the window, still alive
    sm.record(True, -1.0)       # lost
    assert sm.summary()["defensive_rate"] == pytest.approx(2 / 3)


def test_an_untouched_scenario_window_reports_nan_not_a_perfect_score():
    assert np.isnan(ScenarioMetrics().summary()["defensive_rate"])
