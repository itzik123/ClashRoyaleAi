"""EpisodeMetrics: what Win_Rate_100 counts.

A scenario episode must reset its accumulators without contributing an outcome:
an injected threat is a handicap, and averaging it in would make the win rate a
mix of two games.
"""
import numpy as np
import pytest

from python_ai.rl.episode_metrics import DRAW, LOSS, WIN, EpisodeMetrics, outcome_of


def test_outcome_thresholds_match_the_engines_reward_convention():
    """+/-1 for a king kill, ~0 for a timeout; the 0.5 thresholds separate a draw
    from a decisive result.
    """
    assert outcome_of(1.0) == WIN
    assert outcome_of(0.6) == WIN
    assert outcome_of(-1.0) == LOSS
    assert outcome_of(-0.6) == LOSS
    assert outcome_of(0.0) == DRAW
    assert outcome_of(0.4) == DRAW
    assert outcome_of(-0.4) == DRAW


def test_accumulate_then_finish_records_one_episode():
    m = EpisodeMetrics(num_envs=2)
    m.accumulate(np.array([0.5, 0.0]), np.array([0.2, 0.0]))
    m.accumulate(np.array([0.5, 0.0]), np.array([0.3, 0.0]))
    assert m.ep_steps[0] == 2
    assert m.finish_episode(0, raw_reward=1.0, ally_hp_end=0.9,
                            enemy_hp_end=0.1) == WIN
    s = m.summary()
    assert s["n"] == 1 and s["wins"] == 1
    assert s["avg_reward"] == pytest.approx(1.0)
    assert s["avg_shaping"] == pytest.approx(0.5)
    assert s["avg_length"] == pytest.approx(2.0)


def test_finishing_resets_only_that_envs_accumulators():
    m = EpisodeMetrics(num_envs=3)
    m.accumulate(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    m.finish_episode(1, 1.0, 0.0, 0.0)
    assert m.ep_reward[0] == 1.0 and m.ep_reward[2] == 3.0
    assert m.ep_reward[1] == 0.0 and m.ep_steps[1] == 0


def test_reset_env_clears_without_recording_an_outcome():
    """What pipeline 2 needs for scenario episodes."""
    m = EpisodeMetrics(num_envs=1)
    m.accumulate(np.array([5.0]), np.array([1.0]))
    m.reset_env(0)
    assert m.summary()["n"] == 0
    assert m.ep_reward[0] == 0.0 and m.ep_steps[0] == 0


def test_the_decisive_rate_excludes_draws_and_the_raw_rate_does_not():
    """40 win / 13 loss / 47 draw is 75% decisive but a 40% win rate. Both are
    reported; the curriculum gate uses the raw one.
    """
    m = EpisodeMetrics(num_envs=1)
    for raw in [1.0] * 40 + [-1.0] * 13 + [0.0] * 47:
        m.finish_episode(0, raw, 0.0, 0.0)
    s = m.summary()
    assert s["win_rate"] == pytest.approx(0.40)
    assert s["decisive_win_rate"] == pytest.approx(40 / 53)
    assert s["decided"] == 53


def test_an_all_draw_window_reports_an_UNDEFINED_decisive_rate_not_a_crash():
    """An all-draw window has no decided games, so its decisive rate is NaN, not
    0.00, which would read as "loses every decided game" (a different diagnosis
    from the all-draw timeout pathology).
    """
    m = EpisodeMetrics(num_envs=1)
    for _ in range(5):
        m.finish_episode(0, 0.0, 0.0, 0.0)
    assert np.isnan(m.summary()["decisive_win_rate"])


def test_every_undefined_rate_uses_ONE_convention():
    """Every undefined rate uses one convention (NaN), so a caller can branch on
    it.
    """
    s = EpisodeMetrics(num_envs=1).summary()
    for key in ("win_rate", "loss_rate", "draw_rate",
                "decisive_win_rate", "decisive_win_rate_long"):
        assert isinstance(s[key], float) and np.isnan(s[key]), (key, s[key])


def test_windows_have_the_documented_sizes():
    """100 is the curriculum gate's resolution (+/-0.1 band); 500 shows a trend
    through that band; 50 suffices for the smooth continuous quantities.
    """
    m = EpisodeMetrics(num_envs=1)
    assert m.outcomes.maxlen == 100
    assert m.outcomes_long.maxlen == 500
    for window in (m.rewards, m.shaping, m.lengths,
                   m.ally_building_hp_end, m.enemy_building_hp_end):
        assert window.maxlen == 50


def test_the_long_window_keeps_reporting_after_the_short_one_has_rolled():
    m = EpisodeMetrics(num_envs=1)
    for _ in range(150):
        m.finish_episode(0, 1.0, 0.0, 0.0)
    s = m.summary()
    assert s["n"] == 100
    assert s["decisive_win_rate_long"] == pytest.approx(1.0)


def test_empty_windows_report_nan_not_zero():
    """"No data yet" must not read as "measured zero"."""
    s = EpisodeMetrics(num_envs=1).summary()
    assert np.isnan(s["avg_reward"])
    assert np.isnan(s["win_rate"])
    assert np.isnan(s["decisive_win_rate_long"])
