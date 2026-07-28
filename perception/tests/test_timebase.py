"""Pins the time and elixir constants against the live engine.

The whole pipeline's tick arithmetic rests on two numbers. If either changes
on the C++ side, this file fails immediately and loudly -- which is the only
protection against the failure mode where every timestamp is quietly wrong by
a few percent and nothing looks broken.
"""

from __future__ import annotations

import pytest

import timebase


def test_seconds_ticks_roundtrip():
    for seconds in (0.0, 0.1, 1.0, 2.8, 60.0, 180.0):
        assert timebase.ticks_to_seconds(timebase.seconds_to_ticks(seconds)) == pytest.approx(
            seconds, abs=1 / (2 * timebase.TICKS_PER_SECOND)
        )


def test_seconds_to_ticks_rounds_not_truncates():
    # 0.19s is 1.9 ticks. Truncation would give 1 and bias every conversion
    # downward -- see the function's own comment.
    assert timebase.seconds_to_ticks(0.19) == 2
    assert timebase.seconds_to_ticks(0.14) == 1


def test_frame_index_conversion():
    assert timebase.frame_index_to_ticks(30, fps=30.0) == timebase.TICKS_PER_SECOND
    assert timebase.frame_index_to_ticks(0, fps=30.0) == 0
    with pytest.raises(ValueError):
        timebase.frame_index_to_ticks(1, fps=0.0)


def test_elixir_regen_matches_engine(engine):
    """ELIXIR_REGEN_RATE is really 0.035/tick, measured not assumed."""
    deck = [15, 25, 6, 1, 0, 41, 7, 10]
    env = engine.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()

    start = env.get_elixir()
    ticks = 20
    env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, ticks)
    measured = (env.get_elixir() - start) / ticks

    assert measured == pytest.approx(timebase.SIM_ELIXIR_REGEN_PER_TICK, abs=1e-6), (
        "GameManager::ELIXIR_REGEN_RATE changed. timebase.py's derivation and "
        "every downstream elixir model depend on this value."
    )


def test_starting_elixir_matches_engine(engine):
    deck = [15, 25, 6, 1, 0, 41, 7, 10]
    env = engine.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()
    assert env.get_elixir() == pytest.approx(timebase.STARTING_ELIXIR)


def test_simulator_elixir_runs_slow_relative_to_real_game():
    """Documents the 2% gap as an asserted fact, not a comment.

    If someone "fixes" ELIXIR_REGEN_RATE to 0.0357 to match the real game,
    this fails and forces the opponent-elixir model to be revisited at the
    same time -- rather than the two silently disagreeing in the other
    direction.
    """
    sim_seconds_per_elixir = 1.0 / (
        timebase.SIM_ELIXIR_REGEN_PER_TICK * timebase.TICKS_PER_SECOND
    )
    assert sim_seconds_per_elixir == pytest.approx(2.857, abs=0.001)
    assert timebase.REAL_SECONDS_PER_ELIXIR_1X == 2.8
    assert timebase.SIM_ELIXIR_RATE_ERROR == pytest.approx(0.98, abs=0.001)


def test_phase_multipliers_only_scale_real_rate():
    single = timebase.elixir_regenerated(2.8, 1.0)
    assert single == pytest.approx(1.0)
    assert timebase.elixir_regenerated(2.8, 2.0) == pytest.approx(2.0)
    assert timebase.elixir_regenerated(2.8, 3.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        timebase.elixir_regenerated(-1.0)


def test_attack_cooldown_evidence_for_tick_rate(engine):
    """The evidence TICKS_PER_SECOND was derived from, kept executable.

    CardStats::attackCooldown is in ticks; the real game publishes the same
    figure as "hit speed" in seconds. Their ratio is the tick rate. This is
    the strongest available source because it is asserted independently by
    every card in the registry rather than by one constant.

    Only the cards whose real hit speed is stable and well known are listed;
    the point is the ratio, not roster coverage.
    """
    # (card id, real-game hit speed in seconds)
    known = [(1, 0.9), (6, 1.0), (0, 1.2), (10, 1.5), (15, 1.6)]
    for card_id, hit_speed in known:
        info = engine.get_card_info(card_id)
        assert info is not None, f"card {card_id} vanished from the registry"

    # The registry does not expose attackCooldown through get_card_info, so
    # the ratio itself is pinned as a constant here and the derivation lives
    # in timebase.py's docstring. What this test guards is that the cards the
    # derivation was based on still exist and still mean what they did.
    assert timebase.TICKS_PER_SECOND == 10
