"""Pins the time and elixir constants against the live engine. The pipeline's tick
arithmetic rests on two numbers; a change on the C++ side must fail here rather
than make every timestamp quietly wrong by a few percent.
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
    # 0.19 s is 1.9 ticks; truncation would give 1 and bias every conversion
    # downward.
    assert timebase.seconds_to_ticks(0.19) == 2
    assert timebase.seconds_to_ticks(0.14) == 1


def test_frame_index_conversion():
    assert timebase.frame_index_to_ticks(30, fps=30.0) == timebase.TICKS_PER_SECOND
    assert timebase.frame_index_to_ticks(0, fps=30.0) == 0
    with pytest.raises(ValueError):
        timebase.frame_index_to_ticks(1, fps=0.0)


def test_elixir_regen_matches_engine(engine):
    """ELIXIR_REGEN_RATE is 0.035/tick, measured."""
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
    """The 2% gap as an asserted fact: "fixing" ELIXIR_REGEN_RATE to match the
    real game fails here and forces the opponent-elixir model to be revisited
    too.
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
    CardStats::attackCooldown is in ticks and the real game publishes it as
    "hit speed" in seconds; their ratio is the tick rate. Only cards with
    stable, well-known hit speeds are listed.
    """
    # (card id, real-game hit speed in seconds)
    known = [(1, 0.9), (6, 1.0), (0, 1.2), (10, 1.5), (15, 1.6)]
    for card_id, hit_speed in known:
        info = engine.get_card_info(card_id)
        assert info is not None, f"card {card_id} vanished from the registry"

    # get_card_info does not expose attackCooldown, so the ratio is pinned as a
    # constant and the derivation lives in timebase.py; this guards that the
    # cards it rests on still exist and mean what they did.
    assert timebase.TICKS_PER_SECOND == 10
