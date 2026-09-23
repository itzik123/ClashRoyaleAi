"""The match gate: it must not start early, and must not end on a blip."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.match_state import MatchState  # noqa: E402


def feed(state: MatchState, screens):
    return [state.update(s) for s in screens]


def test_starts_closed():
    # The loop may start on the lobby; a gate defaulting open would let one
    # frame of garbage through.
    assert MatchState().in_match is False


def test_requires_enter_hold_consecutive_frames():
    state = MatchState(enter_hold=2, exit_hold=6)
    assert state.update("in_game") is False      # 1 of 2
    assert state.update("in_game") is True       # 2 of 2


def test_a_single_in_game_blip_in_the_lobby_does_not_open_the_gate():
    state = MatchState(enter_hold=2, exit_hold=6)
    feed(state, ["lobby", "in_game", "lobby", "in_game", "lobby"])
    assert state.in_match is False


def test_a_dropped_frame_mid_match_does_not_end_the_match():
    # The expensive failure: on_match_end resets the elixir ledger, so ending
    # early is unrecoverable.
    state = MatchState(enter_hold=2, exit_hold=6)
    feed(state, ["in_game"] * 3)
    assert state.in_match is True
    feed(state, ["unknown", "in_game", "unknown", "unknown", "in_game"])
    assert state.in_match is True


def test_sustained_exit_closes_the_gate():
    state = MatchState(enter_hold=2, exit_hold=6)
    feed(state, ["in_game"] * 3)
    feed(state, ["lobby"] * 6)
    assert state.in_match is False


def test_disagreement_must_be_consecutive_not_cumulative():
    state = MatchState(enter_hold=2, exit_hold=3)
    feed(state, ["in_game"] * 3)
    # Two lobby frames, agreement, then two more: five disagreements, never
    # three in a row.
    feed(state, ["lobby", "lobby", "in_game", "lobby", "lobby"])
    assert state.in_match is True


def test_transitions_fire_once_each():
    state = MatchState(enter_hold=2, exit_hold=3)
    seen = []
    state.on_change(seen.append)
    feed(state, ["in_game"] * 5)
    assert seen == [True]
    feed(state, ["lobby"] * 5)
    assert seen == [True, False]


def test_counts_two_separate_matches():
    state = MatchState(enter_hold=2, exit_hold=3)
    feed(state, ["in_game"] * 4 + ["lobby"] * 4 + ["in_game"] * 4)
    assert state.matches_seen == 2


def test_kept_fraction_reports_what_was_skipped():
    state = MatchState(enter_hold=1, exit_hold=1)
    feed(state, ["lobby"] * 5 + ["in_game"] * 5)
    assert state.frames_skipped == 5
    assert state.frames_in_match == 5
    assert state.kept_fraction == 0.5
