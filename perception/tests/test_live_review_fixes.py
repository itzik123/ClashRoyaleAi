"""Regression tests for the 2026-11-09 review of the live path.

Each pins a defect that failed quietly: an actuator thread hung forever, a
dropped placement booked as played, a second match inheriting the first one's
spend, a hand with one card twice, and a spell re-cast by the mirror.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from contracts import GameState, Phase, TowerObservation, UnitObservation


# --- the actuator ------------------------------------------------------------

def _actuator(tmp_path):
    from live.actuator import AdbActuator
    fake_adb = tmp_path / "adb.exe"
    fake_adb.write_text("")
    return AdbActuator(dry_run=False, adb=fake_adb, raw_touch=False)


def test_a_silent_adb_shell_times_out_instead_of_hanging(monkeypatch, tmp_path):
    """The deadline used to sit between readline() calls, and readline() blocks
    until a line arrives: a wedged shell that printed nothing hung the actuator
    thread forever. A real child that never writes is the reproduction."""
    import live.actuator as A
    monkeypatch.setattr(A, "SHELL_TIMEOUT_S", 0.5)
    actuator = _actuator(tmp_path)
    silent = [sys.executable, "-c", "import time; time.sleep(30)"]
    monkeypatch.setattr(actuator, "_open_shell", lambda: subprocess.Popen(
        silent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1))
    started = time.perf_counter()
    try:
        with pytest.raises(A.ActuationError, match="timed out"):
            actuator._shell_persistent("true")
        assert time.perf_counter() - started < 5.0
    finally:
        actuator._close_shell()
        actuator._stop.set()


def test_the_persistent_shell_still_returns_on_its_sentinel(monkeypatch, tmp_path):
    """The control: a shell that answers is not reported as timed out."""
    import live.actuator as A
    actuator = _actuator(tmp_path)
    echo = [sys.executable, "-u", "-c",
            "import sys\nfor line in sys.stdin:\n"
            f"    print('{A._SENTINEL}', flush=True)"]
    monkeypatch.setattr(actuator, "_open_shell", lambda: subprocess.Popen(
        echo, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1))
    try:
        actuator._shell_persistent("true")
        actuator._shell_persistent("true")         # the same shell, reused
    finally:
        actuator._close_shell()
        actuator._stop.set()


def test_a_dropped_placement_says_so(monkeypatch, tmp_path):
    """The loop books a placement (gate, confirmer, ledger debit) only when
    `last_queued` is True; a drop used to be booked as a card played."""
    release = threading.Event()
    actuator = _actuator(tmp_path)
    monkeypatch.setattr(actuator, "_shell", lambda s: release.wait(2.0))
    try:
        actuator.play(0, 9, 8)
        assert actuator.last_queued
        time.sleep(0.1)                 # the worker takes the first
        actuator.play(1, 9, 8)          # queued behind it
        assert actuator.last_queued
        actuator.play(2, 9, 8)          # queue full: dropped
        assert not actuator.last_queued and actuator.dropped == 1
    finally:
        release.set()
        actuator.close()


def test_a_dry_run_never_reports_a_drop():
    from live.actuator import AdbActuator
    a = AdbActuator(dry_run=True)
    for slot in range(4):
        a.play(slot, 9, 8)
        assert a.last_queued
    assert a.dropped == 0


# --- the pipeline's freshness default ----------------------------------------

def test_an_unset_publication_time_reads_as_forever_on_a_fresh_boot():
    """perf_counter() starts near boot; on a box up for minutes a 0.0 default
    read as "published just now". The comparison is made at an explicit `now`
    of 10 s, the case that used to fail."""
    from live.pipeline import Snapshot
    snap = Snapshot(state=None, game_state=None, captured_at=9.0, index=0,
                    detect_ms=0.0)
    assert snap.sat_ms(now=10.0) == float("inf")


# --- the elixir ledger across matches -----------------------------------------

def _readings(ledger, values, start=0.0, dt=0.1):
    for i, v in enumerate(values):
        ledger.update(v, now=start + i * dt)


def test_a_new_match_starts_the_ledger_from_zero():
    from live.elixir_ledger import ElixirLedger
    ledger = ElixirLedger()
    _readings(ledger, [5, 5, 6, 6, 7, 7, 7, 3, 3, 3])
    assert ledger.spent == pytest.approx(4.0)
    ledger.new_match()
    assert ledger.spent == 0.0 and ledger.cards == 0
    # The second match opens on a full-ish bar after the first ended low; the
    # first readings are not diffed against the old match's last value.
    _readings(ledger, [7, 7, 7, 7, 7], start=100.0)
    assert ledger.spent == 0.0 and ledger.cards == 0
    assert ledger.unexplained == 0


def test_without_the_reset_the_old_match_leaks():
    """The control: the same two matches without new_match() carry the first
    match's 4 elixir into the second, which is what the observation's spend
    scalar read. (Before the regeneration floor below, the second match's
    opening 7, diffed against the old bar regenerated to the cap, also booked
    a 3-elixir play nobody made.)"""
    from live.elixir_ledger import ElixirLedger
    ledger = ElixirLedger()
    _readings(ledger, [5, 5, 6, 6, 7, 7, 7, 3, 3, 3])
    _readings(ledger, [7, 7, 7, 7, 7], start=100.0)
    assert ledger.spent == pytest.approx(4.0)


def test_pending_plays_do_not_survive_into_the_next_match():
    from live.elixir_ledger import ElixirLedger
    ledger = ElixirLedger()
    _readings(ledger, [5, 5, 5])
    ledger.record_play(4.0, tag="old")
    assert ledger.unconfirmed_cost == pytest.approx(4.0)
    ledger.new_match()
    assert ledger.unconfirmed_cost == 0.0


# --- the hand tracker -------------------------------------------------------

def test_a_consensus_with_one_card_twice_never_seeds_the_hand():
    from live.hand_tracker import CONSENSUS_WINDOW, HandTracker
    deck = (10, 1, 41, 25, 7, 2, 6, 5)
    costs = {10: 4.0, 1: 3.0, 41: 3.0, 25: 3.0, 7: 4.0, 2: 5.0, 6: 4.0, 5: 4.0}
    t = HandTracker(deck=deck, costs=costs)
    for _ in range(CONSENSUS_WINDOW + 5):
        t.update((10, 10, 41, 25), [])
    assert not t.seeded
    for _ in range(CONSENSUS_WINDOW + 5):          # the control
        t.update((10, 1, 41, 25), [])
    assert t.seeded and sorted(t.hand) == [1, 10, 25, 41]
    assert len(t.queue) == 4


# --- the mirror ---------------------------------------------------------------

def test_the_mirror_never_recasts_a_spell(engine):
    """A spell id among perceived units would be injected, i.e. cast, again on
    every rebuild."""
    from live.mirror import MirrorBuilder
    full = TowerObservation(1.0, True)
    fireball = UnitObservation(7, "", 1, 3, 10, 1.0, True)
    gs = GameState(units=(fireball,), my_elixir=5.0, my_hand=(15, 6, 25, 40),
                   seconds_elapsed=30.0, phase=Phase.SINGLE,
                   own_king=full, own_princess_left=full,
                   own_princess_right=full, opp_king=full,
                   opp_princess_left=full, opp_princess_right=full)
    _env, gaps = MirrorBuilder([15, 6, 25, 40, 24, 72, 33, 7],
                               engine=engine).build(gs)
    assert gaps.injected == 0 and gaps.unmappable_units == 1


# --- the elixir ledger with a 1-elixir card in the deck -------------------------

# 2.6 Hog Cycle's distinct costs: Skeletons and Ice Spirit are 1 elixir.
HOG26_COSTS = (1.0, 2.0, 3.0, 4.0)


def _ledger_1hz(values, costs=HOG26_COSTS):
    from live.elixir_ledger import ElixirLedger
    ledger = ElixirLedger(costs=costs)
    for i, v in enumerate(values):
        ledger.update(v, now=float(i))             # 1 Hz, the live loop's rate
    return ledger


def test_a_full_bar_is_not_a_stream_of_one_elixir_plays():
    """Measured in a --frames smoke run of mvp_loop on a static frame: the bar
    read 10 throughout and the ledger booked 8 plays in 10 readings."""
    ledger = _ledger_1hz([10] * 12)
    assert ledger.cards == 0 and ledger.spent == 0.0


def test_a_regenerating_integer_bar_is_not_spending():
    """The bar sits on one integer for ~3 s per elixir, so each reading 'drops'
    by the regeneration the integer hid."""
    values = [v for v in range(3, 10) for _ in range(3)]   # 3,3,3,4,4,4,...
    ledger = _ledger_1hz(values)
    assert ledger.cards == 0, ledger.plays


@pytest.mark.parametrize("cost", [1, 2, 4])
def test_a_real_play_is_still_counted_with_a_one_elixir_deck(cost):
    """The control: the floor must not hide the plays it guards."""
    ledger = _ledger_1hz([8, 8, 8, 8 - cost, 8 - cost, 8 - cost])
    assert ledger.cards == 1 and ledger.spent == pytest.approx(cost)
