"""Tests for live/action_gate.py, the last thing between the agent and a real
match: what it must refuse, what it must let through, and what happens when a
tap fails partway.
"""
from __future__ import annotations

from live.action_gate import MAX_STALENESS_MS, ActionGate, GateDecision


def test_a_fresh_new_board_is_allowed():
    gate = ActionGate()
    assert gate.check(board_index=1, age_ms=100.0)


def test_the_same_board_cannot_be_acted_on_twice():
    """The measured failure: the policy repeating itself as the board sat
    unchanged, several Giants for one intent under --act.
    """
    gate = ActionGate()
    assert gate.check(7, 100.0).allowed
    gate.record(7)
    decision = gate.check(7, 100.0)
    assert not decision.allowed
    assert decision.reason == "duplicate"


def test_the_next_board_is_allowed_again():
    gate = ActionGate()
    gate.record(7)
    assert gate.check(8, 100.0).allowed


def test_a_stale_board_is_refused_even_though_it_is_new():
    """Freshness and novelty differ: a board nobody acted on is still wrong to act
    on if the match has moved past it.
    """
    gate = ActionGate()
    decision = gate.check(1, MAX_STALENESS_MS + 1)
    assert not decision.allowed
    assert decision.reason == "stale"


def test_duplicate_outranks_stale():
    """Both can hold at once; `duplicate` is the more specific fact, while `stale`
    would suggest the producer fell behind.
    """
    gate = ActionGate()
    gate.record(3)
    assert gate.check(3, 10_000.0).reason == "duplicate"


def test_staleness_can_be_switched_off_without_losing_idempotency():
    gate = ActionGate(enforce_staleness=False)
    assert gate.check(1, 60_000.0).allowed
    gate.record(1)
    assert gate.check(1, 0.0).reason == "duplicate"


def test_a_failed_tap_does_not_burn_the_board():
    """check() and record() are separate so a tap that raises leaves the board
    available; recording on intent would turn one adb failure into a placement
    skipped for good.
    """
    gate = ActionGate()
    assert gate.check(5, 100.0).allowed
    # ... actuator raises here, so record() is never reached ...
    assert gate.check(5, 100.0).allowed, "a failed tap must be retryable"


def test_counts_separate_the_two_refusals():
    """A run full of `duplicate` means the producer is slower than the loop; full
    of `stale`, that it stopped keeping up. One counter could not tell them
    apart.
    """
    gate = ActionGate()
    gate.record(1)
    for _ in range(3):
        gate.refuse(gate.check(1, 10.0).reason)
    for _ in range(2):
        gate.refuse(gate.check(9, 99_999.0).reason)
    assert gate.refused == {"duplicate": 3, "stale": 2}
    assert "3 duplicate" in gate.summary()
    assert "2 stale" in gate.summary()


def test_summary_says_so_when_nothing_was_blocked():
    gate = ActionGate()
    gate.record(1)
    assert "none blocked" in gate.summary()


def test_decision_is_truthy_so_callers_can_branch_on_it_directly():
    assert bool(GateDecision(True, "ok")) is True
    assert bool(GateDecision(False, "duplicate")) is False


def test_one_action_per_board_against_a_real_slow_producer():
    """The invariant end to end under the condition that broke it, a producer
    slower than the decision loop: every tap lands on a distinct board. Uses
    the real PerceptionWorker, since the property is about how it republishes;
    a stub returning a fresh index each call would prove nothing.
    """
    import time

    from live.pipeline import PerceptionWorker

    class SlowSource:
        def __init__(self):
            self.n = 0

        def read_new(self, timeout_s=1.0):
            self.n += 1
            time.sleep(0.15)                # producer ~6 Hz ...
            return type("F", (), {"index": self.n, "wall_time_ms": 0.0,
                                  "image": None})()

        def close(self):
            pass

    worker = PerceptionWorker(SlowSource(), None, lambda f: ("state", "gs"))
    worker.start()
    gate = ActionGate(enforce_staleness=False)
    acted_on = []
    try:
        deadline = time.perf_counter() + 1.2
        while time.perf_counter() < deadline:
            snap = worker.latest()
            if snap is not None:            # ... consumer polls ~200 Hz
                verdict = gate.check(snap.index, snap.age_ms())
                if verdict:
                    acted_on.append(snap.index)
                    gate.record(snap.index)
                else:
                    gate.refuse(verdict.reason)
            time.sleep(0.005)
    finally:
        worker.stop()

    assert len(acted_on) == len(set(acted_on)), "acted twice on one board"
    assert acted_on, "gate blocked everything"
    # The consumer polled far faster than the producer published, so the gate
    # did real work.
    assert gate.refused.get("duplicate", 0) > len(acted_on)
