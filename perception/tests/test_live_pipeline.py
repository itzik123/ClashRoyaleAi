"""Tests for live/pipeline.py, against a synthetic source. A fake source and a
fake perceive() simulate a slow detector deterministically.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from live.pipeline import PerceptionWorker, Snapshot, Stages


@dataclass
class FakeFrame:
    index: int
    wall_time_ms: float = 0.0
    image: object = None


class FakeSource:
    """Hands out frames on demand, counting how many were asked for."""

    def __init__(self, delay: float = 0.0, limit: int | None = None):
        self.delay = delay
        self.limit = limit
        self.served = 0
        self._lock = threading.Lock()

    def read_new(self, timeout_s: float = 1.0):
        with self._lock:
            if self.limit is not None and self.served >= self.limit:
                time.sleep(0.01)
                return None
            self.served += 1
            index = self.served
        if self.delay:
            time.sleep(self.delay)
        return FakeFrame(index=index)

    def close(self):
        pass


def perceive_ok(frame):
    return f"state{frame.index}", f"gs{frame.index}"


def wait_for(predicate, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_publishes_a_snapshot():
    w = PerceptionWorker(FakeSource(), None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.latest() is not None)
        assert isinstance(w.latest(), Snapshot)
    finally:
        w.stop()


def test_only_the_newest_board_is_kept():
    """No queue: an old board is wrong, not merely less useful, so dropping is
    correct.
    """
    source = FakeSource()
    w = PerceptionWorker(source, None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.frames > 5)
        first = w.latest().index
        assert wait_for(lambda: w.latest().index > first + 3)
        # It always holds the most recent board produced; its index tracks the
        # source rather than lagging.
        assert w.latest().index >= w.frames - 2
    finally:
        w.stop()


def test_a_slow_detector_does_not_slow_the_reader():
    """The whole point: the caller reads whenever it likes and never blocks on a
    detection.
    """
    w = PerceptionWorker(FakeSource(delay=0.25), None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.latest() is not None, timeout=3.0)
        started = time.perf_counter()
        for _ in range(50):
            w.latest()
        assert (time.perf_counter() - started) < 0.05, "reading blocked"
    finally:
        w.stop()


def test_age_includes_the_detection_time():
    """Age counts from when the board looked like that, so it includes detection
    time; starting at the end of detection would understate age by exactly the
    detector's latency. The delay goes in `perceive`, not the source: a slow
    detector is what the design is built around.
    """
    delay = 0.3

    def slow_perceive(frame):
        time.sleep(delay)
        return f"state{frame.index}", f"gs{frame.index}"

    w = PerceptionWorker(FakeSource(), None, slow_perceive)
    w.start()
    try:
        assert wait_for(lambda: w.latest() is not None, timeout=3.0)
        snap = w.latest()
        assert snap.detect_ms >= delay * 1000 * 0.8
        assert snap.age_ms() >= delay * 1000 * 0.8
    finally:
        w.stop()


def test_age_grows_while_nothing_new_arrives():
    source = FakeSource(limit=1)
    w = PerceptionWorker(source, None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.latest() is not None)
        first = w.latest().age_ms()
        time.sleep(0.25)
        assert w.latest().age_ms() > first + 200
    finally:
        w.stop()


def test_an_exception_does_not_kill_the_thread():
    """A dead producer looks like a very stale board from the decision side;
    failures are counted and the thread keeps going.
    """
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("boom")
        return "state", "gs"

    w = PerceptionWorker(FakeSource(), None, flaky)
    w.start()
    try:
        assert wait_for(lambda: w.errors > 0 and w.frames > 0, timeout=3.0)
        assert w.alive
        assert isinstance(w.last_error, RuntimeError)
    finally:
        w.stop()


def test_a_none_state_is_skipped_rather_than_published():
    """The detector returns None for an unreadable frame; publishing it would hand
    the policy a GameState of None.
    """
    w = PerceptionWorker(FakeSource(), None, lambda f: (None, None))
    w.start()
    try:
        time.sleep(0.2)
        assert w.latest() is None
        assert w.alive
    finally:
        w.stop()


def test_stop_is_clean():
    w = PerceptionWorker(FakeSource(), None, perceive_ok)
    w.start()
    assert wait_for(lambda: w.frames > 0)
    w.stop()
    assert not w.alive


def test_waiting_for_a_frame_is_counted_apart_from_perceiving():
    """The producer's period is wait + work, and a starved capture and a slow
    detector look identical from outside.
    """
    source = FakeSource(delay=0.2)          # slow to hand over a frame

    def quick(frame):
        time.sleep(0.05)                    # quick to perceive one
        return "state", "gs"

    w = PerceptionWorker(source, None, quick)
    w.start()
    try:
        assert wait_for(lambda: w.frames >= 3, timeout=5.0)
    finally:
        w.stop()
    report = w.stages.report()
    assert "0 wait for frame" in report
    assert "9 perceive TOTAL" in report
    waits = w.stages._t["0 wait for frame"]
    works = w.stages._t["9 perceive TOTAL"]
    # The split attributes the cost to the right half.
    assert np.median(waits) > np.median(works)


def test_a_timed_out_read_is_not_recorded_as_a_wait():
    """A timeout means no frame arrived, a different fault from a slow one;
    averaging it in would make a dead capture look sluggish.
    """
    w = PerceptionWorker(FakeSource(limit=1), None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.frames >= 1)
        time.sleep(0.3)                     # source now returns None each call
    finally:
        w.stop()
    assert len(w.stages._t["0 wait for frame"]) == w.frames


def test_stages_reports_shares_that_sum_to_the_whole():
    s = Stages()
    for _ in range(5):
        s.add("a", 30.0)
        s.add("b", 70.0)
    report = s.report()
    assert "30.0" in report and "70.0" in report
    assert "( 30.0%)" in report and "( 70.0%)" in report


def test_stages_reports_the_median_not_the_mean():
    """One descheduled frame should not define the summary."""
    s = Stages()
    for ms in (100.0, 100.0, 100.0, 9000.0):
        s.add("x", ms)
    assert "   100.0 ms" in s.report()


def test_snapshot_age_accepts_an_explicit_now():
    """The loop passes its own timestamp so age and the decision refer to the same
    instant.
    """
    snap = Snapshot(state=None, game_state=None,
                    captured_at=time.perf_counter() - 1.0, index=0,
                    detect_ms=0.0)
    assert snap.age_ms(time.perf_counter()) == pytest.approx(1000, abs=100)


# --- producer period and the sampling residual ---

def test_period_is_none_until_it_can_be_measured():
    """Waiting on a guessed period waits for nothing."""
    w = PerceptionWorker(FakeSource(limit=1), None, perceive_ok)
    assert w.period is None
    w.start()
    try:
        assert wait_for(lambda: w.frames >= 1)
        assert w.period is None, "one publication cannot give an interval"
    finally:
        w.stop()


def test_period_tracks_the_real_publication_rate():
    delay = 0.08

    def slow(frame):
        time.sleep(delay)
        return "state", "gs"

    w = PerceptionWorker(FakeSource(), None, slow)
    w.start()
    try:
        assert wait_for(lambda: w.frames > 6, timeout=5.0)
        assert w.period == pytest.approx(delay, abs=delay * 0.6)
    finally:
        w.stop()


def test_period_uses_the_median_not_the_mean():
    """One stall must not drag the period up and make the loop wait for a board
    that is not coming.
    """
    w = PerceptionWorker(FakeSource(limit=0), None, perceive_ok)
    with w._lock:
        w._publishes.extend([0.0, 0.1, 0.2, 0.3, 2.0])   # one huge stall
    assert w.period == pytest.approx(0.1, abs=0.01)


def test_published_at_is_after_detection_and_age_exceeds_sat():
    """`age` counts from capture, `sat` from publication; the difference is
    detection time, and `sat` is the residual this mechanism removes.
    """
    delay = 0.2

    def slow(frame):
        time.sleep(delay)
        return "state", "gs"

    w = PerceptionWorker(FakeSource(limit=1), None, slow)
    w.start()
    try:
        assert wait_for(lambda: w.latest() is not None, timeout=3.0)
        snap = w.latest()
        assert snap.published_at > snap.captured_at
        now = time.perf_counter()
        assert snap.age_ms(now) > snap.sat_ms(now) + delay * 1000 * 0.8
    finally:
        w.stop()


def test_an_unset_published_at_reads_as_sitting_forever():
    """The default fails safe: a missing value makes the freshness wait decline.
    """
    snap = Snapshot(state=None, game_state=None,
                    captured_at=time.perf_counter(), index=0, detect_ms=0.0)
    assert snap.sat_ms() > 1e6
