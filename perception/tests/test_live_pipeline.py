"""Tests for live/pipeline.py, against a synthetic source.

The worker exists to break the coupling that made the live loop 0.48 Hz: with
perception inline, the decision RATE is pinned to the detector's LATENCY. These
tests pin the properties that make the split safe, using a fake source and a
fake perceive() so a slow detector can be simulated deterministically rather
than waited for.
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
    """No queue, on purpose. A backlog would feed the decision loop boards that
    are already superseded -- an old board is not merely less useful, it is
    wrong, so dropping is correct rather than a compromise."""
    source = FakeSource()
    w = PerceptionWorker(source, None, perceive_ok)
    w.start()
    try:
        assert wait_for(lambda: w.frames > 5)
        first = w.latest().index
        assert wait_for(lambda: w.latest().index > first + 3)
        # Whatever it holds is always the most recent one produced, never a
        # backlog entry: its index tracks the source rather than lagging.
        assert w.latest().index >= w.frames - 2
    finally:
        w.stop()


def test_a_slow_detector_does_not_slow_the_reader():
    """The whole point. Perception at ~4 Hz here; the caller reads whenever it
    likes and never blocks on a detection."""
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
    """Age counts from when the board LOOKED like that, so it must already
    include however long detection took. Starting the clock when detection
    FINISHES would understate age by exactly the detector's latency -- which
    is the quantity the whole split exists to expose, currently ~2.4 s.

    The delay goes in `perceive`, not in the source: a slow DETECTOR is what
    this design is built around. An earlier version of this test put it in the
    source and so measured nothing of the sort.
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
    """A dead producer looks exactly like a very stale board from the decision
    side, and the two need completely different responses -- so failures are
    counted and the thread keeps going."""
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
    """The detector returns None for an unreadable frame. Publishing that would
    hand the policy a GameState of None."""
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
    """The producer's period is wait + work and the two have opposite fixes:
    a starved capture and a slow detector are indistinguishable from the
    outside, and the first live run's 2700 ms period was attributed to the
    detector on no evidence -- offline the detector medians 130 ms."""
    source = FakeSource(delay=0.2)          # slow to HAND OVER a frame

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
    # The point of the split: it attributes the cost to the right half.
    assert np.median(waits) > np.median(works)


def test_a_timed_out_read_is_not_recorded_as_a_wait():
    """A timeout means NO frame arrived, which is a different fault from a slow
    one. Averaging it in would drag the median toward the timeout and make a
    dead capture look like a merely sluggish one."""
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
    """One descheduled frame should not define the summary -- the live run's
    own per-frame totals ranged 1.9-9.3 s."""
    s = Stages()
    for ms in (100.0, 100.0, 100.0, 9000.0):
        s.add("x", ms)
    assert "   100.0 ms" in s.report()


def test_snapshot_age_accepts_an_explicit_now():
    """The loop passes its own timestamp so age and the decision refer to the
    same instant."""
    snap = Snapshot(state=None, game_state=None,
                    captured_at=time.perf_counter() - 1.0, index=0,
                    detect_ms=0.0)
    assert snap.age_ms(time.perf_counter()) == pytest.approx(1000, abs=100)
