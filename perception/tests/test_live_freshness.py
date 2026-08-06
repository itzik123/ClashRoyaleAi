"""Tests for the freshness wait in live/mvp_loop.py.

The decision loop samples at 1 Hz on a phase unrelated to the producer's, so
the board it acts on has usually been sitting finished for part of a producer
period. That residual ages the world model without buying anything, and it is
the only part of the end-to-end latency that costs nothing to remove.

Waiting is not free though -- the action lands later in wall time -- so most of
what matters here is when the wait must DECLINE.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("clashroyalebuildabot", reason="vendored bot not importable")
pytest.importorskip("windows_capture", reason="live capture not installed")

from live.mvp_loop import (  # noqa: E402
    FRESHNESS_WAIT_CAP_S,
    wait_for_fresher,
)
from live.pipeline import Snapshot  # noqa: E402


class FakeWorker:
    """A worker whose period is fixed and whose next board arrives on cue."""

    def __init__(self, period, snap, next_snap=None, arrives_in=None):
        self.period = period
        self._snap = snap
        self._next = next_snap
        self._at = (time.perf_counter() + arrives_in
                    if arrives_in is not None else None)
        self.reads = 0

    def latest(self):
        self.reads += 1
        if self._at is not None and time.perf_counter() >= self._at:
            return self._next
        return self._snap


def snapshot(index, sat_s):
    """A board published `sat_s` ago."""
    now = time.perf_counter()
    return Snapshot(state=None, game_state=None, captured_at=now - sat_s - 0.1,
                    index=index, detect_ms=100.0, published_at=now - sat_s)


def test_declines_when_the_period_is_not_yet_known():
    """Waiting on a guessed period waits for nothing."""
    snap = snapshot(1, sat_s=0.5)
    got, waited = wait_for_fresher(FakeWorker(None, snap), snap)
    assert got is snap
    assert waited == 0.0


def test_declines_when_the_board_just_arrived():
    """The next board is a whole period away. Waiting would trade a lot of
    delay for a board that is already as fresh as it gets."""
    snap = snapshot(1, sat_s=0.01)
    worker = FakeWorker(period=0.5, snap=snap)
    started = time.perf_counter()
    got, waited = wait_for_fresher(worker, snap)
    assert got is snap
    assert waited == 0.0
    assert (time.perf_counter() - started) < 0.02, "it blocked anyway"


def test_declines_when_the_next_board_is_further_off_than_the_cap():
    """A long wait is a worse trade than a stale board: the cadence jitter and
    the delayed action cost more than the freshness is worth."""
    snap = snapshot(1, sat_s=0.05)
    worker = FakeWorker(period=FRESHNESS_WAIT_CAP_S + 0.4, snap=snap)
    got, waited = wait_for_fresher(worker, snap)
    assert got is snap
    assert waited == 0.0


def test_waits_and_takes_the_fresher_board():
    """The case it exists for: the board in hand is stale, the next is
    imminent, so holding briefly buys a much newer world model."""
    stale = snapshot(1, sat_s=0.18)
    fresh = snapshot(2, sat_s=0.0)
    worker = FakeWorker(period=0.25, snap=stale, next_snap=fresh,
                        arrives_in=0.04)
    got, waited = wait_for_fresher(worker, stale)
    assert got is fresh
    assert 0 < waited < FRESHNESS_WAIT_CAP_S * 1000 + 50


def test_a_board_that_never_arrives_does_not_block_past_the_cap():
    """A stalled producer must not hold the decision loop open. The cap is what
    bounds the cadence damage when the prediction is wrong."""
    stale = snapshot(1, sat_s=0.20)
    worker = FakeWorker(period=0.25, snap=stale)      # nothing ever arrives
    started = time.perf_counter()
    got, waited = wait_for_fresher(worker, stale)
    elapsed = time.perf_counter() - started
    assert got is stale
    assert elapsed <= FRESHNESS_WAIT_CAP_S + 0.05
    assert waited > 0


def test_it_never_returns_an_older_board():
    """Whatever it returns must be at least as fresh as what it was given --
    the whole point is reducing the gap, never widening it."""
    stale = snapshot(5, sat_s=0.20)
    fresh = snapshot(6, sat_s=0.0)
    worker = FakeWorker(period=0.25, snap=stale, next_snap=fresh,
                        arrives_in=0.03)
    got, _ = wait_for_fresher(worker, stale)
    assert got.index >= stale.index
    assert got.published_at >= stale.published_at
