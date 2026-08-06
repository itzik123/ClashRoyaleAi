"""Run perception continuously on its own thread; decide at a fixed rate.

WHY THE LOOP CANNOT STAY SERIAL
-------------------------------
Measured live on a quiet machine: 29 iterations in 60 s, 0.48 Hz, with the
detector clustering at 2.2-2.5 s. Serially that is fatal in a way no amount of
detector tuning fixes, because the coupling itself is the problem:

  * the decision RATE is pinned to the detector's LATENCY, so a 2.4 s detector
    means at most 0.4 decisions a second;
  * every decision is then acting on a board at least 2.4 s old, and in Clash
    Royale 2.4 s is a Hog Rider crossing the bridge;
  * worst case is unbounded -- one slow frame delays every later decision.

Splitting them converts an unbounded stall into a bounded STALENESS. The
decision loop runs at its own rate on the most recent observation, and how old
that observation is becomes a number we can watch instead of a hang we cannot.

WHAT RUNS WHERE
---------------
    producer thread   capture -> resize -> detect -> GameState, plus the
                      elixir ledger, which wants every observation it can get
    decision loop     encode -> policy -> act, measured at 12 ms total

The adapter sits on the PRODUCER side deliberately. It costs ~137 ms, and
leaving it on the decision side would put the slowest remaining piece back in
the path the whole design exists to keep short.

STALENESS IS REPORTED, NEVER HIDDEN
-----------------------------------
`Snapshot.age_ms` is the age of the board the decision was made on. A consumer
that ignores it is back to the original problem with extra steps, so the loop
prints it every iteration and counts how often it exceeds a threshold. If age
grows without bound the producer is not keeping up and that is visible
immediately, rather than showing up as a policy that plays strangely.

WAITING IS SEPARATED FROM WORKING
---------------------------------
The producer's period is `wait + work`, and the two have opposite fixes. Offline
the whole chain measures 190 ms a frame (115 ms of it the ONNX forward pass);
the first live run showed a ~2700 ms period. A 14x gap that size is not the
model, so `work` is split by stage and `wait` -- time blocked in `read_new`
waiting for the window to paint -- is counted separately. Without that split a
capture starved of frames and a detector that is genuinely slow look identical
from outside, and they lead to completely different work.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

# How many recent publications the period estimate is taken over. Long enough
# to be robust to one slow frame, short enough to track a real change in load.
PERIOD_WINDOW = 12


class Stages:
    """Accumulates per-stage timings across frames, safely across threads.

    Reports the MEDIAN. A per-frame mean is dominated by whichever frame the
    scheduler happened to descheduled -- the live run's own totals ranged
    1.9-9.3 s while the typical frame was nothing like either end.
    """

    def __init__(self):
        self._t: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def time(self, name: str, t0: float) -> float:
        """Record `now - t0` against `name`; return now, to chain calls."""
        now = time.perf_counter()
        with self._lock:
            self._t.setdefault(name, []).append((now - t0) * 1000.0)
        return now

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            self._t.setdefault(name, []).append(ms)

    def report(self, total_key: str | None = None) -> str:
        with self._lock:
            snapshot = {k: np.array(v) for k, v in self._t.items()}
        if not snapshot:
            return "  (no stages recorded)"
        total = (float(np.median(snapshot[total_key]))
                 if total_key and total_key in snapshot
                 else sum(float(np.median(v)) for v in snapshot.values()))
        lines = []
        for name in sorted(snapshot):
            v = snapshot[name]
            med = float(np.median(v))
            share = 100.0 * med / total if total > 0 else 0.0
            lines.append(f"    {name:<26} {med:8.1f} ms  ({share:5.1f}%)  "
                         f"n={len(v)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Snapshot:
    """One perception result, with the time it describes."""

    state: Any
    """The CRBAB State, for callers that need `ready` or `screen`."""

    game_state: Any
    captured_at: float
    """perf_counter when the frame was RECEIVED -- before detection, not after.

    Age has to start before the slow part or it understates by exactly the
    detector's latency, which is the entire quantity of interest (~2.4 s live).

    It is the receive time rather than the true capture instant, so it
    under-counts by however long the surface sat in the source before being
    read. That is bounded by the WGC inter-frame gap, measured at ~85 ms, and
    is negligible against the detection time it exists to expose. Reaching into
    the source's clock to close that gap would couple two time bases for a
    3% correction."""

    index: int
    detect_ms: float

    published_at: float = 0.0
    """perf_counter when this became the newest board -- AFTER detection.

    `captured_at` says how old the board is; this says how long it has been
    sitting available. The difference is the sampling residual: time the
    decision loop held a finished board without acting on it, which is pure
    waste and the only part of the latency that costs nothing to remove.

    Defaults to 0 so an unset value reads as "sitting forever", which makes
    `sat_ms` large and the freshness wait decline to trigger. Degrading to the
    old always-act-now behaviour is the safe direction for a missing field.
    """

    def age_ms(self, now: float | None = None) -> float:
        return ((time.perf_counter() if now is None else now)
                - self.captured_at) * 1000.0

    def sat_ms(self, now: float | None = None) -> float:
        """How long this board has been published and unused."""
        return ((time.perf_counter() if now is None else now)
                - self.published_at) * 1000.0


class PerceptionWorker:
    """Perceives as fast as it can; publishes only the newest result.

    Deliberately keeps NO queue. A backlog would mean the decision loop reads
    boards that are already superseded -- the freshest observation is the only
    one worth having, and an old one is not merely less useful but actively
    wrong. Dropping is the correct behaviour, not a compromise.
    """

    def __init__(self, source, detector, adapt, on_observation=None,
                 name: str = "perception"):
        self._source = source
        self._detector = detector
        self._adapt = adapt
        self._on_observation = on_observation
        self._latest: Snapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.frames = 0
        self.errors = 0
        self.last_error: BaseException | None = None
        # `wait` is time blocked on the window painting; `work` is time in
        # perceive. Their sum is the producer's period, and which of the two
        # dominates decides whether the next move is capture or compute.
        self.stages = Stages()
        # Recent publication times, for the decision loop to predict when the
        # next board lands. Bounded: the rate drifts with machine load, and an
        # average over the whole run would describe neither now nor then.
        self._publishes: deque[float] = deque(maxlen=PERIOD_WINDOW)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def latest(self) -> Snapshot | None:
        with self._lock:
            return self._latest

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def period(self) -> float | None:
        """Seconds between boards, or None until it can be measured.

        The MEDIAN of recent intervals, not the mean: perception occasionally
        takes several times its usual duration when the machine is busy, and a
        mean dragged upward by one of those would tell the decision loop to
        wait for a board that is not coming.

        None rather than a guess when there is not enough history -- a caller
        that waits on a made-up period waits for nothing.
        """
        with self._lock:
            stamps = list(self._publishes)
        if len(stamps) < 3:
            return None
        gaps = np.diff(stamps)
        return float(np.median(gaps)) if len(gaps) else None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                waited = time.perf_counter()
                frame = self._source.read_new(timeout_s=1.0)
                if frame is None:
                    # A timeout is not a wait worth averaging in: it means no
                    # frame arrived at all, which is a different fault from a
                    # slow one and would drag the median toward the timeout.
                    continue
                captured_at = self.stages.time("0 wait for frame", waited)
                started = captured_at
                state, game_state = self._adapt(frame)
                self.stages.time("9 perceive TOTAL", started)
                if state is None:
                    continue
                published_at = time.perf_counter()
                snapshot = Snapshot(
                    state=state, game_state=game_state,
                    captured_at=captured_at, index=frame.index,
                    detect_ms=(published_at - started) * 1000.0,
                    published_at=published_at)
                with self._lock:
                    self._publishes.append(published_at)
                if self._on_observation is not None:
                    self._on_observation(state, game_state)
                with self._lock:
                    self._latest = snapshot
                self.frames += 1
            except Exception as exc:                    # noqa: BLE001
                # Never let the thread die silently. A dead producer looks
                # exactly like a very stale board from the decision side, and
                # the two need completely different responses.
                self.errors += 1
                self.last_error = exc
                time.sleep(0.1)
