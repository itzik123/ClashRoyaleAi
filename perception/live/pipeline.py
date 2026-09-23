"""Run perception continuously on its own thread; decide at a fixed rate.

Serially, the decision rate is pinned to the detector's latency, every decision
acts on a board at least that old (2.4 s is a Hog Rider crossing the bridge),
and one slow frame delays every later decision. Splitting them turns an
unbounded stall into a bounded, visible staleness.

    producer thread   capture -> resize -> detect -> GameState, plus the
                      elixir ledger, which wants every observation
    decision loop     encode -> policy -> act (~12 ms)

The adapter (~137 ms) sits on the producer side, keeping the decision path
short.

`Snapshot.age_ms` is the age of the board a decision was made on; the loop
prints it every iteration and counts threshold breaches, so a producer falling
behind shows up at once rather than as strange play.

The producer's period is `wait + work`, and the two have opposite fixes: `wait`
is time blocked in `read_new` for the window to paint, `work` is split by
stage. Without the split a frame-starved capture and a slow detector look
identical.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

# Recent publications the period estimate uses: robust to one slow frame, still
# tracking a real change in load.
PERIOD_WINDOW = 12

# Recent samples kept per stage timing. Larger than PERIOD_WINDOW, since a
# stage median is read once at the end; a whole-run figure would describe
# neither now nor then, since load drifts.
STAGE_WINDOW = 256


class Stages:
    """Accumulates per-stage timings across frames, thread-safely.

    Reports the median: a per-frame mean is dominated by whichever frame was
    descheduled. Timings live in a bounded deque (STAGE_WINDOW); the total
    count is kept separately and reported as `n=`, because stages before an
    early exit (detector.run) and after it (build_game_state) legitimately
    differ in count.
    """

    def __init__(self):
        self._t: dict[str, deque[float]] = {}
        self._n: dict[str, int] = {}
        self._lock = threading.Lock()

    def _record(self, name: str, ms: float) -> None:
        """Caller must hold the lock."""
        if name not in self._t:
            self._t[name] = deque(maxlen=STAGE_WINDOW)
            self._n[name] = 0
        self._t[name].append(ms)
        self._n[name] += 1

    def time(self, name: str, t0: float) -> float:
        """Record `now - t0` against `name`; return now, to chain calls."""
        now = time.perf_counter()
        with self._lock:
            self._record(name, (now - t0) * 1000.0)
        return now

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            self._record(name, ms)

    def report(self, total_key: str | None = None) -> str:
        with self._lock:
            snapshot = {k: np.array(v) for k, v in self._t.items()}
            counts = dict(self._n)
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
            n = counts.get(name, len(v))
            # "n=1234" is every sample ever seen; "(last 256)" marks the median
            # as over the window.
            windowed = "" if n <= len(v) else f" (last {len(v)})"
            lines.append(f"    {name:<26} {med:8.1f} ms  ({share:5.1f}%)  "
                         f"n={n}{windowed}")
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
    """Perceives as fast as it can; publishes only the newest result. No queue: a
    superseded board is not merely less useful but wrong, so dropping is
    correct.
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
        # `wait` is time blocked on the window painting, `work` time in
        # perceive; their sum is the period, and which dominates decides
        # whether to fix capture or compute.
        self.stages = Stages()
        # Recent publication times, so the decision loop can predict the next
        # board. Bounded, since the rate drifts with load.
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
        """Seconds between boards, or None until measurable. The median of recent
        intervals: one busy-machine outlier would drag a mean up and have the
        loop wait for a board that is not coming. None rather than a guess:
        waiting on a made-up period waits for nothing.
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
                    # A timeout is not a wait to average in: no frame arrived
                    # at all, a different fault, and it would drag the median
                    # toward the timeout.
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
                # Never let the thread die silently: a dead producer looks like
                # a very stale board from the decision side, and needs a
                # different response.
                self.errors += 1
                self.last_error = exc
                time.sleep(0.1)
