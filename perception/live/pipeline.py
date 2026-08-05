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
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


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

    def age_ms(self, now: float | None = None) -> float:
        return ((time.perf_counter() if now is None else now)
                - self.captured_at) * 1000.0


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

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._source.read_new(timeout_s=1.0)
                if frame is None:
                    continue
                captured_at = time.perf_counter()
                started = captured_at
                state, game_state = self._adapt(frame)
                if state is None:
                    continue
                snapshot = Snapshot(
                    state=state, game_state=game_state,
                    captured_at=captured_at, index=frame.index,
                    detect_ms=(time.perf_counter() - started) * 1000.0)
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
