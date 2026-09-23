"""Match clock: free-running between resyncs, corrected from the screen.

Reading the clock every frame would be wasteful and fragile (one misread digit
would jump the timeline). The clock free-runs from frame timestamps, exact on a
CFR source, and is verified against the screen every few seconds. Resync
catches discrete events the frame counter cannot see: the countdown ending, a
pause, a dropped segment.

`phase` is not inferred. Real boundaries move with balance updates, so
`set_phase_schedule()` must be called with them before `phase` returns anything
but SINGLE, and `phase_known` reports whether it was. The consumer that acts on
phase is the opponent-elixir model, where a wrong multiplier silently corrupts
the balance and disarms the missed-placement alarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts import ClockState, Phase
from timebase import TICKS_PER_SECOND, seconds_to_ticks

# How far the free-running clock may disagree with the screen before a reading
# counts as a correction. The on-screen clock has one-second resolution, so
# correcting under half a second would only inject jitter.
RESYNC_TOLERANCE_S = 0.5

# Beyond this a single reading is more likely a misread than a jump, and is
# rejected. A real discontinuity persists and is applied on the next reading.
IMPLAUSIBLE_JUMP_S = 15.0


class PhaseScheduleUnknownError(NotImplementedError):
    """Phase boundaries were never supplied. See this module's docstring."""


@dataclass
class PhaseSchedule:
    """Real-game phase boundaries, in seconds from match start. Supplied by the
    caller, never assumed. Regular time and overtime are separate, so a match
    without overtime never names an overtime boundary.
    """

    double_elixir_at_s: float
    triple_elixir_at_s: float | None
    overtime_at_s: float | None

    def phase_at(self, elapsed_s: float) -> Phase:
        if self.overtime_at_s is not None and elapsed_s >= self.overtime_at_s:
            return Phase.OVERTIME
        if self.triple_elixir_at_s is not None and elapsed_s >= self.triple_elixir_at_s:
            return Phase.TRIPLE
        if elapsed_s >= self.double_elixir_at_s:
            return Phase.DOUBLE
        return Phase.SINGLE


# Standard ladder rules: 1x for the first two minutes, 2x for the last minute
# of regular time, then overtime at 2x escalating to 3x in its final minute.
#
# A named constant a caller must still pass explicitly, not a default: balance
# updates move these boundaries, and requiring the call keeps the assumption
# visible where it is made.
LADDER_REGULAR_LENGTH_S = 180.0
LADDER_SCHEDULE_KWARGS = {
    "double_elixir_at_s": 120.0,
    # Regular time never reaches 3x; overtime does, in its final minute.
    "triple_elixir_at_s": None,
    "overtime_at_s": 180.0,
}


@dataclass
class MatchClock:
    """Converts capture time to match time, and match time to ticks."""

    offset_ms: float = 0.0
    """Capture wall time at which match tick 0 occurred. Everything else is
    derived from this one number, which is exactly what resync adjusts."""

    schedule: PhaseSchedule | None = None
    match_length_s: float | None = None
    """Regular-time length, for turning elapsed into remaining. Left None
    rather than defaulted for the same reason as the phase schedule."""

    last_resync_tick: int = 0
    last_drift_ms: float = 0.0
    resyncs_applied: int = 0
    resyncs_rejected: int = 0
    started: bool = False

    _observations: list[tuple[float, float]] = field(default_factory=list)

    # --- configuration ---

    def set_phase_schedule(self, schedule: PhaseSchedule, match_length_s: float) -> None:
        self.schedule = schedule
        self.match_length_s = match_length_s

    def start_at(self, wall_time_ms: float) -> None:
        """Declare that match tick 0 is at this capture timestamp."""
        self.offset_ms = wall_time_ms
        self.started = True

    # --- queries ---

    def elapsed_s(self, wall_time_ms: float) -> float:
        return max(0.0, (wall_time_ms - self.offset_ms) / 1000.0)

    def tick(self, wall_time_ms: float) -> int:
        return seconds_to_ticks(self.elapsed_s(wall_time_ms))

    @property
    def phase_known(self) -> bool:
        return self.schedule is not None

    def phase(self, wall_time_ms: float) -> Phase:
        if self.schedule is None:
            raise PhaseScheduleUnknownError(
                "phase boundaries were never supplied. The simulator has no "
                "phase concept to fall back on and the real boundaries change "
                "with balance updates, so there is no safe default -- call "
                "set_phase_schedule() with the real values. See "
                "perception/README.md, 'Open questions'."
            )
        return self.schedule.phase_at(self.elapsed_s(wall_time_ms))

    def phase_or_single(self, wall_time_ms: float) -> Phase:
        """Phase, defaulting to SINGLE when the schedule is unknown, for consumers
        that must produce a ClockState regardless. Confidence drops so the
        guess is never mistaken for knowledge.
        """
        if self.schedule is None:
            return Phase.SINGLE
        return self.schedule.phase_at(self.elapsed_s(wall_time_ms))

    def seconds_remaining(self, wall_time_ms: float) -> float:
        if self.match_length_s is None:
            return float("nan")
        return max(0.0, self.match_length_s - self.elapsed_s(wall_time_ms))

    # --- resync ---

    def resync(self, wall_time_ms: float, observed_remaining_s: float) -> float:
        """Correct the offset from an on-screen clock reading.

        Returns the drift in milliseconds (predicted minus observed; positive
        means the free-running clock was ahead).

        Rejects implausible corrections: a misread produces one wild reading
        and goes away, while a real discontinuity persists and is applied on
        the next reading.
        """
        if self.match_length_s is None:
            raise PhaseScheduleUnknownError(
                "resync needs match_length_s to convert an on-screen remaining "
                "time into elapsed time -- call set_phase_schedule() first."
            )
        if not self.started:
            self.start_at(wall_time_ms - (self.match_length_s - observed_remaining_s) * 1000.0)
            return 0.0

        predicted_remaining = self.seconds_remaining(wall_time_ms)
        drift_ms = (predicted_remaining - observed_remaining_s) * -1000.0
        self._observations.append((wall_time_ms, observed_remaining_s))

        if abs(drift_ms) > IMPLAUSIBLE_JUMP_S * 1000.0:
            self.resyncs_rejected += 1
            return drift_ms
        if abs(drift_ms) < RESYNC_TOLERANCE_S * 1000.0:
            self.last_resync_tick = self.tick(wall_time_ms)
            self.last_drift_ms = drift_ms
            return drift_ms

        self.offset_ms += drift_ms
        self.last_resync_tick = self.tick(wall_time_ms)
        self.last_drift_ms = drift_ms
        self.resyncs_applied += 1
        return drift_ms

    # --- output ---

    def state(self, wall_time_ms: float) -> ClockState:
        tick = self.tick(wall_time_ms)
        staleness_ticks = tick - self.last_resync_tick
        # Confidence decays with time since the last screen verification and
        # halves when the phase schedule is unknown.
        confidence = 1.0 / (1.0 + staleness_ticks / (30.0 * TICKS_PER_SECOND))
        if not self.phase_known:
            confidence *= 0.5

        return ClockState(
            tick=tick,
            seconds_remaining=self.seconds_remaining(wall_time_ms),
            phase=self.phase_or_single(wall_time_ms),
            last_resync_tick=self.last_resync_tick,
            drift_ms=self.last_drift_ms,
            confidence=round(confidence, 4),
        )
