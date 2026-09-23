"""The opponent's elixir, derived exactly rather than observed.

Their bar is not on screen, but nothing about it is uncertain: it starts at
5.0, regenerates at a known rate, and falls only by known card costs.
Arithmetic, provided no placement is missed.

That proviso is the useful part: if the balance goes negative, a placement was
definitely missed, since the opponent cannot spend elixir they do not have. A
free, exact detector-recall alarm; `went_negative` / `negative_events` record
it before the running value is clamped.

This models the real opponent, so it uses the real game's 2.8 s per elixir; the
engine's ~2% slower rate would bias the balance ~1.3 low by three minutes and
turn the alarm into false positives (see timebase.py).

`_PHASE_MULTIPLIER` duplicates the engine's 1x/2x/3x schedule
(`GameManager::elixirMultiplierAtTick`), keyed by `contracts.Phase`, the
sensor's reading of the screen, rather than by tick. Merging them means
deciding whether the sensor reads phase off the clock instead; until then the
two must be kept equal by hand, or the alarm goes wrong in the late game only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts import Phase
from timebase import MAX_ELIXIR, STARTING_ELIXIR, elixir_regenerated, ticks_to_seconds

# How far below zero the balance may drift before it counts as proof of a miss
# rather than rounding: well under any card's cost, well above clock-drift and
# frame-quantisation error.
NEGATIVE_TOLERANCE = 0.1

_PHASE_MULTIPLIER = {
    Phase.SINGLE: 1.0,
    Phase.DOUBLE: 2.0,
    Phase.TRIPLE: 3.0,
    # Real overtime runs at the triple rate. It stays its own Phase because it
    # differs in other respects (sudden death, tower activation).
    Phase.OVERTIME: 3.0,
}


@dataclass
class OpponentElixirTracker:
    """Running derived balance for the opponent."""

    value: float = STARTING_ELIXIR
    last_tick: int = 0

    spent: float = 0.0
    unknown_cost_plays: int = 0
    """Plays whose cost could not be determined (unmapped card). The balance
    is not debited for them, so from the first one it is an OVER-estimate --
    which also disarms the negative-balance alarm. Both facts are surfaced
    via flags() rather than silently absorbed."""

    went_negative: bool = False
    negative_events: list[tuple[int, float]] = field(default_factory=list)
    """(tick, balance_before_clamp) for each proof-of-miss event."""

    def advance_to(self, tick: int, phase: Phase = Phase.SINGLE) -> None:
        """Regenerate up to `tick`, at one phase for the whole interval. Advancing
        across a phase boundary in one step is off by at most one interval.
        """
        if tick < self.last_tick:
            raise ValueError(
                f"opponent elixir cannot go backwards: at tick {self.last_tick}, "
                f"asked to advance to {tick}"
            )
        seconds = ticks_to_seconds(tick - self.last_tick)
        gained = elixir_regenerated(seconds, _PHASE_MULTIPLIER[phase])
        self.value = min(self.value + gained, MAX_ELIXIR)
        self.last_tick = tick

    def on_play(self, cost: float | None, tick: int) -> None:
        """Debit a play. `cost` None means the card could not be identified."""
        if cost is None:
            self.unknown_cost_plays += 1
            return

        self.value -= cost
        self.spent += cost

        if self.value < -NEGATIVE_TOLERANCE:
            # Recorded before clamping, which would erase the most valuable
            # signal this module produces.
            self.went_negative = True
            self.negative_events.append((tick, self.value))

        self.value = max(0.0, self.value)

    @property
    def confidence(self) -> float:
        if self.unknown_cost_plays:
            return 0.0  # the balance over-estimates by an unknown amount
        if self.went_negative:
            return 0.3  # at least one placement is known missed
        return 1.0

    def flags(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.went_negative:
            out.append("opp_elixir_negative")
        if self.unknown_cost_plays:
            out.append("opp_elixir_unknown_cost")
        return tuple(out)
