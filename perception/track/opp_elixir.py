"""The opponent's elixir, derived exactly rather than observed.

Their bar is not on screen. But nothing about it is uncertain: the starting
value is fixed (5.0), the regeneration rate is a known constant, and the only
thing that ever subtracts from it is playing a card whose cost is known. So
this is arithmetic, not estimation -- provided no placement is missed.

THAT PROVISO IS THE MOST USEFUL PART
------------------------------------
If the balance ever goes NEGATIVE, a placement was definitely missed. Not
"probably" -- the opponent cannot spend elixir they do not have, so a
negative balance is arithmetic proof that a card was played and not seen.

Free, exact, no ground truth, no labelling. It is the cheapest detector-recall
alarm available anywhere in this pipeline, and `went_negative` /
`negative_events` exist to make sure it is never silently clamped away. The
running value is clamped for downstream sanity, but the fact is recorded
first.

WHY THE REAL RATE AND NOT THE SIMULATOR'S
------------------------------------------
This models the real opponent, so it uses the real game's 2.8s per elixir.
The simulator regenerates about 2% slower (see timebase.py's own discussion),
and using its rate here would bias the balance roughly 1.3 elixir low by the
three-minute mark -- turning the certain-proof alarm above into a steady
stream of false positives. The two rates are kept apart deliberately.

PHASE MULTIPLIERS: THE SIMULATOR HAS THEM NOW (2026-09-02)
-----------------------------------------------------------
This section used to read "the simulator has no phase concept to feed them
into", and confining the multipliers here was how that gap was kept from
leaking. **That gap is closed.** `GameManager::elixirMultiplierAtTick` now runs
the real 1x/2x/3x schedule (double from 2:00, triple from 3:00), the engine
exposes `elixir_multiplier_at_tick(tick)` and the phase is observation scalar 9
-- see `perception/UPSTREAM_REQUESTS.md` item 26.

Two consequences, and neither is done here:

  * The multiplier is no longer a real-game-only fact. A live mirror driven
    through `set_current_tick` gets the correct phase automatically, because
    the engine derives it from its own clock.
  * `_PHASE_MULTIPLIER` below is therefore now a SECOND COPY of a schedule the
    engine owns, which is exactly what CLAUDE.md's no-second-copies rule is
    about. It is left alone deliberately for now, because the two are not yet
    the same question: this table is keyed by `contracts.Phase` (a state the
    SENSOR infers from the screen, and which carries OVERTIME as distinct for
    reasons unrelated to elixir), while the engine's is keyed by tick. Merging
    them means deciding whether the sensor should read the phase off the clock
    instead of off the screen, which is a real design question and not a
    rename.

Until that is settled, the rates must be kept EQUAL by hand. If the engine's
schedule moves and this table does not, the missed-placement alarm goes wrong
in the late game only -- the hardest possible window to notice it in.

WHY THE REAL RATE HERE AND THE ENGINE'S RATE THERE, still
----------------------------------------------------------
Unchanged by the above: the ~2% base-rate difference is deliberate and the
phase multiplier composes on top of whichever base rate a given model uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts import Phase
from timebase import MAX_ELIXIR, STARTING_ELIXIR, elixir_regenerated, ticks_to_seconds

# How far below zero the balance may drift before it is called proof of a
# miss rather than accumulated rounding. One tenth of an elixir is well under
# any single card's cost and well above the error from clock drift or a
# frame-quantised placement time.
NEGATIVE_TOLERANCE = 0.1

_PHASE_MULTIPLIER = {
    Phase.SINGLE: 1.0,
    Phase.DOUBLE: 2.0,
    Phase.TRIPLE: 3.0,
    # Overtime's rate is not a separate rate -- real overtime runs at the
    # triple rate. Kept as its own Phase because it differs in every other
    # respect (sudden death, tower activation), and collapsing the two would
    # lose that.
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
        """Regenerate up to `tick`.

        Regeneration is applied at one phase for the whole interval, so
        callers must advance across a phase boundary in two steps if they
        want it exact. In practice the clock ticks far more often than
        phases change, making the error at a boundary at most one interval.
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
            # Recorded BEFORE clamping. Clamping first would erase the single
            # most valuable signal this module produces.
            self.went_negative = True
            self.negative_events.append((tick, self.value))

        self.value = max(0.0, self.value)

    @property
    def confidence(self) -> float:
        if self.unknown_cost_plays:
            return 0.0  # the balance is an over-estimate of unknown size
        if self.went_negative:
            return 0.3  # known to have missed at least one placement
        return 1.0

    def flags(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.went_negative:
            out.append("opp_elixir_negative")
        if self.unknown_cost_plays:
            out.append("opp_elixir_unknown_cost")
        return tuple(out)
