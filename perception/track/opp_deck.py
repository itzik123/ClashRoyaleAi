"""Discovering the opponent's deck during the match.

The deck is revealed one card at a time as it is played. The FIFO plays every
card once before any card twice, so the first repeat means the cycle wrapped
and the whole deck has been seen: a completion detector, not an estimator. If
fewer than eight distinct cards were seen at that moment, a placement was
missed, and the gap is loud.

Cards are held as simulator ids. An unmapped card (mapping/) has no id and
cannot occupy a slot; `unmapped_seen` counts those, so an incomplete deck for
that reason is distinguishable from one with a missed play.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts import UNKNOWN_CARD_SIM_ID
from track.cycle import DECK_SIZE, CycleTracker


@dataclass
class OpponentDeckTracker:
    """Builds up the opponent's 8-card deck from observed plays."""

    known: list[int] = field(default_factory=list)
    """Distinct simulator card ids seen, in order of first appearance."""

    first_seen_tick: dict[int, int] = field(default_factory=dict)
    unmapped_seen: int = 0
    complete_tick: int | None = None

    wrapped_tick: int | None = None
    """Tick at which a card repeated -- i.e. the cycle demonstrably wrapped."""

    missed_at_wrap: int = 0
    """How many deck slots were still unaccounted for when the cycle wrapped.
    Non-zero is proof that at least this many placements were missed. This is
    the cheapest hard evidence of a detection failure the pipeline has, and it
    needs no ground truth at all."""

    cycle: CycleTracker = field(default_factory=CycleTracker)

    def on_play(self, card_sim_id: int, tick: int) -> None:
        if card_sim_id == UNKNOWN_CARD_SIM_ID:
            # A real card with no simulator counterpart occupies one of their
            # eight slots, but with no id the deck can never complete: counted,
            # not guessed.
            self.unmapped_seen += 1
            return

        if card_sim_id in self.first_seen_tick:
            if self.wrapped_tick is None:
                self.wrapped_tick = tick
                self.missed_at_wrap = max(0, DECK_SIZE - len(self.known) - self.unmapped_seen)
        else:
            self.known.append(card_sim_id)
            self.first_seen_tick[card_sim_id] = tick
            if len(self.known) == DECK_SIZE and self.complete_tick is None:
                self.complete_tick = tick
                self.cycle.set_deck(tuple(self.known))

        self.cycle.on_play(card_sim_id)

    @property
    def is_complete(self) -> bool:
        return len(self.known) == DECK_SIZE

    @property
    def deck(self) -> frozenset[int]:
        return frozenset(self.known)

    @property
    def confidence(self) -> float:
        """How much of the deck is pinned down: membership, as opposed to the
        cycle tracker's confidence about ordering.
        """
        return round(len(self.known) / DECK_SIZE, 4)

    def flags(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.missed_at_wrap > 0:
            out.append("opp_deck_incomplete_at_wrap")
        if self.unmapped_seen > 0:
            out.append("opp_deck_has_unmapped_card")
        return tuple(out)
