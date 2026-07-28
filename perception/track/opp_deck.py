"""Discovering the opponent's deck during the match.

The opponent's deck is not shown anywhere. It is revealed one card at a time,
by being played, and is fully known once all eight have appeared -- which
happens by definition at the end of their first full cycle, since the FIFO
guarantees every card is played once before any card is played twice.

That guarantee is worth stating explicitly, because it is what makes this a
completion detector rather than an estimator: the moment a card REPEATS, we
know the cycle has wrapped, and therefore that everything in the deck has now
been seen. So the deck can be declared complete at the first repeat, even if
fewer than eight distinct cards have been observed -- which is exactly the
situation when a placement was missed. Comparing "cards seen" against "cards
that must exist" at that moment turns a silent gap into a loud one.

Cards are held here as simulator ids. An unmapped card (see mapping/) has no
id, so it cannot occupy a deck slot; `unmapped_seen` counts those separately
so a deck that looks incomplete for that reason is distinguishable from one
that is incomplete because a play was missed.
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
            # Real card, no simulator counterpart. It genuinely occupies one
            # of their eight slots, but we have no id to put in that slot, so
            # the deck can never complete -- counted, not guessed at.
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
        """How much of the deck is pinned down. Distinct from the cycle
        tracker's own confidence, which is about ordering rather than
        membership."""
        return round(len(self.known) / DECK_SIZE, 4)

    def flags(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.missed_at_wrap > 0:
            out.append("opp_deck_incomplete_at_wrap")
        if self.unmapped_seen > 0:
            out.append("opp_deck_has_unmapped_card")
        return tuple(out)
