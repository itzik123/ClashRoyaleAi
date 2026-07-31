"""Cumulative elixir spend, tracked across frames from the elixir bar.

Separate from `live/adapter.py` on purpose. The adapter is stateless -- one
frame in, one GameState out -- so a dropped frame degrades exactly one output
and nothing carries forward. A cumulative total is the opposite: it accumulates,
and an error in it is permanent for the rest of the match. Mixing the two would
make every GameState depend on every prior call.

WHY THE ELIXIR BAR AND NOT THE HAND
-----------------------------------
The obvious way to spot our own placement is a hand slot changing. Measured over
one live match that scored **1 of 76** against the elixir drop, which is not a
surprise in hindsight: README already records the card-icon template agreeing
with the elixir ledger on cost only 33.8% of the time and over-predicting Giant
at 35% against a 12.5% prior. The elixir bar is the strongest reader in the
pipeline (mean confidence 0.978-0.991), so the ledger is built on it and the
hand is consulted only to NAME the card afterwards.

THE READER HAS ONE FAILURE MODE, AND PHYSICS REMOVES IT
--------------------------------------------------------
Spurious single-frame drops to zero, sitting inside otherwise stable runs:

    7 7 7 7 0 4 4 3 3 3
    4 4 4 4 4 0 4 4 1

Visible in CRBAB's `_calculate_elixir`: it takes the FIRST window whose rolling
std falls under a threshold, so a momentary low-variance patch at the left edge
of the ROI reads ~0 regardless of the true level. 4.2% of samples.

They are removable with no labels because elixir obeys rules:

  * it rises at most ~0.4 per 200 ms step even at 3x, so any larger rise is
    impossible;
  * it falls only by a card's cost;
  * a real drop PERSISTS -- you cannot spend 4 and have it back next frame.

So an isolated sample disagreeing with both neighbours is a glitch, and a
3-median removes exactly that shape while preserving real steps, which last
many frames at any sane sampling rate. Applied twice, because two adjacent
glitches survive one pass.

WHAT IT SCORED
--------------
On one live match, 204 s in-game: **27 cards implied against an affordable
ceiling of 28**, and a conservation residual of +14%.

Conservation is the check, and it needs no labels:

    gained  ==  spent + (final - initial)  [+ whatever overflowed at the cap]

The overflow term makes it one-sided -- elixir regenerated while the bar sits at
10 is invisible, so `gained` under-counts and a small positive residual is
expected. The bar was capped in 13% of samples. A large residual means missed
placements.

THE OPPONENT IS NOT TRACKED HERE
--------------------------------
Deliberately. There is no opponent elixir bar, so their spend could only come
from units appearing, and that over-counts 2.1x with implied elixir below zero
for 98% of a match. See BOT_REQUESTS.md item 8, and `GameState.opp_elixir_spent`,
which stays None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

# Deck costs are supplied by the caller rather than imported, so this module
# does not need the engine binding and stays testable without the .pyd.
DEFAULT_COSTS = (3.0, 4.0, 5.0)

# Two cards inside one sample is common; three is rare but happens in a
# double-elixir scramble. Beyond that a "drop" is a reader glitch, not play.
MAX_CARDS_PER_STEP = 3

# How far a drop may sit from a legal cost sum and still be accepted. Slightly
# over half an elixir: the reader is integer-quantised, and a regen tick can
# land inside the same sample as a placement.
COST_TOLERANCE = 0.55


def decompositions(costs=DEFAULT_COSTS, max_cards=MAX_CARDS_PER_STEP):
    """Every drop total a legal set of 1..max_cards cards can produce.

    A drop outside this set cannot be play and is treated as a reader glitch.
    Rejecting anything above the single-card maximum was the bug in the first
    version: two cards played inside one sample give a drop of 6-10, which is a
    real double-play, and discarding those left 39% of the match's elixir
    unaccounted for.
    """
    out: dict[float, tuple[float, ...]] = {}
    for n in range(1, max_cards + 1):
        for combo in combinations_with_replacement(sorted(costs), n):
            out.setdefault(float(sum(combo)), combo)
    return out


@dataclass
class ElixirLedger:
    """Running total of our own spend, fed one elixir reading per frame.

    Deliberately not fed the whole trace at once: live, the readings arrive one
    at a time, and a design that needs the future would work offline and fail
    on the only path that matters.
    """

    costs: tuple[float, ...] = DEFAULT_COSTS
    spent: float = 0.0
    cards: int = 0
    gained: float = 0.0
    glitches: int = 0
    unexplained: int = 0

    _recent: list[float] = field(default_factory=list)
    _last: float | None = None
    _first: float | None = None
    _table: dict = field(default_factory=dict)

    def __post_init__(self):
        self._table = decompositions(self.costs)

    def update(self, elixir: float | None) -> None:
        """One elixir reading. None when the bar could not be read at all."""
        if elixir is None:
            return
        self._recent.append(float(elixir))
        if len(self._recent) > 3:
            self._recent.pop(0)
        if len(self._recent) < 3:
            return

        # Median of the last three, so the value acted on is one sample behind
        # live. That lag is the cost of despiking without seeing the future,
        # and it is 200 ms at the rate this runs.
        a, b, c = self._recent
        value = sorted((a, b, c))[1]
        if value != b:
            self.glitches += 1
        if self._first is None:
            self._first = value
        if self._last is None:
            self._last = value
            return

        delta = value - self._last
        self._last = value
        if delta > 0:
            self.gained += delta
            return
        if delta == 0:
            return

        drop = -delta
        best = None
        for total, combo in self._table.items():
            if abs(total - drop) <= COST_TOLERANCE:
                if best is None or total < best[0]:
                    best = (total, combo)
        if best is None:
            # Below the cheapest card, so it cannot be a placement. Almost
            # always integer quantisation noise of 1-2 elixir.
            self.unexplained += 1
            return
        total, combo = best
        self.spent += total
        self.cards += len(combo)

    @property
    def residual(self) -> float:
        """gained - spent - (current - initial). Expected small and POSITIVE.

        Positive because regen while the bar sits at its 10 cap is invisible, so
        `gained` under-counts. A large positive residual means placements were
        missed; a negative one means spend was invented.
        """
        if self._first is None or self._last is None:
            return 0.0
        return self.gained - self.spent - (self._last - self._first)
