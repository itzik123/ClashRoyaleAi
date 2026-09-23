"""Cumulative elixir spend, tracked across frames from the elixir bar.

Separate from `live/adapter.py`, which is stateless (one frame in, one
GameState out) so a bad frame degrades one output. A cumulative total carries
every error forward for the rest of the match.

The bar is the strongest reader in the pipeline (mean confidence 0.978-0.991).
When we are the one playing, our placements are known exactly at issue
(`note_issued`); inference from drops remains for recordings of someone else
and whenever nothing is pending.

The reader's one failure mode is a spurious single-frame drop to zero inside a
stable run:

    7 7 7 7 0 4 4 3 3 3
    4 4 4 4 4 0 4 4 1

(CRBAB's `_calculate_elixir` takes the first window whose rolling std falls
under a threshold, so a low-variance patch at the ROI's left edge reads ~0.)
Elixir obeys rules, so these are removable without labels: it rises at most
~0.4 per 200 ms even at 3x, falls only by card costs, and a real drop persists.
A 3-median removes exactly an isolated disagreeing sample and keeps real steps;
applied twice, since two adjacent glitches survive one pass.

The check is conservation, which needs no labels:

    gained  ==  spent + (final - initial)

`gained` is modelled regeneration clipped at the cap, so time at 10 is credited
nothing. What remains is integer quantisation on the two endpoints. A large
positive residual means missed placements; a negative one, invented spend.

The ledger must be independent of the sample rate: between samples elixir
regenerates (0.036 at 10 Hz, 0.24 at the live 1.5 Hz), and each endpoint of a
drop carries +/-0.5 of quantisation. Recovered spend on a synthetic trace with
known ground truth:

              rate-blind    with regen + nearest-match
    10 Hz        -9.8%          -0.1%
     3 Hz       -27.8%          -1.0%
   1.5 Hz       -64.2%          -3.6%      <- the live rate
     1 Hz       -91.9%         -11.2%

The opponent is not tracked here: with no opponent bar, spend could only come
from units appearing, which over-counts 2.1x (BOT_REQUESTS.md item 8).
`GameState.opp_elixir_spent` stays None.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement

from timebase import MAX_ELIXIR, elixir_regenerated

# How long an issued placement may go unconfirmed by the bar before it is
# written off. A confirmation legitimately arrives ~1.6 s late (~900 ms tap,
# ~700 ms until the board is seen); 4 s allows a slow frame without holding a
# rejected play so long that the optimistic debit starves the agent.
PLAY_CONFIRM_WINDOW_S = 4.0

# Deck costs come from the caller, so this module needs no engine binding and
# stays testable without the .pyd.
DEFAULT_COSTS = (3.0, 4.0, 5.0)

# Two cards inside one sample is common; three happens in a double-elixir
# scramble. Beyond that a "drop" is a reader glitch.
MAX_CARDS_PER_STEP = 3

# How far a drop may sit from a legal cost sum and still be accepted. The
# reader returns an integer, so a drop carries up to +/-1.0. Costs are spaced 1
# apart, so beyond ~1.0 a drop matches several decompositions: a real limit of
# an integer bar, which is why the matcher takes the nearest total rather than
# pretending to resolve 3 from 4.
COST_TOLERANCE = 1.0


def decompositions(costs=DEFAULT_COSTS, max_cards=MAX_CARDS_PER_STEP):
    """Every drop total a legal set of 1..max_cards cards can produce. Anything
    else cannot be play and is a reader glitch. Double plays inside one sample
    are real and must stay in the set.
    """
    out: dict[float, tuple[float, ...]] = {}
    for n in range(1, max_cards + 1):
        for combo in combinations_with_replacement(sorted(costs), n):
            out.setdefault(float(sum(combo)), combo)
    return out


@dataclass
class ElixirLedger:
    """Running total of our own spend, fed one elixir reading per frame. Never the
    whole trace at once: live, readings arrive one at a time, and a design
    needing the future would fail on the only path that matters.
    """

    costs: tuple[float, ...] = DEFAULT_COSTS
    tolerance: float = COST_TOLERANCE
    spent: float = 0.0
    cards: int = 0
    gained: float = 0.0
    glitches: int = 0
    unexplained: int = 0

    plays: list[tuple[float, ...]] = field(default_factory=list)
    """One entry per detected placement step, holding the card costs that
    explain the drop -- (4.0,) for one card, (3.0, 4.0) for two in one sample.

    Published so the hand tracker consumes THESE rather than re-deriving drops
    from the same elixir trace. Two independent drop detectors reading one
    signal is two things that can disagree about whether a card was played,
    and the hand tracker's FIFO cannot recover from being advanced a different
    number of times than the ledger thinks."""

    rejected: int = 0
    """Placements we issued that the bar never confirmed. Almost always a tap
    that arrived after the elixir it needed had already gone."""

    confirmed_tags: list = field(default_factory=list)
    rejected_tags: list = field(default_factory=list)
    """Caller-supplied identity of each play this ledger confirmed or wrote off.

    `rejected` alone is a COUNT, which is why "17 issued plays never confirmed"
    could never be cross-examined: there was no way to ask which 17, or what
    they had in common. Tagging costs one field and turns the ledger's verdict
    into something `live/placement_confirm.py` can put on the other axis of a
    2x2 against an independent oracle."""

    _recent: list[tuple[float, float]] = field(default_factory=list)
    _last: float | None = None
    _last_t: float | None = None
    _first: float | None = None
    _table: dict = field(default_factory=dict)
    _min_cost: float = 0.0
    _pending: list[tuple[float, float]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self._table = decompositions(self.costs)
        self._min_cost = min(self.costs)

    # --- ground truth from our own play stream ---

    def record_play(self, cost: float, now: float | None = None,
                    tag=None) -> None:
        """A placement we issued, with its cost known exactly.

        Not spend yet: a tap can be rejected, so this is a hypothesis the bar
        must confirm. It turns the 3-vs-4 ambiguity of an integer bar into
        arithmetic. Inference is used only when nothing is pending.

        `now` defaults to the ledger's own clock (the last reading's timestamp)
        and must never be another time base. Mixing frame time with
        `time.monotonic()` makes every expiry comparison true, nothing is
        written off, and the optimistic debit reads zero elixir forever.
        """
        with self._lock:
            if now is None:
                now = self._last_t if self._last_t is not None else 0.0
            self._pending.append((float(cost), float(now), tag))

    @property
    def unconfirmed_cost(self) -> float:
        """Elixir committed by taps the bar has not caught up with. The agent
        decides on a ~1 s old board and its tap lands ~0.9 s later, so the bar
        still shows elixir already promised; subtracting this stops it spending
        the same elixir three times.
        """
        with self._lock:
            return sum(item[0] for item in self._pending)

    def _expire(self, now: float) -> None:
        """Drop issued plays the bar never accounted for."""
        keep, gone = [], []
        for item in self._pending:
            (keep if now - item[1] <= PLAY_CONFIRM_WINDOW_S else gone).append(item)
        self.rejected += len(gone)
        self.rejected_tags.extend(item[2] for item in gone)
        self._pending = keep

    def _confirm(self, drop: float):
        """The subset of pending plays whose costs best explain `drop`."""
        best = None
        for n in range(1, min(len(self._pending), MAX_CARDS_PER_STEP) + 1):
            for idx in combinations(range(len(self._pending)), n):
                total = sum(self._pending[i][0] for i in idx)
                err = abs(total - drop)
                if err <= self.tolerance and (best is None or err < best[0]):
                    best = (err, total, idx)
        return best

    def update(self, elixir: float | None, now: float | None = None,
               multiplier: float = 1.0) -> None:
        """One elixir reading; None when the bar could not be read.

        `now` is the reading's own timestamp, without which the ledger cannot
        model regeneration between samples and becomes a function of the sample
        rate (module docstring).

        `multiplier` is 1/2/3 for single/double/triple elixir. It defaults to 1
        and the live loop has no phase source yet (the clock reader does not
        transfer to 549x976), so late-match totals under-model regen.
        """
        if elixir is None:
            return
        now = time.monotonic() if now is None else now
        self._recent.append((now, float(elixir)))
        if len(self._recent) > 3:
            self._recent.pop(0)
        if len(self._recent) < 3:
            return

        # Median of the last three, so the value acted on is one sample behind
        # live: the cost of despiking without seeing the future. The time it
        # describes is the middle sample's, since a 3-median is centred on it;
        # taking the timestamp of whichever sample supplied the value makes dt
        # jump by a whole interval on ties.
        stamp = self._recent[1][0]
        value = sorted(v for _, v in self._recent)[1]
        if value != self._recent[1][1]:
            self.glitches += 1
        if self._first is None:
            self._first = value
        if self._last is None:
            self._last, self._last_t = value, stamp
            return

        dt = max(0.0, stamp - self._last_t)
        # The bar cannot climb past MAX_ELIXIR, so time at the cap must not be
        # credited.
        regen = min(elixir_regenerated(dt, multiplier),
                    max(0.0, MAX_ELIXIR - self._last))
        self.gained += regen
        expected = self._last + regen
        self._last, self._last_t = value, stamp

        drop = expected - value
        with self._lock:
            self._expire(stamp)
            pending = list(self._pending)

        if drop < self._min_cost - self.tolerance:
            # Ordinary regeneration or noise below the cheapest card; counting
            # it as unexplained would fire nearly every frame.
            return

        if pending:
            # Match against the costs we issued rather than every combination
            # the deck allows: a 3 and a 4 are indistinguishable on an integer
            # bar, a 3 we issued and a 4 we did not are not.
            with self._lock:
                hit = self._confirm(drop)
                if hit is not None:
                    _err, total, idx = hit
                    combo = tuple(self._pending[i][0] for i in idx)
                    self.confirmed_tags.extend(self._pending[i][2] for i in idx)
                    for i in sorted(idx, reverse=True):
                        self._pending.pop(i)
                    self.spent += total
                    self.cards += len(combo)
                    self.plays.append(combo)
                    return
            # A drop with plays outstanding that matches none of them: someone
            # else is spending, or a reading is wrong. Fall through to
            # inference.
        # Nearest total, not the smallest qualifying one, which biased every
        # ambiguous drop downward. Ties go to the cheaper explanation.
        best = None
        for total, combo in self._table.items():
            err = abs(total - drop)
            if err <= self.tolerance and (best is None or err < best[0] - 1e-9
                                          or (abs(err - best[0]) < 1e-9
                                              and total < best[1])):
                best = (err, total, combo)
        if best is None:
            # Large enough to be a placement but matching no legal combination:
            # a glitch that survived both medians, or more than
            # MAX_CARDS_PER_STEP cards in one sample.
            self.unexplained += 1
            return
        _err, total, combo = best
        self.spent += total
        self.cards += len(combo)
        self.plays.append(tuple(combo))

    @property
    def residual(self) -> float:
        """gained - spent - (current - initial). Expected near zero: `gained` is
        modelled regeneration clipped at the cap, leaving endpoint
        quantisation. Large positive means missed placements; negative,
        invented spend.
        """
        if self._first is None or self._last is None:
            return 0.0
        return self.gained - self.spent - (self._last - self._first)
