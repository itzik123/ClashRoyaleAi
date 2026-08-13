"""Cumulative elixir spend, tracked across frames from the elixir bar.

Separate from `live/adapter.py` on purpose. The adapter is stateless -- one
frame in, one GameState out -- so a dropped frame degrades exactly one output
and nothing carries forward. A cumulative total is the opposite: it accumulates,
and an error in it is permanent for the rest of the match. Mixing the two would
make every GameState depend on every prior call.

WHY THE ELIXIR BAR AND NOT THE HAND
-----------------------------------
The obvious way to spot our own placement is a hand slot changing. Measured over
one live match that scored **1 of 76** against the elixir drop, so the ledger
was built on the bar -- the strongest reader in the pipeline (mean confidence
0.978-0.991) -- and the hand consulted only to NAME the card afterwards.

**That measurement is no longer trustworthy.** It predates the 2026-08-05 fix
to `adapter._hand_ids`, which was reading `state.cards[:4]` when `cards[0]` is
the "Next" preview box and only `[1:5]` are hand slots. So the hand it was
scored against had a card the player did not hold in slot 0, every real card
shifted one right, and slot 3 missing -- of course it did not track placements.
Live runs since the fix show the hand cycling correctly (play slot 1, slot 1
goes blank, the next card fills it), which is exactly the signal that scored
1 of 76.

Re-measure before treating "the bar, not the hand" as settled. It is not a
tie-break either: we are now the one PLAYING, so our own placements are known
exactly at the moment they are issued, and inferring them from pixels at all is
a design left over from when perception was the only source.

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
ceiling of 28**, and a conservation residual of +14%. That was at ~10 fps, off
a recording.

Conservation is the check, and it needs no labels:

    gained  ==  spent + (final - initial)

`gained` is MODELLED regeneration, clipped at the cap, rather than observed
rises off the bar -- so time spent sitting at 10 is credited nothing instead of
being invisible, and the old one-sided positive bias is gone. What remains is
integer quantisation on the two endpoint readings. A large positive residual
means missed placements; a negative one means spend was invented.

RATE INDEPENDENCE IS THE PROPERTY THAT MATTERS
----------------------------------------------
Fed one reading per frame, this used to be a function of the FRAME RATE, which
is not a property anyone intended it to have. `drop = last - value` assumed no
regeneration between samples: true at 10 fps (0.036 elixir) and badly false at
the 1.5 Hz the live producer runs at (0.24). Measured against a synthetic trace
with known ground truth, recovered spend:

              rate-blind    with regen + nearest-match
    10 Hz        -9.8%          -0.1%
     3 Hz       -27.8%          -1.0%
   1.5 Hz       -64.2%          -3.6%      <- the live rate
     1 Hz       -91.9%         -11.2%

Two causes, both fixed: the missing regeneration term, and a COST_TOLERANCE of
0.55 on a reader quantised to whole elixir, where a drop carries +/-1.0.

THE OPPONENT IS NOT TRACKED HERE
--------------------------------
Deliberately. There is no opponent elixir bar, so their spend could only come
from units appearing, and that over-counts 2.1x with implied elixir below zero
for 98% of a match. See BOT_REQUESTS.md item 8, and `GameState.opp_elixir_spent`,
which stays None.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement

from timebase import MAX_ELIXIR, elixir_regenerated

# How long an ISSUED placement may go unconfirmed by the bar before it is
# written off as never having landed.
#
# A tap takes ~900 ms to reach the game and the board it produces is seen
# ~700 ms after that, so a confirmation legitimately arrives ~1.6 s late. Four
# seconds leaves room for a slow frame without holding a rejected play so long
# that the optimistic debit starves the agent of elixir it actually has.
PLAY_CONFIRM_WINDOW_S = 4.0

# Deck costs are supplied by the caller rather than imported, so this module
# does not need the engine binding and stays testable without the .pyd.
DEFAULT_COSTS = (3.0, 4.0, 5.0)

# Two cards inside one sample is common; three is rare but happens in a
# double-elixir scramble. Beyond that a "drop" is a reader glitch, not play.
MAX_CARDS_PER_STEP = 3

# How far a drop may sit from a legal cost sum and still be accepted.
#
# The reader returns an INTEGER 0..10, so each endpoint of a drop carries up to
# +/-0.5 and the drop itself up to +/-1.0. 0.55 was fitted at 10 fps, where a
# drop spans two samples 100 ms apart and the true elixir barely moves between
# the rounding; it is far too tight once samples are two thirds of a second
# apart. Measured: a real 3-cost play, true elixir 3.4 -> 0.4, reads 3 -> 1 and
# presents as a drop of 2.24. Rejected at 0.55, accepted at 1.0.
#
# It cannot be widened without limit: costs are spaced 1 apart, so beyond ~1.0
# a drop matches several decompositions and the choice becomes arbitrary. That
# is a real information limit of an integer bar, not a tuning failure -- which
# is why the matcher takes the NEAREST total and accepts ~0.5 of error per
# card rather than pretending to resolve 3 from 4.
COST_TOLERANCE = 1.0


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

    # -- ground truth from our own play stream --------------------------------

    def record_play(self, cost: float, now: float | None = None,
                    tag=None) -> None:
        """A placement we ISSUED, with its cost known exactly.

        Not spend yet. A tap can be rejected -- by the time it reaches the game
        the elixir it needed may already have gone -- so this is a hypothesis
        the bar still has to confirm.

        What it buys is the end of guessing. Inferring spend from a drop alone
        cannot separate a 3-cost from a 4-cost on a bar quantised to whole
        elixir; knowing what we played turns that ambiguity into arithmetic.
        Inference remains for the observer case (a recording of someone else
        playing), and is used only when nothing is pending.

        `now` DEFAULTS TO THE LEDGER'S OWN CLOCK -- the timestamp of the last
        reading -- and must never be a different time base from the one
        `update()` is fed. The live loop stamps readings with the frame's
        capture time, which starts near zero, while `time.monotonic()` is a
        number in the hundreds of thousands: mixing them makes every expiry
        comparison true, nothing is ever written off, and `unconfirmed_cost`
        grows without bound until the optimistic debit reports zero elixir
        forever. Defaulting to the frame clock makes that unrepresentable.
        """
        with self._lock:
            if now is None:
                now = self._last_t if self._last_t is not None else 0.0
            self._pending.append((float(cost), float(now), tag))

    @property
    def unconfirmed_cost(self) -> float:
        """Elixir committed by taps the bar has not caught up with yet.

        The agent decides at 1 Hz on a board ~1 s old and its tap lands ~0.9 s
        later, so the bar it is reading still shows the elixir it has already
        promised away. Subtracting this is what stops it spending the same
        elixir three times -- the burst that fills the actuator queue.
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
        """One elixir reading. None when the bar could not be read at all.

        `now` is the reading's own timestamp; without it the ledger cannot know
        how much elixir REGENERATED between samples, and it silently becomes a
        function of the sample rate. Measured against a synthetic trace with
        known ground truth, spend recovered by the old rate-blind version:

            10 Hz  -5%      the rate it was built and scored against
             2 Hz  -32%
           1.5 Hz  -48%     the live rate, once the producer became the
             1 Hz  -82%     detector's rate

        A card costs 3-5, so at 10 fps the 0.036 elixir of regen inside a
        sample is invisible, while at 1.5 Hz it is 0.24 -- a quarter of the way
        to the next legal cost, on a reader already quantised to whole elixir.

        `multiplier` is 1/2/3 for single/double/triple elixir. It DEFAULTS TO 1
        and the live loop has no phase source yet -- the clock reader does not
        transfer to 549x976 -- so double elixir currently under-models regen and
        re-introduces a milder version of the same under-count. Worth knowing
        before trusting a late-match total.
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
        # live. That lag is the cost of despiking without seeing the future.
        # The timestamp travels WITH the reading: pairing the median value with
        # the newest time would overstate dt by one whole sample interval, and
        # dt is now load-bearing.
        # The value is the median of the three; the TIME it describes is the
        # middle sample's, because a 3-median is centred on it. Taking the
        # timestamp of whichever sample supplied the median value instead makes
        # dt jump around by a whole sample interval on ties -- which was
        # harmless when dt was unused and is not now.
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
        # Regeneration is capped: the bar cannot climb past MAX_ELIXIR, so time
        # spent sitting at the cap adds nothing and must not be credited.
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
            # Ordinary regeneration, or noise below the cheapest card. Not a
            # play, and not worth counting as unexplained -- at this sample
            # rate that would fire on nearly every frame.
            return

        if pending:
            # We know what we tried to play, so match the drop against THOSE
            # costs rather than every combination the deck could produce. A
            # 3 and a 4 are indistinguishable on an integer bar; a 3 we issued
            # and a 4 we did not are not.
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
            # else is spending on this account, or a reading is wrong. Fall
            # through to inference rather than silently dropping it.
        # NEAREST total, not the smallest qualifying one. The old rule biased
        # every ambiguous drop downward, and with an integer reader most drops
        # are ambiguous -- a systematic under-count on top of the missed cards.
        # Ties go to the cheaper explanation, which is the conservative side.
        best = None
        for total, combo in self._table.items():
            err = abs(total - drop)
            if err <= self.tolerance and (best is None or err < best[0] - 1e-9
                                          or (abs(err - best[0]) < 1e-9
                                              and total < best[1])):
                best = (err, total, combo)
        if best is None:
            # Large enough to be a placement but matching no legal combination
            # of card costs. The below-cheapest case returned above, so this is
            # a genuine oddity: a reader glitch that survived both medians, or
            # more cards inside one sample than MAX_CARDS_PER_STEP allows.
            self.unexplained += 1
            return
        _err, total, combo = best
        self.spent += total
        self.cards += len(combo)
        self.plays.append(tuple(combo))

    @property
    def residual(self) -> float:
        """gained - spent - (current - initial). Expected near zero.

        `gained` is now MODELLED regeneration, clipped at the 10 cap rather
        than read off the bar, so the old one-sided bias is gone: time spent
        capped is credited nothing instead of being invisible. What is left is
        integer quantisation on the two endpoint readings.

        A large positive residual still means placements were missed; a
        negative one means spend was invented.
        """
        if self._first is None or self._last is None:
            return 0.0
        return self.gained - self.spent - (self._last - self._first)
