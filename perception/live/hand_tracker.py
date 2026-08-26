"""Our hand, deduced from the cycle rather than read off the screen.

WHY THE SCREEN IS NOT THE SOURCE
--------------------------------
Hand identity is 744 of the observation's 13,606 floats, and read per frame it
is not usable. Measured over one real match (935 in-game frames, 22 plays):

    slot changes with no elixir drop      64 / 73   (88%) invented
    real plays that produced a change      9 / 22   (41%) seen
    cards returning before an 8-card cycle allows
                                          47 / 74   (64%) impossible
    median unchanged run                  0.8 s, where a real hand holds ~9.3 s

So the reading churns about 11x faster than the game permits.

WHAT IS RELIABLE INSTEAD
------------------------
The engine's cycle is a strict 8-slot FIFO (`track/cycle.py` documents it from
PlayerState.h), which gives an exact identity:

    queue == the last four cards played, in order
    hand  == the other four

So the PLAY HISTORY alone determines the hand. And plays are the thing we read
well: the elixir ledger recovered 27 cards against an affordable ceiling of 28.

Cost narrows identity a long way on this deck -- 5 is only Giant, 3 is one of
three, 4 one of four -- and intersecting with the four cards the tracker
believes are in hand usually leaves exactly one.

THE DETECTOR IS A TIE-BREAKER, NOT A SOURCE
-------------------------------------------
Two of its properties survive the churn and are worth using. It never reports a
card outside our deck (0 / 3,740) and never duplicates one (0 / 935 frames). And
its MODE over a window converges on the truth -- 70 distinct hands read raw,
27 at a 25-frame window, against a true ~23 -- so its errors are noise around
the right answer rather than a bias.

Hence: the mode over a wide window seeds the tracker and detects desync; the
per-frame reading only breaks ties between candidates of equal cost. Note this
deliberately does NOT use `CycleTracker.observe_hand` on every frame -- that
method overwrites the model with the observation, which is right when vision is
the better signal and exactly backwards here.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from contracts import UNKNOWN_CARD_SIM_ID
from track.cycle import QUEUE_SIZE, advance

# Wide enough that the mode has converged (measured: 25 frames reaches 27
# distinct hands against a true ~23) while still under the ~9 s a real hand
# holds between plays.
CONSENSUS_WINDOW = 25

# Sustained disagreement before the tracker yields to the screen. One frame is
# noise; this many consecutive consensus reads that contradict the model means
# the model is wrong, most likely from a missed play.
DESYNC_PATIENCE = 3


@dataclass
class HandTracker:
    """Our four cards, maintained from plays and corrected rarely."""

    deck: tuple[int, ...]
    costs: dict[int, float]
    hand: list[int] = field(default_factory=list)
    queue: deque = field(default_factory=lambda: deque(maxlen=QUEUE_SIZE))

    seeded: bool = False
    plays_applied: int = 0
    ambiguous: int = 0
    desyncs: int = 0

    _recent: deque = field(default_factory=lambda: deque(maxlen=CONSENSUS_WINDOW))
    _disagree_run: int = 0
    _since_play: int = 10_000

    def reset(self) -> None:
        """Forget everything, for the start of a new match.

        The FIFO is only meaningful within one match: a new battle deals a
        fresh opening hand, so carrying the previous match's queue over means
        every play advances a cycle that describes a game that already ended.
        Re-seeding costs one consensus window; not re-seeding is wrong for the
        whole match and self-corrects only through the desync path, which is
        deliberately slow.
        """
        self.hand = []
        self.queue = deque(maxlen=QUEUE_SIZE)
        self.seeded = False
        self.plays_applied = 0
        self.ambiguous = 0
        self.desyncs = 0
        self._recent.clear()
        self._disagree_run = 0
        self._since_play = 10_000

    def consensus(self) -> tuple[int, ...] | None:
        """The modal reading over the window, or None until it is full."""
        if len(self._recent) < self._recent.maxlen:
            return None
        return Counter(self._recent).most_common(1)[0][0]

    def update(self, observed_hand, new_plays) -> None:
        """One frame. `new_plays` is the costs the ledger newly explained."""
        clean = tuple(c for c in observed_hand if c != UNKNOWN_CARD_SIM_ID)
        if len(clean) == 4:
            self._recent.append(tuple(observed_hand))

        if not self.seeded:
            seed = self.consensus()
            if seed is not None and all(c in self.deck for c in seed):
                self.hand = list(seed)
                self.queue = deque(
                    [c for c in self.deck if c not in set(seed)], maxlen=QUEUE_SIZE)
                self.seeded = True
            return

        for combo in new_plays:
            for cost in combo:
                self._apply_play(cost, observed_hand)

        self._check_desync()

    def _apply_play(self, cost: float, observed_hand) -> None:
        """Advance the FIFO by one card of the given cost."""
        candidates = [c for c in self.hand
                      if abs(self.costs.get(c, -1) - cost) < 1e-6]
        if not candidates:
            # The ledger is surer that SOMETHING was played than we are about
            # what is in hand, so record the desync rather than dropping the
            # play -- dropping it would leave the FIFO permanently behind.
            self.desyncs += 1
            return
        if len(candidates) > 1:
            self.ambiguous += 1
            # Break the tie with the screen: prefer a candidate the detector
            # has stopped reporting, since that is the one that just left.
            visible = set(observed_hand)
            gone = [c for c in candidates if c not in visible]
            played = gone[0] if gone else candidates[0]
        else:
            played = candidates[0]

        advance(self.hand, self.queue, played)
        self.plays_applied += 1
        self._since_play = 0

    def _check_desync(self) -> None:
        """Yield to the screen only on sustained, consistent disagreement.

        THE CONSENSUS LAGS, AND THAT IS NOT A DISAGREEMENT. A mode over
        CONSENSUS_WINDOW frames still describes the PREVIOUS hand for about
        half a window after a play, so checking during that period reports a
        desync every single time a card is played -- measured, it fired 44
        times against 20 real plays and pinned confidence at zero, because the
        tracker kept being overwritten with a hand the game had already left.

        So the check is suppressed until the window can have caught up. This is
        the difference between the model being wrong and the evidence being
        stale.
        """
        self._since_play += 1
        if self._since_play < CONSENSUS_WINDOW:
            self._disagree_run = 0
            return
        seen = self.consensus()
        if seen is None or set(seen) == set(self.hand):
            self._disagree_run = 0
            return
        self._disagree_run += 1
        if self._disagree_run < DESYNC_PATIENCE:
            return
        self.desyncs += 1
        self._disagree_run = 0
        if all(c in self.deck for c in seen):
            self.hand = list(seen)
            self.queue = deque(
                [c for c in self.deck if c not in set(seen)], maxlen=QUEUE_SIZE)

    @property
    def confidence(self) -> float:
        """Low until seeded, then degraded by how often we have been wrong."""
        if not self.seeded:
            return 0.0
        total = max(self.plays_applied, 1)
        return max(0.0, 1.0 - self.desyncs / total)

    def as_tuple(self) -> tuple[int, ...]:
        if not self.seeded:
            return (UNKNOWN_CARD_SIM_ID,) * 4
        return tuple(self.hand)
