"""Our hand, deduced from the cycle rather than read off the screen.

Read per frame, hand identity churns far faster than the game allows: most slot
changes have no elixir drop behind them, most real plays produce no change, and
cards "return" before an 8-card cycle permits.

The engine's cycle is a strict 8-slot FIFO (`track/cycle.py`), so the play
history alone determines the hand:

    queue == the last four cards played, in order
    hand  == the other four

and plays are what we read well (the elixir ledger). Cost narrows identity a
long way, and intersecting with the four cards believed to be in hand usually
leaves exactly one.

The detector is a tie-breaker, not a source. It never reports an off-deck card
or a duplicate, and its mode over a window converges on the truth, so its
errors are noise around the right answer. The mode over a wide window seeds the
tracker and detects desync; the per-frame reading only breaks ties between
equal-cost candidates. `CycleTracker.observe_hand` is not used per frame: it
overwrites the model with the observation, which is backwards here.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from contracts import UNKNOWN_CARD_SIM_ID
from track.cycle import QUEUE_SIZE, advance

# Wide enough for the mode to converge while staying under the ~9 s a real hand
# holds between plays.
CONSENSUS_WINDOW = 25

# Sustained disagreement before the tracker yields to the screen: this many
# consecutive contradicting consensus reads means the model is wrong, most
# likely from a missed play.
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
        """Forget everything, for a new match. A new battle deals a fresh opening
        hand, so the previous FIFO no longer applies. Re-seeding costs one
        consensus window; not re-seeding is wrong all match.
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
            # The ledger is surer that something was played than we are about
            # the hand: record the desync rather than drop the play, which
            # would leave the FIFO permanently behind.
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

        A mode over CONSENSUS_WINDOW frames still describes the previous hand
        for about half a window after a play, so checking then reports a desync
        on every play and keeps overwriting the tracker with a hand the game
        has left. The check waits until the window can have caught up: stale
        evidence is not a wrong model.
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
