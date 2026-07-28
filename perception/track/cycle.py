"""Card-cycle tracking, for either player.

THE ENGINE'S CYCLE, EXACTLY
---------------------------
PlayerState holds `hand` (4) and `deckQueue` (4), and playCard does this and
nothing else (PlayerState.h:208-216):

    nextCard = deckQueue.front();
    deckQueue.pop_front();
    deckQueue.push_back(cardId);      // the card just played goes to the back
    hand[handIndex] = nextCard;       // the front fills the slot it vacated
    handCooldownTicks[handIndex] = 20;

A strict 8-slot FIFO. The played card goes to the back, the front comes into
the vacated slot -- position in hand is preserved, which matches what the
real game does visually.

THE CONSEQUENCE THAT MAKES OPPONENT TRACKING POSSIBLE AT ALL
------------------------------------------------------------
The queue is exactly four long and only ever grows at the back, one entry per
play. So:

    queue == the last four cards played, in order
    hand  == the other four
    next  == the fourth-most-recent play

We cannot see the opponent's hand -- it is not on screen. But we can see what
they play. And the identity above says the play history ALONE determines
their entire cycle state, exactly, with no inference and no probability,
from the fourth play onward (before that the queue's initial contents are
still partly unknown).

That is why this module has no heuristics in it. Opponent hand prediction is
not a guess here; it is bookkeeping.

WHAT THIS DOES NOT MODEL
------------------------
`handCooldownTicks` -- the 20-tick (2s) delay before a freshly cycled slot
can be played again. It affects whether a play is LEGAL, not what the cycle
contains, and perception only ever reports plays that already happened. It
would matter for an agent choosing actions, and is deliberately out of scope
for a state estimator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from contracts import UNKNOWN_CARD_SIM_ID, CycleState

HAND_SIZE = 4
DECK_SIZE = 8
QUEUE_SIZE = DECK_SIZE - HAND_SIZE


class CycleDesyncError(RuntimeError):
    """Tracked state contradicts an observation, unrecoverably."""


@dataclass
class CycleTracker:
    """Tracks one player's hand and queue through a match.

    Two ways in, matching the two things that are actually observable:

      * `observe_hand()` -- vision read our four hand slots. Authoritative;
        used to seed and to re-verify.
      * `on_play()` -- a card was played. Advances the FIFO.

    For the opponent only `on_play()` is ever available, which is enough (see
    module docstring) once the deck is known.
    """

    deck: tuple[int, ...] = ()
    """The eight card ids. Empty until discovered -- see opp_deck.py."""

    hand: list[int] = field(default_factory=list)
    queue: deque[int] = field(default_factory=deque)

    play_history: list[int] = field(default_factory=list)
    """Every card played, oldest first. The ground truth this rebuilds from."""

    desyncs: int = 0
    """Times an observation contradicted the tracked hand. Feeds confidence:
    a tracker that keeps being corrected is not to be trusted between
    corrections either."""

    corrections: int = 0
    observations: int = 0

    # -- seeding ---------------------------------------------------------

    def set_deck(self, deck: tuple[int, ...]) -> None:
        if len(set(deck)) != len(deck):
            raise ValueError(f"deck has duplicates: {deck}")
        self.deck = tuple(deck)
        self._rebuild_from_history()

    def seed_hand(self, hand: tuple[int, ...]) -> None:
        """Set the opening hand, with the queue as whatever is left.

        The queue ORDER is not determined by this -- four cards are known to
        be in it but their sequence is not, and nothing on screen reveals it.
        The order becomes known only as they are played. `queue_certain`
        reports which situation we are in, so consumers can tell an unknown
        `next_card` from a known one instead of trusting a placeholder.
        """
        self.hand = list(hand)
        if self.deck:
            remainder = [c for c in self.deck if c not in set(hand)]
            self.queue = deque(remainder, maxlen=QUEUE_SIZE)
        self.observations += 1

    # -- advancing -------------------------------------------------------

    def on_play(self, card_id: int) -> None:
        """Record that `card_id` was played. Advances the FIFO."""
        self.play_history.append(card_id)

        if card_id in self.hand:
            index = self.hand.index(card_id)
            incoming = self.queue.popleft() if self.queue else UNKNOWN_CARD_SIM_ID
            self.hand[index] = incoming
        elif self.hand:
            # Played something we did not believe was in hand. Real causes:
            # a missed earlier play, or a misread hand. Not fatal -- the
            # play itself is a harder fact than our model of the hand, so
            # trust it and rebuild rather than discarding it.
            self.desyncs += 1
            self._rebuild_from_history()
            return

        self.queue.append(card_id)
        while len(self.queue) > QUEUE_SIZE:
            self.queue.popleft()

    def observe_hand(self, hand: tuple[int, ...]) -> bool:
        """Vision read the hand. Returns True if it matched the prediction.

        Corrects on mismatch. Position matters: two hands with the same cards
        in different slots are a genuine disagreement, because the engine
        preserves slot position across a cycle, so a positional mismatch
        means one of the two is wrong about the play history.
        """
        self.observations += 1
        observed = list(hand)
        if self.hand and observed == self.hand:
            return True

        if self.hand:
            self.desyncs += 1
            self.corrections += 1
        self.hand = observed

        if self.deck:
            remainder = [c for c in self.deck if c not in set(observed)]
            # Keep whatever order the play history implies for the cards that
            # are still in the queue; only append the ones we had no ordering
            # information for. Discarding the known order would throw away
            # `next_card` for no reason.
            ordered = [c for c in self.queue if c in remainder]
            ordered += [c for c in remainder if c not in ordered]
            self.queue = deque(ordered[:QUEUE_SIZE], maxlen=QUEUE_SIZE)
        return False

    # -- derived ---------------------------------------------------------

    def _rebuild_from_history(self) -> None:
        """Recompute hand and queue from the play history and the deck.

        Uses the identity from the module docstring: the queue is the last
        four plays, the hand is everything else. Exact, not approximate --
        which is what makes recovery from a desync a rebuild rather than a
        patch-up.
        """
        if not self.deck:
            self.queue = deque(self.play_history[-QUEUE_SIZE:], maxlen=QUEUE_SIZE)
            return
        recent = self.play_history[-QUEUE_SIZE:]
        self.queue = deque(recent, maxlen=QUEUE_SIZE)
        in_queue = set(recent)
        self.hand = [c for c in self.deck if c not in in_queue]

    @property
    def queue_certain(self) -> bool:
        """Whether the queue's ORDER is known, not merely its membership."""
        return len(self.play_history) >= QUEUE_SIZE and len(self.deck) == DECK_SIZE

    @property
    def next_card(self) -> int:
        if not self.queue or not self.queue_certain:
            return UNKNOWN_CARD_SIM_ID
        return self.queue[0]

    @property
    def confidence(self) -> float:
        """[0,1]. Product of how complete the model is and how often it has
        been right, so an unverified tracker and a repeatedly-wrong one are
        both discounted -- for different reasons, but both correctly."""
        completeness = 0.0
        if len(self.deck) == DECK_SIZE:
            completeness += 0.5
        if len(self.hand) == HAND_SIZE:
            completeness += 0.3
        if self.queue_certain:
            completeness += 0.2

        if self.observations == 0:
            agreement = 1.0
        else:
            agreement = max(0.0, 1.0 - self.desyncs / self.observations)
        return round(completeness * agreement, 4)

    def state(self) -> CycleState:
        return CycleState(
            hand=tuple(self.hand),
            next_card=self.next_card,
            queue=tuple(self.queue) if self.queue_certain else (),
            confidence=self.confidence,
        )
