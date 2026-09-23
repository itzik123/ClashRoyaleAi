"""Card-cycle tracking, for either player.

PlayerState holds `hand` (4) and `deckQueue` (4), and PlayerState::playCard
does exactly this:

    nextCard = deckQueue.front();
    deckQueue.pop_front();
    deckQueue.push_back(cardId);      // the card just played goes to the back
    hand[handIndex] = nextCard;       // the front fills the slot it vacated
    handCooldownTicks[handIndex] = 20;

A strict 8-slot FIFO, preserving hand position as the real game does. The queue
is four long and grows only at the back, one entry per play, so:

    queue == the last four cards played, in order
    hand  == the other four
    next  == the fourth-most-recent play

The opponent's hand is not on screen, but their plays are, and from the fourth
play on the play history alone determines their cycle exactly. Opponent hand
prediction is bookkeeping, not a guess, which is why this module has no
heuristics.

Not modelled: `handCooldownTicks`, the 2 s before a freshly cycled slot can be
played. It affects legality, not what the cycle contains, and perception only
reports plays that already happened.
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


def advance(hand: list[int], queue: "deque[int]", played: int,
            index: int | None = None) -> None:
    """Advance the 8-slot FIFO by one play: the one implementation of
    PlayerState::playCard's rule,

        nextCard = deckQueue.front(); deckQueue.pop_front();
        deckQueue.push_back(cardId);
        hand[handIndex] = nextCard;

    shared by live/hand_tracker.py and track/deduce_identity.py. Callers differ
    in how they work out which card was played (a known id, a cost, a candidate
    permutation), never in what the FIFO then does.

    `index` lets a caller that knows the slot skip the lookup; the permutation
    search needs it, since `hand.index()` finds the wrong slot for duplicate
    ids. Mutates `hand` and `queue` in place. An empty `hand` means no hand
    model is seeded yet; the play is still recorded in the queue.
    """
    if hand:
        if index is None:
            index = hand.index(played)
        hand[index] = queue.popleft() if queue else UNKNOWN_CARD_SIM_ID
    queue.append(played)
    # Explicit trim: `queue` is not always constructed with a maxlen.
    while len(queue) > QUEUE_SIZE:
        queue.popleft()


@dataclass
class CycleTracker:
    """Tracks one player's hand and queue through a match.

    Two ways in, matching what is observable:

      * `observe_hand()` -- vision read our four hand slots. Authoritative; used to
        seed and to re-verify.
      * `on_play()` -- a card was played. Advances the FIFO.

    For the opponent only `on_play()` exists, which is enough once the deck is
    known.
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

    # --- seeding ---

    def set_deck(self, deck: tuple[int, ...]) -> None:
        if len(set(deck)) != len(deck):
            raise ValueError(f"deck has duplicates: {deck}")
        self.deck = tuple(deck)
        self._rebuild_from_history()

    def seed_hand(self, hand: tuple[int, ...]) -> None:
        """Set the opening hand, with the queue as whatever is left. The queue's
        order is unknown until its cards are played; `queue_certain` says which
        situation we are in, so an unknown `next_card` is not trusted as a
        placeholder.
        """
        self.hand = list(hand)
        if self.deck:
            remainder = [c for c in self.deck if c not in set(hand)]
            self.queue = deque(remainder, maxlen=QUEUE_SIZE)
        self.observations += 1

    # --- advancing ---

    def on_play(self, card_id: int) -> None:
        """Record that `card_id` was played. Advances the FIFO."""
        self.play_history.append(card_id)

        if self.hand and card_id not in self.hand:
            # Played a card we did not believe was in hand: a missed earlier
            # play or a misread hand. The play is the harder fact, so trust it
            # and rebuild.
            self.desyncs += 1
            self._rebuild_from_history()
            return

        advance(self.hand, self.queue, card_id)

    def observe_hand(self, hand: tuple[int, ...]) -> bool:
        """Vision read the hand. Returns True if it matched the prediction;
        corrects on mismatch. Position matters: the engine preserves slot
        position across a cycle, so the same cards in different slots mean one
        side is wrong about the play history.
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
            # Keep the order the play history implies for cards still in the
            # queue; only append those with no ordering information, so
            # `next_card` survives.
            ordered = [c for c in self.queue if c in remainder]
            ordered += [c for c in remainder if c not in ordered]
            self.queue = deque(ordered[:QUEUE_SIZE], maxlen=QUEUE_SIZE)
        return False

    # --- derived ---

    def _rebuild_from_history(self) -> None:
        """Recompute hand and queue from the play history and the deck: the queue
        is the last four plays, the hand everything else. Exact, so recovering
        from a desync is a rebuild.
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
        """[0,1]. Completeness times how often the model has been right, so an
        unverified tracker and a repeatedly wrong one are both discounted.
        """
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
