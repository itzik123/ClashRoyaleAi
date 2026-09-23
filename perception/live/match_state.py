"""Is a battle actually running right now?

Every reader would otherwise run on the lobby, matchmaking and victory screens
and feed the stream the agent acts on. A quarter of a live capture is not a
battle: the card slots are empty, so the classifier reports whatever its blank
padding settles on, and the elixir ROI reads a confident 0.00. None of that is
a classifier error, and none of it can be debounced downstream.

The screen classifier reads `in_game` on every frame mid-match and mixes
`unknown`/`lobby` only at the boundaries, so the debounce is asymmetric.
Entering is held `ENTER_HOLD` frames: acting on a half-loaded arena is worse
than missing its first second. Leaving is held longer, `EXIT_HOLD`: one dropped
frame must not end the match in our bookkeeping, since `on_match_end` resets
the ledger irrecoverably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IN_GAME = "in_game"

# Frames of agreement before a transition is believed: at ~5 Hz capture, ~0.4 s
# in and ~1.2 s out.
ENTER_HOLD = 2
EXIT_HOLD = 6


@dataclass
class MatchState:
    """Debounced "a battle is running", from the screen classification. Feed it
    every frame's screen name. `in_match` is the gate; `started` and `ended`
    fire once per transition so callers can reset per-match state (ledger,
    cycle tracker).
    """

    enter_hold: int = ENTER_HOLD
    exit_hold: int = EXIT_HOLD

    in_match: bool = False
    matches_seen: int = 0
    frames_in_match: int = 0
    frames_skipped: int = 0

    _agree: int = 0
    _listeners: list = field(default_factory=list)

    def on_change(self, callback) -> None:
        """Register `callback(in_match: bool)`, fired once per transition."""
        self._listeners.append(callback)

    def update(self, screen_name: str) -> bool:
        """Feed one frame's screen. Returns the current gate."""
        looks_live = screen_name == IN_GAME

        if looks_live == self.in_match:
            # Agreeing with the current belief clears any pending flip, so a
            # transition needs consecutive disagreement.
            self._agree = 0
        else:
            self._agree += 1
            needed = self.enter_hold if looks_live else self.exit_hold
            if self._agree >= needed:
                self.in_match = looks_live
                self._agree = 0
                if looks_live:
                    self.matches_seen += 1
                for callback in self._listeners:
                    callback(self.in_match)

        if self.in_match:
            self.frames_in_match += 1
        else:
            self.frames_skipped += 1
        return self.in_match

    @property
    def kept_fraction(self) -> float:
        total = self.frames_in_match + self.frames_skipped
        return (self.frames_in_match / total) if total else 0.0
