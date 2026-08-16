"""Is a battle actually running right now?

WHY THIS EXISTS
---------------
`mvp_loop.perceive()` gated exactly one thing on the screen -- the elixir
ledger -- and built the GameState unconditionally. So every reader ran on the
lobby, the matchmaking screen and the victory screen, and their output went
into the same stream the agent acts on.

Measured over the 2,510-frame live capture in
`assets/live/match_practice_01`, using the vendored ScreenDetector as the gate:

                              ungated        gated on in_game
    off-deck cards read         7.8%              0.5%
    duplicate cards             8.7%              1.7%
    median unchanged hand      0.60 s            1.20 s
    frames kept                 100%             74.5%

A quarter of every capture is not a battle. The card slots there are empty, so
the classifier is forced onto its blank padding and reports whatever the
Hungarian assignment settles on; the elixir ROI reads a confident 0.00 because
the bar has not been drawn yet. None of that is a classifier error and none of
it can be debounced away downstream -- it is a question nobody asked.

WHY DEBOUNCE, AND WHY ASYMMETRICALLY
------------------------------------
Per-decile on the same capture, the detector reads `in_game` on 100% of frames
through the middle of the match and mixes `unknown`/`lobby` only at the two
boundaries. So mid-match blips are not the risk; the boundaries are.

Entering is held for `ENTER_HOLD` frames because acting on a half-loaded arena
is worse than missing the first second of it. Leaving is held for `EXIT_HOLD`,
which is longer, because a single dropped frame at the wrong moment would end
the match in our bookkeeping while the real one is still being played -- and
`on_match_end` resets the ledger, which is not recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IN_GAME = "in_game"

# Frames of agreement before a transition is believed. Capture runs at ~5 Hz,
# so these are ~0.4 s in and ~1.2 s out.
ENTER_HOLD = 2
EXIT_HOLD = 6


@dataclass
class MatchState:
    """Debounced "a battle is running", from the screen classification.

    Feed it every frame's screen name. `in_match` is the gate; `started` and
    `ended` fire exactly once per transition so callers can reset per-match
    state (the elixir ledger, the cycle tracker) without tracking edges
    themselves.
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
            # transition needs CONSECUTIVE disagreement rather than cumulative.
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
