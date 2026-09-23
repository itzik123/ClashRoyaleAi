"""At most one action per perceived board, and never on a stale one.

The decision loop runs at 1 Hz and the producer at whatever perception manages;
nothing structurally stops two decisions reading the same board and both acting
on it. With `--act` that is several Giants for one intent, and the agent then
reads a bar its own duplicate taps emptied. A faster producer makes it rare,
which is why this is an invariant and not a workaround: a double-placed card
looks legal to everything downstream.

Two refusals, reported separately: `duplicate` (the board was already acted on)
means the producer is slower than the loop; `stale` (the board is too old to
act on) means it has stopped keeping up.

Only the tap is suppressed, never the decision: the policy is recurrent and
trained stepping once per second, so the network still steps every tick.
"""
from __future__ import annotations

from dataclasses import dataclass

# How old a board may be before acting on it is a guess. Two seconds is roughly
# a Hog Rider crossing the bridge. Defined here so the threshold that reports
# staleness and the one that acts on it cannot drift apart.
MAX_STALENESS_MS = 2000.0


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class ActionGate:
    """Decides whether a tap may be issued for a given perceived board."""

    def __init__(self, max_age_ms: float = MAX_STALENESS_MS,
                 enforce_staleness: bool = True):
        self.max_age_ms = max_age_ms
        self.enforce_staleness = enforce_staleness
        self._acted_on: int | None = None
        self.allowed = 0
        self.refused: dict[str, int] = {}

    def check(self, board_index: int, age_ms: float) -> GateDecision:
        """Whether an action on this board is permitted. Does not record it: a tap
        that raises must leave the board available to retry.
        """
        if self._acted_on is not None and board_index == self._acted_on:
            return GateDecision(False, "duplicate")
        if self.enforce_staleness and age_ms > self.max_age_ms:
            return GateDecision(False, "stale")
        return GateDecision(True, "ok")

    def record(self, board_index: int) -> None:
        """Mark this board as acted on. Call only after the tap succeeded."""
        self._acted_on = board_index
        self.allowed += 1

    def refuse(self, reason: str) -> None:
        self.refused[reason] = self.refused.get(reason, 0) + 1

    def summary(self) -> str:
        if not self.refused:
            return f"actions: {self.allowed} allowed, none blocked"
        blocked = ", ".join(f"{n} {reason}"
                            for reason, n in sorted(self.refused.items()))
        return f"actions: {self.allowed} allowed, blocked {blocked}"
