"""At most one action per perceived board, and never on a stale one.

WHY THIS EXISTS
---------------
The decision loop runs at a fixed 1 Hz; the producer runs at whatever rate
perception manages. Those are independent by design -- that is the whole point
of the split -- so nothing structurally prevents two decisions from reading the
SAME board and both acting on it.

It was not hypothetical. Measured in dry run before DirectML landed, with the
producer at 0.36 Hz against a 1 Hz loop: 44 taps for ~22 placements, the policy
repeating itself two and three times over as the board sat unchanged
(`net slot 0 -> (3,8)` three decisions running). In dry run that is a cosmetic
oddity in a log. With `--act` it is three Giants for one intent, at 5 elixir
each, and the agent then reads an elixir bar that its own duplicate taps
emptied.

DirectML made the producer FASTER than the decision loop (3.99 Hz against 1 Hz),
so the duplicates are currently rare -- and that is exactly why the gate has to
exist as an invariant rather than as a performance workaround. The condition
that produced them is one GPU hiccup away, and the failure mode is silent: a
double-placed card looks like a legal placement to everything downstream.

TWO REFUSALS, DELIBERATELY DISTINGUISHED
----------------------------------------
`duplicate` means the board was already acted on. `stale` means the board is old
enough that acting on it is a guess about a game that has moved on. They are
reported separately because they mean different things about the system: a run
full of `duplicate` means the producer is slower than the loop, a run full of
`stale` means it has stopped keeping up altogether. Collapsing them into one
"blocked" counter would hide which.

WHAT THIS DOES NOT DO
---------------------
It does not suppress the DECISION, only the tap. The policy is recurrent and
was trained stepping once per second; skipping steps to match the producer's
rate would change the LSTM's cadence away from what it was trained on. So the
network steps every tick as before and only the action is gated.
"""
from __future__ import annotations

from dataclasses import dataclass

# How old a board may be before acting on it is a guess about a game that has
# already moved on. Two seconds is roughly a Hog Rider crossing the bridge.
# Lives here rather than in mvp_loop so the threshold that REPORTS staleness and
# the one that ACTS on it cannot drift apart.
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
        """Whether an action on this board is permitted. Does not record it.

        Checking is separate from recording so a caller can ask without
        committing -- and, more importantly, so a tap that RAISES is not
        counted as having happened. An actuation failure must leave the board
        available to retry rather than burning it.
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
