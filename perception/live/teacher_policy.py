"""`--policy teacher`: the UtilityTeacher plays as us (team 0) in a real match.

Each decision: rebuild the mirror from the latest GameState (live/mirror.py),
let the teacher propose and rank candidates by rolling them forward on
`env.snapshot()`, and hand the winner to the actuator. Design:
docs/design/specs/2026-08-24-live-teacher-play-design.md; TODO.md item 0.

The ENV is rebuilt per decision; the TEACHER persists, because it carries the
second half of a committed combo (`self.pending`) across decisions.

The bias throughout is no-op over a wrong action: a mistimed card is worse
than a skipped decision. Every path that cannot vouch for its answer holds:

  the hand could not be set   the mirror's slot numbers are not the screen's
  the teacher raised          counted, never fatal
  the slot is not ready       the screen disagrees with the mirror's elixir
  the row is not tappable     the tap would land outside the arena

Three couplings to the read-only teacher, each pinned by a test:

  * Plan clock. `tick_plan` subtracts COMBO_FOLLOWUP_DELAY_TICKS (1 s) per
    `act()`, so the teacher assumes exactly one decision a second. A stalled
    decision makes a planned escort gap longer in wall time than it was scored
    at. Measured here, not corrected: `drift_s` and the `[drift]` tag.
  * Opponent spend. The teacher switches its reactive rollout off after
    COUNTER_PASSIVE_DECISIONS decisions in which `env.get_elixir_spent(1)`
    did not rise. A rebuilt mirror's spend is always 0, so unattended that rule
    would decide the opponent is passive 8 s into every match. The policy
    feeds it perception's own signal instead: a new opponent card on the board
    counts as spending (`_note_opponent_activity`).
  * Frames. For team 0 the teacher's own frame IS the engine frame, so its
    (x, y) is already the engine tile the actuator expects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from live.decision import DEFAULT_TILE, Decision
from live.mirror import MirrorBuilder

#: Seconds over the nominal period before a decision interval counts as drift.
DRIFT_TOLERANCE_S = 0.5


@dataclass
class TeacherStats:
    decisions: int = 0
    plays: int = 0
    holds: int = 0
    hand_refused: int = 0
    teacher_errors: int = 0
    not_ready: int = 0
    untappable: int = 0
    drift_events: int = 0
    build_ms: list = field(default_factory=list)
    teacher_ms: list = field(default_factory=list)

    def summary(self) -> str:
        def p50(xs):
            return f"{float(np.median(xs)):.1f}" if xs else "-"
        return (f"decisions {self.decisions}, plays {self.plays}, holds "
                f"{self.holds}, hand refused {self.hand_refused}, teacher "
                f"errors {self.teacher_errors}, not ready {self.not_ready}, "
                f"untappable {self.untappable}, drift {self.drift_events}; "
                f"build p50 {p50(self.build_ms)} ms, teacher p50 "
                f"{p50(self.teacher_ms)} ms")


class TeacherPolicy:
    """Implements the loop's `decide(gs, ready, now) -> Decision` interface."""

    def __init__(self, deck, stage: int | None = None, profile="balanced",
                 seed: int = 0, decision_hz: float = 1.0, builder=None,
                 teacher=None, tappable=None):
        from python_ai.opponents import teacher as T  # noqa: PLC0415

        self._T = T
        self.deck = [int(c) for c in deck]
        # The teacher's probe board defaults to the first eight registered ids,
        # which is unordered_map iteration order. MSVC happens to put a legal
        # deck first; libstdc++ puts a Hero in slot 0, the engine refuses it,
        # and no UtilityTeacher can be built. The probe's cards are documented
        # as irrelevant (the card under test is injected), so a deck the engine
        # accepts is pinned here, before the first probe. TODO.md item 0.
        if getattr(T, "_PROBE_DECK", None) is None:
            T._PROBE_DECK = list(self.deck)
        self.builder = builder if builder is not None else MirrorBuilder(self.deck)
        if teacher is None:
            teacher = T.UtilityTeacher(self.deck, team=0, profile=profile,
                                       seed=seed)
            teacher.set_stage(len(T.TEACHER_STAGES) - 1 if stage is None
                              else int(stage))
        self.teacher = teacher
        self.teacher.reset()
        self.period_s = 1.0 / float(decision_hz)
        if tappable is None:
            from live.actuator import engine_row_is_tappable  # noqa: PLC0415
            tappable = engine_row_is_tappable
        self._tappable = tappable
        self.stats = TeacherStats()
        self.last_gaps = None
        self.drift_s = 0.0
        """Wall time minus teacher time since the match began: how far the
        teacher's one-second-per-decision plan clock has fallen behind."""
        self._match_start = None
        self._last_now = None
        self._prev_opp_bodies: dict[int, int] = {}
        self._was_in_game = False

    # -- lifecycle --------------------------------------------------------
    def on_screen_change(self, in_game: bool) -> None:
        """A new match starts the teacher's plan and the drift clock afresh."""
        if in_game and not self._was_in_game:
            self.reset_match()
        self._was_in_game = in_game

    def reset_match(self) -> None:
        self.teacher.reset()
        self._match_start = None
        self._last_now = None
        self._prev_opp_bodies = {}
        self.drift_s = 0.0

    # -- the decision -----------------------------------------------------
    def decide(self, gs, ready, now: float) -> Decision:
        self.stats.decisions += 1
        drift_tag = self._track_cadence(now)

        t0 = time.perf_counter()
        env, gaps = self.builder.build(gs)
        self.stats.build_ms.append((time.perf_counter() - t0) * 1000.0)
        self.last_gaps = gaps
        self._note_opponent_activity(self.builder.last_opp_bodies)

        if not gaps.hand_set:
            # Do not even let the teacher plan: a combo committed against a
            # misaligned hand would be offered again next second.
            self.stats.hand_refused += 1
            self.stats.holds += 1
            return Decision(None, DEFAULT_TILE,
                            f"hold: hand not set ({gaps.hand_reason}){drift_tag}")

        t1 = time.perf_counter()
        try:
            obs = env.get_observation_for_team(0)
            slot, x, y = self.teacher.act(env, obs)
        except Exception as exc:                            # noqa: BLE001
            self.stats.teacher_errors += 1
            self.stats.holds += 1
            return Decision(None, DEFAULT_TILE,
                            f"hold: teacher raised {type(exc).__name__}: "
                            f"{exc}{drift_tag}")
        finally:
            self.stats.teacher_ms.append((time.perf_counter() - t1) * 1000.0)

        slot = int(slot)
        if slot >= self._T.HAND_SIZE or slot < 0:
            self.stats.holds += 1
            return Decision(None, DEFAULT_TILE, f"teacher holds{drift_tag}")
        tile = (int(x), int(y))
        if ready is not None and slot not in set(ready):
            self.stats.not_ready += 1
            self.stats.holds += 1
            return Decision(None, tile, f"hold: slot {slot} not ready on "
                                        f"screen{drift_tag}")
        if not self._tappable(tile[1]):
            self.stats.untappable += 1
            self.stats.holds += 1
            return Decision(None, tile, f"hold: row {tile[1]} not "
                                        f"tappable{drift_tag}")
        self.stats.plays += 1
        return Decision(slot, tile,
                        f"teacher {self.teacher.last_kind} slot {slot} card "
                        f"{gs.my_hand[slot]} -> {tile}{drift_tag}")

    # -- couplings to the teacher ----------------------------------------
    def _track_cadence(self, now: float) -> str:
        """Measure how far the teacher's plan clock lags the wall clock."""
        tag = ""
        if self._match_start is None:
            self._match_start = now
            self._teacher_decisions = 0
        else:
            interval = now - self._last_now
            if interval > self.period_s + DRIFT_TOLERANCE_S:
                self.stats.drift_events += 1
                tag = f" [drift {interval:.2f}s]"
        self._teacher_decisions += 1
        self._last_now = now
        self.drift_s = ((now - self._match_start)
                        - (self._teacher_decisions - 1) * self.period_s)
        return tag

    def _note_opponent_activity(self, opp_bodies: dict[int, int]) -> None:
        """Stand in for the opponent elixir spend the mirror cannot carry.

        `UtilityTeacher._observe_opponent_spend` counts decisions in which the
        engine's opponent spend did not rise. Clearing its last-seen value makes
        it treat this decision as one where the opponent spent; leaving it
        lets the idle count grow. A card on the board that was not there (or
        more of it) last decision is the spend perception can actually see.
        """
        active = any(n > self._prev_opp_bodies.get(cid, 0)
                     for cid, n in opp_bodies.items())
        self._prev_opp_bodies = dict(opp_bodies)
        if active:
            self.teacher._opp_spent_seen = None
