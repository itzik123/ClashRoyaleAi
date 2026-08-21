"""Capturing the teacher's candidate rollouts for `web/viewer.html`.

WHAT THIS IS FOR. `UtilityTeacher` ranks candidates by rolling each one forward
on `env.snapshot()` and scoring the result, and until now every one of those
rollouts was thrown away the instant it produced a number. This records them:
each candidate's placement, its score, the margin it had to beat, and -- for the
top few -- the PREDICTED BOARD at the end of the horizon. The viewer renders
that as a ranked list of "if I play here, this is what I think happens".

WHY A SUBCLASS AND NOT A FLAG INSIDE UtilityTeacher.act. The action always comes
from `super().act()`, so a capture bug CANNOT change which card the teacher
plays. That matters more than it sounds: `UtilityTeacher` is phase 1's opponent,
and a debug hook that perturbed it would silently invalidate every run taken
against it. Folding capture into `act()` would have made action-identity depend
on the capture code being correct, and on a test noticing when it was not.
Delegating makes it structural. `tests/test_teacher_debug.py` pins it anyway.

The cost of that choice is one extra scoring pass per captured decision, which
is ~3.6 rollouts at 0.31 ms. That is noise beside the ~32 ms each captured board
costs, and it is only paid when capture is explicitly switched on.

COST, MEASURED. A bare candidate rollout is 0.31 ms. Reading the predicted board
back out costs ~32 ms, because `save_log` is a file round-trip and the cost is
the disk sync, not the parsing -- a tail-read optimisation was tried and did not
help. So capturing every candidate of every decision is ~100x a normal rollout,
and `top_k` exists to bound it. Over 1,080 decisions of contested
teacher-vs-teacher play there are 3.60 candidates per decision on average
(median 3, p90 9, max 13), 22% of decisions have none at all, and 33% have more
than four -- so the cap binds about a third of the time and total capture runs
~40 s for a 360-decision replay.

NEVER ENABLE THIS IN THE TRAINING LOOP. Nothing here runs unless a
`CapturingTeacher` is explicitly constructed, and the trainers do not construct
one.

HOW THE PREDICTED BOARD IS OBTAINED, since it is not obvious there is a way at
all: `ClashEnv::snapshot()` copies everything EXCEPT the replay logger, which
starts empty and stays live. So a rollout on a snapshot accumulates ONLY its own
ticks, and `save_log` on it writes a self-contained mini-replay in exactly the
schema the viewer already renders -- with the parent match untouched. No C++
change was needed and none was made.
"""
import json
import os
import tempfile

from python_ai.advisors import tactics
from python_ai.opponents.teacher import HAND_SIZE, NOOP, UtilityTeacher

#: Candidates that get a predicted board. The rest are recorded with their score
#: only. Four keeps the viewer's comparison readable and bounds the cost.
DEFAULT_TOP_K = 4


def _final_entities(env_snapshot, path):
    """The rollout's final board, as the engine's own logger sees it."""
    env_snapshot.save_log(path)
    with open(path) as f:
        data = json.load(f)
    if not data.get("ticks"):
        return []
    return data["ticks"][-1]["entities"]


class CapturingTeacher(UtilityTeacher):
    """A `UtilityTeacher` that records what it considered. Action-identical.

    Use exactly like `UtilityTeacher`; read `debug_records` afterwards and hand
    it to `attach_teacher_debug`.
    """

    def __init__(self, *a, top_k=DEFAULT_TOP_K, **kw):
        super().__init__(*a, **kw)
        self.top_k = int(top_k)
        self.debug_records = []
        self._tmp = os.path.join(tempfile.gettempdir(),
                                 "clash_teacher_debug_%d.json" % id(self))

    def _capture_board(self, env, cand):
        s = env.snapshot()
        self.execute_steps(s, cand, self.rollout_ticks())
        return _final_entities(s, self._tmp)

    def act(self, env, obs_own):
        # The ACTION first, and from the parent, so it cannot diverge. `pending`
        # is saved beforehand because act() mutates it and the candidate set has
        # to be re-derived against the state the parent actually saw.
        pre_pending = self.pending
        pre_ticks = self.pending_ticks
        action = super().act(env, obs_own)

        rec = {"team": self.team, "kind": "rollout",
               "horizonTicks": int(self.rollout_ticks()),
               "chosenIndex": -1, "held": False,
               "baseline": None, "candidates": []}

        if self.horizon_ticks <= 0:
            # Stages 0-1 take `_rules_only`, which never rolls out and has no
            # scores at all. Recording one here would be inventing it.
            rec["kind"] = "rules_only"
            self.debug_records.append(rec)
            return action

        saved = (self.pending, self.pending_ticks)
        self.pending, self.pending_ticks = pre_pending, pre_ticks
        try:
            cands = [c for c in self.candidates(env, obs_own) if c.steps]
            if not cands:
                rec["kind"] = "no_candidates"
                self.debug_records.append(rec)
                return action

            baseline = self.rollout_stats(env, NOOP)
            elixir_now = tactics.own_elixir(obs_own)
            scored = []
            for c in cands:
                sc = self.score(env, c, baseline, obs_own)
                scored.append((c, float(sc), float(self.margin_for(c, elixir_now))))
            scored.sort(key=lambda t: -t[1])

            # The no-op board is the comparator the scores are DEFINED against
            # (`score` returns exactly 0.0 for it), so the viewer shows it too.
            rec["baseline"] = {"entities": self._capture_board(env, NOOP)}
            for i, (c, sc, mg) in enumerate(scored):
                row = {
                    "steps": [{"cardId": int(st.card_id), "slot": int(st.slot),
                               "x": float(st.x), "y": float(st.y),
                               "delayTicks": int(st.delay_ticks)}
                              for st in c.steps],
                    "kind": c.kind,
                    "score": sc,
                    "margin": mg,
                    "rejected": bool(sc <= mg),
                }
                if i < self.top_k:
                    row["final"] = {"entities": self._capture_board(env, c)}
                rec["candidates"].append(row)

            chosen = [i for i, (c, sc, mg) in enumerate(scored)
                      if sc > mg and c.slot == action[0]
                      and abs(c.x - action[1]) < 1e-6
                      and abs(c.y - action[2]) < 1e-6]
            rec["chosenIndex"] = chosen[0] if chosen else -1

            # A decision that played nothing is a legitimate outcome of the
            # rollout branch -- every candidate fell below its margin -- and is
            # NOT the same as the exploration branch firing. The distinction
            # matters to a reader: the candidate list is meaningful in the first
            # case and misleading in the second, where the parent ignored the
            # ranking entirely and picked uniformly.
            rec["held"] = bool(action[0] == HAND_SIZE)
            if rec["chosenIndex"] == -1 and not rec["held"] and self.epsilon > 0.0:
                rec["kind"] = "epsilon"
        finally:
            self.pending, self.pending_ticks = saved

        self.debug_records.append(rec)
        return action


def capturing_like(teacher, top_k=DEFAULT_TOP_K):
    """A `CapturingTeacher` configured identically to an existing teacher.

    Copies the constructor-relevant configuration AND the live mutable state
    (`rng`, `profile`, `lane_bias`, `cycle`), because `reset()` REDRAWS profile
    and lane bias from the RNG on purpose -- so a naively reconstructed teacher
    is a DIFFERENT opponent, not the same one instrumented.
    """
    out = CapturingTeacher(
        list(teacher.deck), teacher.team,
        profile=teacher._fixed_profile,
        horizon_ticks=teacher.horizon_ticks,
        k_cells=teacher.k_cells,
        epsilon=teacher.epsilon,
        wincon_mode=teacher.wincon_mode,
        max_combos=teacher.max_combos,
        combo_reserve=teacher.combo_reserve,
        top_k=top_k,
    )
    out.profile = teacher.profile
    out.lane_bias = teacher.lane_bias
    out.rng = teacher.rng
    out.cycle = teacher.cycle
    out.pending = teacher.pending
    out.pending_ticks = teacher.pending_ticks
    return out


def attach_debug_capture(env, top_k=DEFAULT_TOP_K):
    """Swap a `MicroRoyaleEnv`'s teacher for a capturing one, in place.

    Returns the `CapturingTeacher`, or None if the env has no teacher (its
    opponent is the C++ heuristic or a neural league member, neither of which
    has candidate rollouts to record).
    """
    teacher = getattr(env, "teacher", None)
    if teacher is None:
        return None
    env.teacher = capturing_like(teacher, top_k=top_k)
    return env.teacher


def attach_teacher_debug(replay_path, records, skip_frames=10):
    """Merge decision records into an already-saved replay.

    STORED ONCE AT TOP LEVEL AND INDEXED, not stamped onto every tick of a
    window the way `annotate_replay_with_agent_info` stamps its three scalars.
    That difference is not stylistic: a record here carries up to `top_k + 1`
    predicted boards, and stamping it across a 10-tick window serialised ten
    identical copies and made a test fixture 11 MB against 2.97 MB.

    The viewer recovers a tick's record as
    `decisions[floor(tickIndex / skipFrames)]`, which is the same mapping the
    stamping produced, without the duplication.
    """
    with open(replay_path) as f:
        data = json.load(f)
    data["teacherDebug"] = {
        "skipFrames": int(skip_frames),
        "team": records[0]["team"] if records else None,
        "decisions": records,
    }
    with open(replay_path, "w") as f:
        json.dump(data, f)
    return replay_path
