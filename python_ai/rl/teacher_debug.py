"""Capturing the teacher's candidate rollouts for `web/viewer.html`.

Records each candidate's placement, score and margin, and for the top few the
predicted board at the end of the horizon, for the viewer's "if I play here,
this is what happens" list.

A subclass rather than a flag: the action always comes from `super().act()`, so
capture cannot change what the teacher plays (`tests/test_teacher_debug.py`).

Costly (a bare rollout is 0.31 ms; reading a predicted board back is ~32 ms, a
`save_log` file round-trip), so `top_k` bounds it: ~40 s for a 360-decision
replay. Never used by the trainers.

The predicted board comes from `ClashEnv::snapshot()`, which copies everything
except the replay logger; a snapshot's log therefore holds only its own rollout
ticks, in the schema the viewer already renders.
"""
import json
import os
import tempfile

from python_ai.advisors import tactics
from python_ai.opponents.teacher import HAND_SIZE, NOOP, UtilityTeacher

#: Candidates that get a predicted board; the rest record only their score.
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
        # Take the action from the parent first. `pending` is saved beforehand
        # because act() mutates it, and the candidates must be re-derived
        # against the state the parent saw.
        pre_pending = self.pending
        pre_ticks = self.pending_ticks
        action = super().act(env, obs_own)

        rec = {"team": self.team, "kind": "rollout",
               "horizonTicks": int(self.rollout_ticks()),
               "chosenIndex": -1, "held": False,
               "baseline": None, "candidates": []}

        if self.horizon_ticks <= 0:
            # Rung 0 is rules-only: no rollouts, so no scores to record.
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

            # The no-op board is what scores are defined against (it scores
            # exactly 0.0).
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

            # Holding (every candidate below its margin) is a real outcome of
            # the ranking; the epsilon branch ignored it, which makes the
            # candidate list misleading.
            rec["held"] = bool(action[0] == HAND_SIZE)
            if rec["chosenIndex"] == -1 and not rec["held"] and self.epsilon > 0.0:
                rec["kind"] = "epsilon"
        finally:
            self.pending, self.pending_ticks = saved

        self.debug_records.append(rec)
        return action


def capturing_like(teacher, top_k=DEFAULT_TOP_K):
    """A `CapturingTeacher` configured identically to an existing teacher.

    Also copies the live state (`rng`, `profile`, `lane_bias`, `cycle`):
    `reset()` redraws profile and lane bias, so a freshly built teacher would
    be a different opponent.
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

    Returns the `CapturingTeacher`, or None if the opponent is not a teacher.
    """
    teacher = getattr(env, "teacher", None)
    if teacher is None:
        return None
    env.teacher = capturing_like(teacher, top_k=top_k)
    return env.teacher


def attach_teacher_debug(replay_path, records, skip_frames=10):
    """Merge decision records into a saved replay, stored once at top level.

    Each record carries up to `top_k + 1` boards, so stamping it onto every
    tick (as `annotate_replay_with_agent_info` does) multiplied the file size.
    The viewer reads `decisions[floor(tickIndex / skipFrames)]`.
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
