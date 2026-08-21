"""Teacher rollout capture for the viewer's Simulation View.

Lives OUTSIDE python_ai/ on purpose: a training run is live and
python_ai/opponents/teacher.py must not be edited. Importing it is harmless --
a running process holds its own copy -- but writing to it would be picked up by
the processes train.py spawns. When the run finishes this becomes
python_ai/rl/teacher_debug.py; see the plan's "Deferred" section.

COST: capturing a candidate's predicted board costs ~32 ms (a save_log file
round-trip; the cost is disk sync, not parsing -- a tail-read optimisation was
tried and did not help) against a 0.31 ms bare rollout. So only the chosen
candidate and the next `top_k - 1` are captured, and nothing is captured unless
a CapturingTeacher is explicitly constructed.

MEASURED, over 1,080 decisions of contested teacher-vs-teacher play: 3.60
candidates per decision on average (median 3, p90 9, max 13), with 22% of
decisions having none at all and 33% having more than four. So the top_k cap
binds about a third of the time.
"""
import json
import os
import tempfile

from python_ai.advisors import tactics
from python_ai.opponents.teacher import HAND_SIZE, NOOP, UtilityTeacher


def _final_entities(env_snapshot, path):
    """The rollout's final board, as the engine's own logger sees it.

    ClashEnv::snapshot() copies everything EXCEPT the replay logger, which
    starts empty and stays live -- so save_log on a snapshot writes ONLY the
    rollout's own ticks, in the schema the viewer already renders, with the
    parent match untouched.
    """
    env_snapshot.save_log(path)
    with open(path) as f:
        data = json.load(f)
    if not data.get("ticks"):
        return []
    return data["ticks"][-1]["entities"]


class CapturingTeacher(UtilityTeacher):
    """UtilityTeacher that records what it considered. Action-identical.

    The action always comes from `super().act()`, so the chosen play cannot
    diverge from the real teacher's no matter what the capture code does.
    `check_action_identity.py` pins that.
    """

    def __init__(self, *a, top_k=4, **kw):
        super().__init__(*a, **kw)
        self.top_k = int(top_k)
        self.debug_records = []
        self._tmp = os.path.join(tempfile.gettempdir(), f"td_{id(self)}.json")

    def _capture_board(self, env, cand):
        s = env.snapshot()
        self.execute_steps(s, cand, self.rollout_ticks())
        return _final_entities(s, self._tmp)

    def act(self, env, obs_own):
        # Delegate for the ACTION first, so the chosen play can never diverge.
        # `pending` is captured beforehand because act() mutates it and the
        # candidate set has to be re-derived against the state the parent saw.
        pre_pending = self.pending
        pre_ticks = self.pending_ticks
        action = super().act(env, obs_own)

        rec = {"team": self.team, "kind": "rollout",
               "horizonTicks": int(self.rollout_ticks()),
               "chosenIndex": -1, "held": False,
               "baseline": None, "candidates": []}

        if self.horizon_ticks <= 0:
            # Stages 0-1 take _rules_only, which never rolls out and has no
            # scores at all. Recording a score here would be inventing one.
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
            # rollout branch (every candidate fell below its margin), and it is
            # NOT the same as the exploration branch firing. Distinguishing
            # them matters: the candidate list is meaningful in the first case
            # and misleading in the second, where the parent ignored the
            # ranking entirely and picked uniformly.
            rec["held"] = bool(action[0] == HAND_SIZE)
            if rec["chosenIndex"] == -1 and not rec["held"] and self.epsilon > 0.0:
                rec["kind"] = "epsilon"
        finally:
            self.pending, self.pending_ticks = saved

        self.debug_records.append(rec)
        return action


def attach_teacher_debug(replay_path, records, skip_frames=10):
    """Merge the decision records into an already-saved replay.

    STORED ONCE AT TOP LEVEL, indexed by decision, NOT stamped onto every tick.
    `annotate_replay_with_agent_info` stamps its per-decision fields onto all
    `skip_frames` ticks of a window, which is fine for three scalars and very
    much not fine here: a record carries up to five predicted boards, so
    stamping it across a 10-tick window serialises ten identical copies and
    multiplied the fixture by 10 (110 MB -> 11 MB when this was fixed).

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
