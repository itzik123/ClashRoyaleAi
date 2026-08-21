"""Generate replay fixtures for web/viewer.html development.

`modern.json`  -- a current-schema replay: cardMeta, cardId, result.
`legacy.json`  -- the same match with cardMeta, cardId and result STRIPPED,
                  standing in for a replay saved before those fields existed.
                  This is the graceful-degradation fixture; the viewer must
                  render it with no console error.
`teacher.json` -- a replay carrying the teacher's candidate rollouts.
`mixed.json`   -- teacher.json with deliberate damage applied, so every
                  missing-field path in the viewer is exercised.

Run:
    python_ai/venv/Scripts/python.exe tools/viewer_fixtures/make_fixtures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import python_ai  # noqa: E402,F401  -- puts the unpackaged .pyd on sys.path

import clash_royale_env as E  # noqa: E402

DECK = [15, 6, 25, 40, 24, 72, 33, 7]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web", "fixtures")


def play_match(path, seed=17, decisions=220):
    """A teacher-vs-teacher match, saved through the engine's own logger."""
    import numpy as np
    from python_ai.opponents.teacher import UtilityTeacher

    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    t0 = UtilityTeacher(DECK, team=0, seed=seed)
    t0.set_stage(5)
    t0.reset()
    t1 = UtilityTeacher(DECK, team=1, seed=seed)
    t1.set_stage(5)
    t1.reset()
    for _ in range(decisions):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    env.save_log(path)
    return path


def play_match_with_debug(path, seed=17, decisions=220, top_k=4):
    """The same match, with team 1's candidate rollouts captured."""
    import numpy as np
    from python_ai.opponents.teacher import UtilityTeacher
    # From the PACKAGE, not a local copy. This module held its own version
    # while a training run made python_ai/ unwritable; that copy is gone.
    from python_ai.rl.teacher_debug import CapturingTeacher, attach_teacher_debug

    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    t0 = UtilityTeacher(DECK, team=0, seed=seed)
    t0.set_stage(5)
    t0.reset()
    t1 = CapturingTeacher(DECK, team=1, seed=seed, top_k=top_k)
    t1.set_stage(5)
    t1.reset()
    for _ in range(decisions):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    env.save_log(path)
    attach_teacher_debug(path, t1.debug_records, skip_frames=10)
    return path


def strip_to_legacy(src, dst):
    """Remove every field added after the original replay schema."""
    with open(src) as f:
        data = json.load(f)
    data.pop("cardMeta", None)
    data.pop("result", None)
    for tick in data["ticks"]:
        for ent in tick["entities"]:
            ent.pop("cardId", None)
            ent.pop("isFlying", None)
    with open(dst, "w") as f:
        json.dump(data, f)


def make_mixed(src, dst):
    """teacher.json with deliberate damage applied.

    Every missing-field path the viewer has to survive is represented: a null
    decision, one with no candidates, one with no baseline, a candidate with no
    predicted board, a candidate with no steps, and each of the non-rollout
    decision kinds. The viewer must render all of them with no console error
    and no empty framed region.
    """
    with open(src) as f:
        data = json.load(f)
    decisions = data.get("teacherDebug", {}).get("decisions", [])
    for i, td in enumerate(decisions):
        if td is None:
            continue
        if i % 3 == 0:
            decisions[i] = None                      # no decision this window
        elif i % 11 == 1:
            td["candidates"] = []                    # decided, nothing listed
        elif i % 11 == 3:
            td["baseline"] = None                    # no baseline board
        elif i % 11 == 5 and td.get("candidates"):
            td["candidates"][0].pop("final", None)   # top candidate, no board
        elif i % 11 == 7 and td.get("candidates"):
            td["candidates"][0]["steps"] = []        # candidate with no steps
        elif i % 11 == 9:
            td["kind"] = "rules_only"
        elif i % 11 == 10:
            td["kind"] = "epsilon"
    with open(dst, "w") as f:
        json.dump(data, f)


def main():
    os.makedirs(OUT, exist_ok=True)
    modern = os.path.join(OUT, "modern.json")
    play_match(modern)
    strip_to_legacy(modern, os.path.join(OUT, "legacy.json"))

    teacher = os.path.join(OUT, "teacher.json")
    play_match_with_debug(teacher)
    make_mixed(teacher, os.path.join(OUT, "mixed.json"))

    for name in ("modern.json", "legacy.json", "teacher.json", "mixed.json"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            print(f"{name}: {os.path.getsize(p) / 1024:.0f} KB")

    # Trimmed copies for automated browser checks -- the full teacher.json is
    # ~3 MB, which a throttled automation renderer cannot parse in time.
    import make_small
    make_small.main()


if __name__ == "__main__":
    main()
