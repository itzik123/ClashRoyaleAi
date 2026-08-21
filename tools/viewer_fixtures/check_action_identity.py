"""CapturingTeacher must choose exactly what UtilityTeacher would choose.

This is the invariant the whole capture approach rests on: a debug hook that
perturbs phase 1's opponent would silently corrupt every run taken against it.
Follows the precedent of the existing `max_combos = 0` action-identity test.

Run:
    python_ai/venv/Scripts/python.exe tools/viewer_fixtures/check_action_identity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import python_ai  # noqa: E402,F401

import numpy as np  # noqa: E402

import clash_royale_env as E  # noqa: E402
from python_ai.opponents.teacher import UtilityTeacher  # noqa: E402

from teacher_debug_capture import CapturingTeacher  # noqa: E402

DECK = [15, 6, 25, 40, 24, 72, 33, 7]


def run(make_t1, seed=23, n=120, stage=5):
    # BOTH teachers must be seeded, and this is not optional plumbing.
    # UtilityTeacher.__init__ defaults seed=None -> np.random.default_rng(None),
    # i.e. OS entropy, and reset() then REDRAWS `profile` and `lane_bias` from
    # it on purpose ("a FULLY deterministic opponent is memorizable"). Two
    # unseeded arms therefore run different scoring weights and a different
    # lane preference, and diverge for reasons that have nothing to do with
    # what is being tested. An earlier version of this file omitted the seed
    # and reported 30/120 "mismatches" that were entirely its own doing --
    # the same cross-invocation trap CLAUDE.md records for prove_combos.
    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    t0 = UtilityTeacher(DECK, team=0, seed=seed)
    t0.set_stage(stage)
    t0.reset()
    t1 = make_t1()
    t1.set_stage(stage)
    t1.reset()
    acts = []
    for _ in range(n):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        acts.append(tuple(round(float(v), 6) for v in a1))
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    return acts


def main():
    failures = 0
    # Stage 5 is the shipped rung and the one the fixtures use. Stage 3 is
    # included because it has a nonzero epsilon, which exercises the
    # exploration branch the capture code has to leave alone.
    for stage in (5, 3):
        for seed in (23, 101):
            plain = run(lambda: UtilityTeacher(DECK, team=1, seed=seed),
                        seed=seed, stage=stage)
            capt = run(lambda: CapturingTeacher(DECK, team=1, seed=seed, top_k=4),
                       seed=seed, stage=stage)
            mismatch = [i for i, (a, b) in enumerate(zip(plain, capt)) if a != b]
            print(f"stage {stage} seed {seed}: {len(plain)} vs {len(capt)} "
                  f"decisions, {len(mismatch)} mismatches")
            if len(plain) != len(capt) or mismatch:
                print(f"  ACTION DIVERGENCE at {mismatch[:10]}")
                failures += 1

    # Control: two PLAIN arms under the same seeding must also agree. If this
    # fails, the harness is broken and a passing capture result means nothing.
    a = run(lambda: UtilityTeacher(DECK, team=1, seed=77), seed=77, stage=5)
    b = run(lambda: UtilityTeacher(DECK, team=1, seed=77), seed=77, stage=5)
    ctrl = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"control (plain vs plain): {len(ctrl)} mismatches")
    if ctrl:
        print("  HARNESS IS NOT DETERMINISTIC -- capture result is meaningless")
        failures += 1

    if failures:
        print("FAIL: capture is NOT action-identical")
        sys.exit(1)
    print("PASS: capture is action-identical")


if __name__ == "__main__":
    main()
