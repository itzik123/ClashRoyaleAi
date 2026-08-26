"""THE SIDE NULL: is team 0 favoured by the board itself?

Run two IDENTICAL policies against each other and confirm team 0 scores ~0.50.

WHY THIS HARNESS EXISTS. CLAUDE.md names this the one diagnostic that catches a
fault the ordinary metrics cannot see at all: the 2026-07-31 observation bug --
`extractObservationForTeam` mirroring the truncated row, `33 - int(y)` instead
of `int(33 - y)` -- displaced team 1's whole observation by one row, every tick,
while every training metric read healthy. It measured 0.598 here before the fix
and 0.520 after (n=400).

It is worth re-running after ANY change to the observation, the board geometry,
or `stepSelfPlay`.

WHAT MAKES IT A VALID NULL, and each of these was got wrong on the way here:

  * BOTH sides must be the same policy at the same strength. A first attempt
    used the env's own teacher against a separately-constructed one and read
    0.300, which was not an asymmetry at all.
  * The PROFILE must be pinned on both sides. `UtilityTeacher.reset()` redraws
    a random profile per match unless one was fixed at construction, so two
    unpinned teachers are not a mirror -- they are two different opponents, and
    that alone produced the 0.300.
  * Read the share of DECIDED games, not of all games. Draws are symmetric and
    dilute the estimate toward 0.5, which HIDES the very asymmetry being
    measured.
  * n matters. At n=30 the 95% band is about +/-0.18, which cannot resolve the
    0.598 this exists to catch. Use several hundred for a real verdict.

MEASURED 2026-08-26, stage 0, pinned "balanced", n=100: team-0 share of decided
games 0.510, 95% CI [0.412, 0.608], 0 draws. The null is inside the interval --
the board is symmetric.

    python_ai/venv/Scripts/python.exe -m python_ai.eval.side_null_ab --n 400
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("python_ai", 1)[0])

from python_ai.envs import gym_wrapper                       # noqa: E402
from python_ai.opponents.teacher import UtilityTeacher       # noqa: E402


def run(n, stage, profile, max_steps=400):
    """Returns (team0_wins, team1_wins, draws)."""
    w0 = w1 = draws = 0
    for ep in range(n):
        env = gym_wrapper.MicroRoyaleEnv(
            {"opponent": "teacher", "teacher_stage": stage,
             "scenario_seed": 90210 + ep})
        # Pin the env-side teacher too: an unpinned reset() redraws the profile
        # and the two sides stop being a mirror.
        env.teacher._fixed_profile = profile
        obs, _ = env.reset()
        ref = UtilityTeacher(list(gym_wrapper.DEFAULT_DECK), team=0,
                             profile=profile, seed=4241 + ep)
        ref.set_stage(stage)
        ref.reset()
        done, steps, reward = False, 0, 0.0
        while not done and steps < max_steps:
            slot, x, y = ref.act(env.game, obs)
            obs, reward, term, trunc, _ = env.step(
                {"card_index": int(slot), "target_x": float(x),
                 "target_y": float(y)})
            done = term or trunc
            steps += 1
        if reward > 0.5:
            w0 += 1
        elif reward < -0.5:
            w1 += 1
        else:
            draws += 1
    return w0, w1, draws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stage", type=int, default=0,
                    help="teacher rung; 0 is rules-only and by far the cheapest")
    ap.add_argument("--profile", default="balanced")
    args = ap.parse_args()

    w0, w1, draws = run(args.n, args.stage, args.profile)
    decided = w0 + w1
    if not decided:
        print("every game drew -- no decided games, no verdict")
        return
    rate = w0 / decided
    se = (rate * (1.0 - rate) / decided) ** 0.5
    lo, hi = rate - 1.96 * se, rate + 1.96 * se
    print(f"n={args.n} stage={args.stage} profile={args.profile}")
    print(f"team0 {w0}   team1 {w1}   draws {draws}")
    print(f"team-0 share of DECIDED games: {rate:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("the null is 0.500 -- "
          + ("INSIDE the CI (symmetric)" if lo <= 0.5 <= hi
             else "OUTSIDE the CI (ASYMMETRIC -- investigate)"))


if __name__ == "__main__":
    main()
