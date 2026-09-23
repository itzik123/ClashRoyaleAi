"""The side null: is team 0 favoured by the board itself?

Runs two identical teachers against each other; team 0 should score ~0.50. It
catches faults every training metric misses, such as the team-1 observation
once being displaced by a row. Re-run after any change to the observation, the
board geometry or `stepSelfPlay`.

For a valid null: both sides must be the same policy with the profile pinned
(an unpinned reset() redraws it, making two different opponents); read the
share of decided games (draws dilute toward 0.5); and use several hundred games
(at n=30 the 95% band is about +/-0.18).

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
        # Pin the env-side teacher too, or the two sides stop being a mirror.
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
