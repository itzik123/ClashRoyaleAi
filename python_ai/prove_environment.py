"""Does THIS environment price the win condition positively? Zero training.

    python_ai/venv/Scripts/python.exe python_ai/prove_environment.py --n 120

THE FALSIFIER CLAUDE.md ASKS FOR, WITHOUT A NETWORK
----------------------------------------------------
The 1.5x hypothesis says a permanent opponent-elixir multiplier suppresses
punish cards, because a punish window lasts about `answer_cost / (m * r)` while
the counter-cost of our spend scales with `m`. Every previous test of it ran a
POLICY through the environment, which confounds "the environment prices this
badly" with "this particular net cannot execute it".

This harness removes the policy entirely. Both sides are the deterministic
`UtilityTeacher`, and the ONLY thing that differs between arms is what the
team-0 teacher does with its win condition.

THE ARM DESIGN, AND WHY THE OBVIOUS CONTROL IS WRONG
-----------------------------------------------------
The obvious arm B is "never play the win condition". It is CONFOUNDED. A card
that is never played never leaves the hand, so banning it also permanently
clogs one of four hand slots -- arm B would then lose at every multiplier, for
a reason that has nothing to do with the economy under test, and that would read
as support for the hypothesis while measuring hand mechanics.

So the arms are:

    attack   the win condition goes to a bridge (the real rule)
    cycle    it is still drawn, still played, still costs 4 elixir and still
             rotates the hand -- but it is placed in our own back half, where
             CLAUDE.md measures it at ~3 enemy tower damage against 535.6 at the
             bridge.

Identical spend, identical cycle, identical hand occupancy. The only variable is
WHERE the card lands, which is exactly and only what the hypothesis is about.
`--include-ban` runs the confounded arm too, reported separately and labelled.

WHAT THE RESULT MEANS
---------------------
    attack beats cycle at 1.0x, and the gap SHRINKS as m rises
        -> hypothesis supported; the new environment prices the win condition
           correctly and the pivot is justified.
    no gap at any multiplier
        -> the win condition is unviable in this engine's physics regardless of
           economy. The honest response is to stop rehabilitating it.
    attack beats cycle equally at every multiplier
        -> the multiplier is not what suppressed it; look elsewhere before
           rebuilding the curriculum around this.

TWO INSTRUMENTS, BECAUSE ONE CANNOT ANSWER BOTH QUESTIONS
----------------------------------------------------------
`--opponent teacher` (the default) is the SYMMETRIC test: both sides are equally
strong bots at whatever multiplier is set. It answers "in a fair environment,
does attacking beat cycling?" -- which is the question the pivot rests on. It is
useless above 1.0x: measured at n=12, a mirror-strength opponent given 1.25x or
1.5x wins EVERY game and both arms score 0.000, so the delta is pinned at the
FLOOR by the opponent rather than by the treatment.

`--opponent heuristic` is the DOSE-RESPONSE test: team 0 is the teacher, team 1
is the C++ HeuristicOpponent at multiplier m, through `env.step()`. That is the
same setup every historical number in CLAUDE.md used, so the sweep is comparable
to the SMART-forced A/B it replaces -- with the policy confound removed, since
both arms are now the same deterministic bot.

CEILING AND FLOOR CAVEATS
--------------------------
A delta is only interpretable if the arms are off the rails at BOTH ends. If
`attack` wins >= 0.95 the row is pinned by a ceiling (this is what voided the
original test's 1.00x row); if BOTH arms sit <= 0.05 it is pinned by a floor.
Either way the row is printed as VOID rather than dropped -- deleting a void row
after seeing it is how a saturated arm gets mistaken for a result.

Both teachers are seeded per pairing and every arm starts from the SAME
`snapshot()` of one `reset()`, so the shuffled hands are bit-identical across
arms and the comparison is paired.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from teacher import TEACHER_STAGES, UtilityTeacher  # noqa: E402

CE = E.ClashRoyaleEnv
MAX_STEPS = 400


def duel(env, t0, t1):
    """Teacher t0 (team 0) against teacher t1 (team 1). Returns t0's score."""
    t0.reset()
    t1.reset()
    for _ in range(MAX_STEPS):
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        if env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10).done:
            break
    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def duel_heuristic(env, t0):
    """Teacher t0 (team 0) against the C++ HeuristicOpponent, which runs inside
    `env.step()`. Returns t0's score."""
    t0.reset()
    for _ in range(MAX_STEPS):
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t0.act(env, o0)
        if env.step(slot, x, y, 10).done:
            break
    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def arm(mode, stage, seed, base, multiplier, opponent):
    """One episode of `mode` for team 0, from a bit-identical opening."""
    env = base.snapshot()
    env.set_opponent_elixir_multiplier(multiplier)
    t0 = UtilityTeacher(DEFAULT_DECK, team=0, seed=seed, wincon_mode=mode)
    t0.set_stage(stage)
    if opponent == "heuristic":
        return duel_heuristic(env, t0)
    t1 = UtilityTeacher(DEFAULT_DECK, team=1, seed=seed + 7777)
    t1.set_stage(stage)
    return duel(env, t0, t1)


def paired(diffs):
    """(mean, lo, hi, better, worse, p) -- bootstrap CI plus an exact sign test.

    Reported together because they answer different questions and this project
    has had them disagree: the CI is about the average size of the effect, the
    sign test about how often it points the same way.
    """
    from math import comb
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    better = int((d > 0).sum())
    worse = int((d < 0).sum())
    n = better + worse
    if n == 0:
        return d.mean(), lo, hi, better, worse, 1.0
    k = min(better, worse)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return d.mean(), lo, hi, better, worse, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5])
    ap.add_argument("--opponent", choices=["teacher", "heuristic"],
                    default="teacher")
    ap.add_argument("--include-ban", action="store_true",
                    help="also run the CONFOUNDED never-play arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    modes = ["attack", "cycle"] + (["ban"] if args.include_ban else [])

    print(f"\ndeck {DEFAULT_DECK}")
    print(f"teacher stage {args.stage} {TEACHER_STAGES[args.stage]}")
    print(f"{args.n} paired episodes per multiplier, "
          f"teacher vs {args.opponent}\n")
    print(f"{'opp elixir':>10} {'attack':>8} {'cycle':>8} {'delta':>9} "
          f"{'95% CI':>20} {'better/worse':>13} {'p':>10}")
    print("-" * 84)

    rows = []
    for m in args.multipliers:
        scores = {k: [] for k in modes}
        for i in range(args.n):
            root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
            root.reset()
            base = root.snapshot()
            for mode in modes:
                scores[mode].append(arm(mode, args.stage, args.seed + i,
                                        base, m, args.opponent))
        a = np.asarray(scores["attack"])
        c = np.asarray(scores["cycle"])
        mean, lo, hi, better, worse, p = paired(a - c)
        ceiling = a.mean() >= 0.95
        floor = a.mean() <= 0.05 and c.mean() <= 0.05
        void = ceiling or floor
        flag = ("  VOID (attack arm at ceiling)" if ceiling else
                "  VOID (both arms at floor)" if floor else "")
        print(f"{m:>10.2f} {a.mean():>8.3f} {c.mean():>8.3f} {mean:>+9.4f} "
              f"  [{lo:>+7.4f}, {hi:>+7.4f}] {better:>6}/{worse:<6} {p:>10.2}"
              f"{flag}")
        rows.append((m, a.mean(), c.mean(), mean, lo, hi, p, void))
        if "ban" in scores:
            b = np.asarray(scores["ban"])
            bm, blo, bhi, _, _, bp = paired(a - b)
            print(f"{'':>10} {'':>8} ban={b.mean():>6.3f} {bm:>+9.4f} "
                  f"  [{blo:>+7.4f}, {bhi:>+7.4f}] {'':>13} {bp:>10.2}"
                  f"   (CONFOUNDED: a banned card also clogs a hand slot)")

    print("\nREADING IT")
    live = [r for r in rows if not r[7]]
    if not live:
        print("  Every arm is at ceiling. Nothing is measurable here -- rerun at")
        print("  a stage where the attack arm sits nearer 0.7-0.8.")
        return
    lowest, highest = live[0], live[-1]
    print(f"  lowest multiplier tested with a live baseline: {lowest[0]:.2f}x, "
          f"delta {lowest[3]:+.4f}")
    print(f"  highest:                                       {highest[0]:.2f}x, "
          f"delta {highest[3]:+.4f}")
    if lowest[3] > 0 and lowest[4] > 0:
        print("  The win condition PAYS at the lowest live multiplier.")
        if len(live) > 1 and highest[3] < lowest[3]:
            print("  ...and the payoff SHRINKS as the opponent's economy grows, "
                  "which is\n  the hypothesis's central prediction.")
        elif len(live) > 1:
            print("  ...but it does NOT shrink as the economy grows, so the "
                  "multiplier is\n  not what was suppressing it. Do not rebuild "
                  "the curriculum on this.")
    else:
        print("  The win condition does NOT pay even at the lowest live "
              "multiplier.\n  Either the teacher is too weak for its own economy "
              "to matter, or the\n  card is unviable in this engine regardless "
              "of economy.")
    print("\n  HEADROOM NOTE: a baseline near 1.0 has less room to lose, so part "
          "of any\n  shrinkage is compression. Normalise by headroom before "
          "claiming an effect:")
    for m, am, cm, d, lo, hi, p, void in rows:
        if not void and am > 0:
            print(f"    {m:.2f}x: delta {d:+.4f} = {abs(d) / am * 100:5.1f}% "
                  f"of the attack arm's own win rate")


if __name__ == "__main__":
    main()
