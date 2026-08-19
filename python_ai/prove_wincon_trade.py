"""What does the defence actually PAY to answer a win condition?

    python_ai/venv/Scripts/python.exe python_ai/prove_wincon_trade.py --n 60

WHY THIS EXISTS
---------------
`prove_environment.py --mode winrate` measured something startling: at a
SYMMETRIC 1.0x economy, a teacher that commits its win condition at the bridge
scores 0.510 against one that dumps the same card in its own back half at 0.840
(n=100 paired, delta -0.330, p=5.7e-08). That is the opposite of what the
curriculum pivot predicted.

Two explanations survive that measurement and they demand opposite responses:

  A. THE TRADE IS BAD. A 4-elixir commitment is answered for less than 4, so
     every Hog is a losing exchange and a correct player never sends one. If so
     the card is unviable in this engine and no curriculum fixes it -- the
     honest response is to stop rehabilitating it, exactly as CLAUDE.md says.

  B. THE TRADE IS FINE AND THE TEACHER'S TIMING IS BAD. The advisor commits at
     moments that happen to be wrong, and the A/B measured an over-eager rule
     rather than the card.

This harness isolates A. It is deliberately NOT a win-rate test.

THE MEASUREMENT
---------------
Paired on identical snapshots. Arm HOG injects a Hog at the advisor's bridge
cell; arm NONE injects nothing. Both then let the defending teacher play
normally for `--horizon` decisions.

`inject` costs the attacker NO elixir, which is the point: it removes our own
spend from the ledger so the number that comes back is purely what the DEFENCE
had to pay, plus what it failed to prevent. The attacker's 4 elixir is then
charged explicitly at the end, once, where it can be seen.

    defence cost  = (elixir team 1 spent WITH the hog) - (WITHOUT it)
    damage        = enemy tower damage the hog caused

A win condition is a positive trade when the defence pays more than the 4 elixir
it cost, or when it pays less but concedes enough tower damage to compensate.
Reporting both halves separately is the point -- a single "value" number would
hide which of the two is failing.

UNOPPOSED CONTROL, measured 2026-08-19 and worth stating because it rules out
the simplest story: a lone Hog injected at either bridge on an EMPTY board deals
2536 tower damage (a full Princess Tower) and dies at tick 190. So the card is
not structurally weak in this engine, and the enemy King firing from tick 0
(`Tower.h` gives it no activation condition, unlike the real game) is not by
itself what suppresses it. Whatever goes wrong, goes wrong once the position is
DEFENDED.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
import tactics  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from teacher import UtilityTeacher  # noqa: E402

CE = E.ClashRoyaleEnv


def play_on(env, defender, steps):
    """Only the DEFENDER acts. The attacker deliberately does nothing further,
    so the ledger measures the cost of answering ONE win condition rather than
    the cost of answering a whole push."""
    for _ in range(steps):
        if env.is_game_over():
            break
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a1 = defender.act(env, o1)
        if env.step_self_play(-1, 0.0, 0.0, a1[0], a1[1], a1[2], 10).done:
            break


def one_trial(seed, stage, horizon, warmup):
    root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
    root.reset()
    # Let the position develop a little so this is not always the opening.
    warm = UtilityTeacher(DEFAULT_DECK, team=1, seed=seed)
    warm.set_stage(stage)
    warm.reset()
    for _ in range(warmup):
        o1 = np.asarray(root.get_observation_for_team(1), np.float32)
        a1 = warm.act(root, o1)
        if root.step_self_play(-1, 0.0, 0.0, a1[0], a1[1], a1[2], 10).done:
            return None
    base = root.snapshot()

    obs0 = np.asarray(base.get_observation_for_team(0), np.float32)
    x, y, _ = tactics.best_hog_cell(obs0)

    out = {}
    for label, inject in (("hog", True), ("none", False)):
        env = base.snapshot()
        if inject:
            env.inject(15, float(x), float(y), 0)
        d = UtilityTeacher(DEFAULT_DECK, team=1, seed=seed + 4242)
        d.set_stage(stage)
        d.reset()
        spent_before = env.get_elixir_spent(1)
        dmg_before = env.get_tower_damage_dealt(0)
        play_on(env, d, horizon)
        out[label] = (float(env.get_elixir_spent(1) - spent_before),
                      float(env.get_tower_damage_dealt(0) - dmg_before))
    return (out["hog"][0] - out["none"][0],      # defence elixir cost
            out["hog"][1] - out["none"][1])      # tower damage caused


def report(name, vals, unit=""):
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(v, len(v), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:<34} {v.mean():>9.2f}{unit}  95% CI "
          f"[{lo:>8.2f}, {hi:>8.2f}]")
    return v.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=30,
                    help="defender decisions after the hog lands (1 = 1 s)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    costs, dmgs = [], []
    for i in range(args.n):
        r = one_trial(args.seed + i, args.stage, args.horizon, args.warmup)
        if r is not None:
            costs.append(r[0])
            dmgs.append(r[1])

    hog_cost = float(E.get_card_info(15)["cost"])
    print(f"\ndeck {DEFAULT_DECK}   stage {args.stage}   "
          f"{len(costs)} paired trials   horizon {args.horizon}s\n")
    print("WHAT ONE WIN CONDITION AT THE BRIDGE BUYS AND COSTS")
    c_mean, c_lo, c_hi = report("defence elixir spent to answer", costs, " el")
    d_mean, d_lo, d_hi = report("enemy tower damage caused", dmgs, " hp")
    print(f"  {'our elixir spent':<34} {hog_cost:>9.2f} el   (fixed)")
    print()
    print(f"  ELIXIR TRADE      {c_mean - hog_cost:>+8.2f} elixir in our favour")
    print(f"  DAMAGE PER ELIXIR {d_mean / hog_cost:>8.1f} hp/elixir spent")
    print()
    if c_hi < hog_cost and d_mean < 200:
        print("  THE TRADE IS BAD. The defence answers a 4-elixir commitment for")
        print("  less than 4 and concedes little -- the card is a losing exchange")
        print("  in this engine, and no curriculum change repairs that.")
    elif c_lo > hog_cost:
        print("  THE TRADE IS GOOD. The defence pays more than the commitment")
        print("  cost, so the win condition is positive-value and a policy that")
        print("  never plays it is leaving elixir on the table.")
    else:
        print("  The elixir trade is roughly even; read it together with the")
        print("  damage line, which is what a win condition is actually for.")


if __name__ == "__main__":
    main()
