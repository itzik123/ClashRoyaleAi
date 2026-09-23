"""What does the defence actually pay to answer a win condition?

    python_ai/venv/Scripts/python.exe python_ai/eval/prove_wincon_trade.py --n 60

Paired on identical snapshots: an attacking arm injects a push at the advisor's
bridge cell, arm NONE injects nothing, and the defending teacher then plays
normally for `--horizon` decisions. Injection is free, so our own spend stays
off the ledger and what comes back is purely what the defence paid and failed
to prevent; the attacker's cost is charged explicitly in the report.

    defence cost  = elixir team 1 spent with the push - without it
    damage        = enemy tower damage the push caused

A win condition is a positive trade when the defence pays more than it cost, or
pays less but concedes enough damage. Both halves are reported separately so
the failing one is visible.
"""
import argparse
import os
import sys

import numpy as np

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.opponents.teacher import UtilityTeacher  # noqa: E402

CE = E.ClashRoyaleEnv
HOG = 15
ICE_GOLEM = 40
# What actually answers a Hog in this deck.
CHEAP_ANSWERS = (24, 72)   # Skeletons, Ice Spirit


def play_on(env, defender, steps):
    """Only the defender acts, so the ledger measures answering one push, not a
    sustained attack.
    """
    for _ in range(steps):
        if env.is_game_over():
            break
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a1 = defender.act(env, o1)
        if env.step_self_play(-1, 0.0, 0.0, a1[0], a1[1], a1[2], 10).done:
            break


def one_trial(seed, stage, horizon, warmup, defender_elixir,
              no_cheap_answer=False):
    """{arm: (defence elixir spent, enemy tower damage caused)}.

    Arms, all injected for free:

        none        nothing; the baseline every arm is differenced against
        hog         a lone Hog at the advisor's bridge cell            (4 elixir)
        supported   Ice Golem at the bridge with the Hog behind it     (6 elixir)

    The supported arm is the one that matters: real 2.6 never sends a naked win
    condition, and a lone-Hog measurement condemns the card on a play no
    competent player makes.
    """
    root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
    root.reset()
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

    arms = {
        "none": [],
        "hog": [(HOG, 0.0)],
        "supported": [(ICE_GOLEM, 0.0), (HOG, -1.5)],
    }
    out = {}
    for label, spawns in arms.items():
        env = base.snapshot()
        for cid, dy in spawns:
            env.inject(int(cid), float(x), float(y + dy), 0)
        # inject queues the spawn; it is on the board one tick later.
        env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 1)
        if no_cheap_answer:
            # The cycle test: force the 1-cost answers out of the defender's
            # hand, the window a cycle deck plays for, to measure what that
            # window is worth.
            hand = list(env.get_hand_for_team(1))
            keep = [c for c in DEFAULT_DECK if c not in CHEAP_ANSWERS]
            want = [c for c in hand if c not in CHEAP_ANSWERS]
            for c in keep:
                if len(want) >= len(hand):
                    break
                if c not in want:
                    want.append(c)
            # set_hand_for_team returns False on an invalid hand instead of
            # raising; unchecked, the arm would measure nothing.
            if not env.set_hand_for_team(1, want[:len(hand)]):
                return None
        if defender_elixir is not None:
            # The punish window: a defender at full elixir always has the
            # answer affordable.
            env.set_elixir_for_team(1, float(defender_elixir))
        d = UtilityTeacher(DEFAULT_DECK, team=1, seed=seed + 4242)
        d.set_stage(stage)
        d.reset()
        spent_before = env.get_elixir_spent(1)
        dmg_before = env.get_tower_damage_dealt(0)
        play_on(env, d, horizon)
        out[label] = (float(env.get_elixir_spent(1) - spent_before),
                      float(env.get_tower_damage_dealt(0) - dmg_before))
    return out


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
                    help="defender decisions after the push lands (1 = 1 s)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--defender-elixir", type=float, default=None,
                    help="force the defender's bar (the PUNISH WINDOW); "
                         "omit to leave it wherever the match put it")
    ap.add_argument("--no-cheap-answer", action="store_true",
                    help="force the 1-cost answers OUT of the defender's hand "
                         "-- the cycle window a 2.6 deck plays for")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    costs = {k: [] for k in ("hog", "supported")}
    dmgs = {k: [] for k in ("hog", "supported")}
    for i in range(args.n):
        r = one_trial(args.seed + i, args.stage, args.horizon, args.warmup,
                      args.defender_elixir, args.no_cheap_answer)
        if r is None:
            continue
        for k in costs:
            costs[k].append(r[k][0] - r["none"][0])
            dmgs[k].append(r[k][1] - r["none"][1])

    n = len(costs["hog"])
    committed = {"hog": float(E.get_card_info(HOG)["cost"]),
                 "supported": float(E.get_card_info(HOG)["cost"]
                                    + E.get_card_info(ICE_GOLEM)["cost"])}
    dstr = ("match state" if args.defender_elixir is None
            else f"forced to {args.defender_elixir:.1f}")
    if args.no_cheap_answer:
        dstr += ", cheap answers cycled OUT"
    print(f"\ndeck {DEFAULT_DECK}   stage {args.stage}   {n} paired trials   "
          f"horizon {args.horizon}s   defender elixir: {dstr}\n")
    print("WHAT A PUSH AT THE BRIDGE BUYS AND COSTS, vs not sending one")
    print(f"  {'arm':<12} {'we commit':>10} {'they spend':>12} "
          f"{'trade':>8} {'tower dmg':>11} {'dmg/elixir':>11}")
    results = {}
    for k in ("hog", "supported"):
        c = np.asarray(costs[k], dtype=np.float64)
        d = np.asarray(dmgs[k], dtype=np.float64)
        trade = c.mean() - committed[k]
        print(f"  {k:<12} {committed[k]:>10.1f} {c.mean():>12.2f} "
              f"{trade:>+8.2f} {d.mean():>11.1f} {d.mean() / committed[k]:>11.1f}")
        results[k] = (c, d, trade)

    print()
    ch, dh, _ = results["hog"]
    cs, ds, _ = results["supported"]
    m, lo, hi = report("supported - lone, tower damage", ds - dh, " hp")
    m2, lo2, hi2 = report("supported - lone, defence elixir", cs - ch, " el")
    print()
    if lo > 0:
        print("  ESCORTING THE WIN CONDITION HELPS. A lone Hog is a strawman "
              "and any\n  verdict on the card has to be taken from the "
              "supported arm.")
    elif hi < 0:
        print("  Escorting makes it WORSE -- the extra 2 elixir buys negative "
              "damage.")
    else:
        print("  Escorting resolves no difference in damage; read the trade "
              "column,\n  where the supported arm still commits 2 more elixir "
              "for it.")


if __name__ == "__main__":
    main()
