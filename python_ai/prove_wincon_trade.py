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
HOG = 15
ICE_GOLEM = 40


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


def one_trial(seed, stage, horizon, warmup, defender_elixir):
    """Returns {arm: (defence elixir spent, enemy tower damage caused)}.

    Arms, all injected for FREE so the attacker's own spend stays off the
    engine's ledger and is charged explicitly in the report:

        none        nothing -- the baseline everything is differenced against
        hog         a lone Hog at the advisor's bridge cell            (4 elixir)
        supported   Ice Golem at the bridge with the Hog behind it     (6 elixir)

    THE SUPPORTED ARM IS THE ONE THAT MATTERS. Real 2.6 never sends a naked win
    condition; the Ice Golem goes first so it eats the building's targeting and
    the tower shots while the Hog connects. A measurement that only ever tests a
    LONE Hog is testing a play no competent player makes, and would condemn the
    card on evidence about a strawman.
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
        # inject QUEUES the spawn -- one tick is required before it is on the
        # board at all (measured: 0.0 mass before, 0.399 after).
        env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 1)
        if defender_elixir is not None:
            # THE PUNISH WINDOW. A defender at full elixir always has the answer
            # affordable, which is the situation a punish card is specifically
            # NOT for. Setting the bar down is the only way to test the moment
            # the card actually exists for.
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    costs = {k: [] for k in ("hog", "supported")}
    dmgs = {k: [] for k in ("hog", "supported")}
    for i in range(args.n):
        r = one_trial(args.seed + i, args.stage, args.horizon, args.warmup,
                      args.defender_elixir)
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
