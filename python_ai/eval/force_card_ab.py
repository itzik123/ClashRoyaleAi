"""Does playing a card MORE actually win more games, on the training distribution?

WHY THIS EXISTS. The ep-59,000 gate re-run found that forcing The Log at the
policy's own placement argmax is worth **+749 tower HP** in high-opportunity
states, against only +46 HP for improving where it lands. That reads like "play
it more" -- but the probe SELECTS the 32 states most favourable to playing, so it
cannot answer whether playing it more is good on average.

The prior is against it. `force_hog_ab.py` records that forcing Fireball once
dropped win rate 97% -> 23%, and CLAUDE.md lists four separate attempts to raise
these cards by pushing the marginal, every one of which cost win rate. So this
measures before anything is changed.

DESIGN
------
Two arms, same seeds, same pool deck per trial:

    A  the policy, greedy
    B  identical, except that whenever the target card is in hand, affordable,
       and the board offers it something, it is played with probability
       `--force-prob` -- at the cell the net's OWN placement head chose

So the arms differ in WHEN the card is played and never in WHERE, which keeps the
question about selection rather than placement. `--force-prob` below 1.0 makes
this an epsilon-style nudge, the shape any real fix would take, rather than a
hard override.

PAIRED BY SEED, NOT BY SNAPSHOT. The opponent here is the UtilityTeacher on the
16-deck pool -- the distribution the finding came from -- and a raw
`env.snapshot()` copies the engine but not the Python-side teacher, so a
snapshot-paired arm would silently face the C++ HeuristicOpponent instead. Both
arms therefore construct their own env with the same seed and the same deck;
at rung 10 the teacher's epsilon is 0.00, so it is deterministic given that.

`--gate` restricts forcing to states where the card's own value map says there
is something to hit, which is the state-conditional version of the intervention.
Ungated, this is the indiscriminate version that has failed before.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.eval.measure_deck_matchups import log_catch_map  # noqa: E402
from python_ai.eval.stats import paired  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.trainers.expert_collect import PoolTeacherEnv  # noqa: E402

NOOP = E.ClashRoyaleEnv.HAND_SIZE
VALUE_MAP = {"Fireball": lambda o: tactics.spell_catch_map(o),
             "The Log": log_catch_map,
             "Cannon": lambda o: np.full((1, 1), tactics.threat_level(o), np.float32)}


@torch.no_grad()
def play(net, env, card_id, force_prob, rng, vmap, gate_at, stats,
         max_steps=400):
    hx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
    cx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
    obs = env.get_observation_for_team(0)
    res = None
    for _ in range(max_steps):
        o32 = np.asarray(obs, dtype=np.float32)
        t = torch.tensor(o32).unsqueeze(0)
        mask = net.affordability_mask(t)
        feats, embeds, spatial = net.extract_features(t)
        logits, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
        card = int(logits.argmax(dim=-1).item())

        if force_prob > 0.0:
            hand = net.hand_card_ids(t)[0].tolist()
            slot = next((k for k, c in enumerate(hand) if c == card_id), None)
            if (slot is not None and bool(mask[0, slot]) and card != slot
                    and rng.random() < force_prob):
                offered = float(vmap(o32).max()) if vmap else float("inf")
                if offered >= gate_at:
                    card = slot
                    stats["forced"] += 1

        if card == NOOP:
            px = py = 0.0
        else:
            pl = net.placement_given_card(hx2, embeds, torch.tensor([card]),
                                          t, spatial)
            pm = net.placement_mask(t, torch.tensor([card]))
            cell = int(pl.masked_fill(~pm, float("-inf")).argmax(1))
            xt, yt = net.cell_to_xy(torch.tensor([cell]))
            px, py = float(xt.item()), float(yt.item())
            stats["plays"] += 1
        res = env.step(card, px, py)
        obs, hx, cx = res.observation, hx2, cx2
        stats["steps"] += 1
        if res.done:
            break
    r = float(res.reward) if res is not None else 0.0
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--card", default="The Log")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--force-prob", type=float, default=1.0)
    ap.add_argument("--stage", type=int, default=10)
    ap.add_argument("--seed", type=int, default=770)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--gate", action="store_true",
                    help="only force where the card's value map says there is "
                         "something to hit (>= the bank's HIGH-opportunity cut)")
    ap.add_argument("--gate-at", type=float, default=0.0)
    args = ap.parse_args()

    from python_ai.opponents import deck_pool
    pool = deck_pool.load_pool()
    net = load_net(args.weights if os.path.isabs(args.weights)
                   else os.path.join(python_ai.PACKAGE_DIR, args.weights),
                   torch.device("cpu"), verbose=True)
    cid = next(c for c in gym_wrapper.DEFAULT_DECK
               if E.get_card_info(c)["name"] == args.card)
    vmap = VALUE_MAP.get(args.card) if args.gate else None
    gate_at = args.gate_at if args.gate else 0.0

    print(f"card     : {args.card} (id {cid})")
    print(f"arms     : A greedy   |   B force p={args.force_prob}"
          f"{' gated at ' + str(gate_at) if args.gate else ' ungated'}")
    print(f"opponent : UtilityTeacher rung {args.stage}, {len(pool)} pool decks")
    print(f"paired   : {args.n} trials, same seed and deck per trial\n")

    A, B = [], []
    sa = {"plays": 0, "forced": 0, "steps": 0}
    sb = {"plays": 0, "forced": 0, "steps": 0}
    for i in range(args.n):
        deck = pool[i % len(pool)]
        seed = args.seed + i
        envA = PoolTeacherEnv(args.stage, args.max_ticks, seed, deck)
        A.append(play(net, envA, cid, 0.0, np.random.default_rng(seed), None, 0.0, sa))
        envB = PoolTeacherEnv(args.stage, args.max_ticks, seed, deck)
        B.append(play(net, envB, cid, args.force_prob,
                      np.random.default_rng(seed), vmap, gate_at, sb))
        if (i + 1) % 20 == 0:
            print(f"  {i+1:4d}/{args.n}  A {np.mean(A):.3f}  B {np.mean(B):.3f}  "
                  f"forced {sb['forced']}", flush=True)

    a, b = np.array(A), np.array(B)
    pr = paired(a, b)
    print(f"\n  A greedy       {a.mean():.4f}")
    print(f"  B forced       {b.mean():.4f}")
    print(f"  paired delta   {pr.delta:+.4f}   95% CI [{pr.lo:+.4f}, {pr.hi:+.4f}]")
    print(f"  {pr.better} better / {pr.worse} worse / {pr.tied} tied"
          f"   sign test p = {pr.p:.4g}")
    print(f"  forced plays   {sb['forced']} over {sb['steps']} decisions "
          f"({sb['forced']/max(1,sb['steps']):.1%})")
    verdict = ("HELPS" if pr.delta > 0 and pr.p < 0.05 else
               "HURTS" if pr.delta < 0 and pr.p < 0.05 else "no effect resolved")
    print(f"\n  VERDICT: forcing {args.card} {verdict}")
    if pr.p >= 0.05:
        print("  (n is the binding constraint on this class of comparison -- "
              "CLAUDE.md records the control arm alone varying 0.570-0.700 "
              "across runs)")


if __name__ == "__main__":
    main()
