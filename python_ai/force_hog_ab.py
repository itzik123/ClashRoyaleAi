"""Is playing the Hog Rider WORTH IT? Forced usage, paired, engine-scored.

THE QUESTION
------------
The 2.6 net plays its win condition on 0.0-0.4% of decisions. Two explanations
fit that equally well and they call for opposite responses:

  PATHOLOGY   the card head is stuck in a zero-gradient trap -- the
              win-condition reward can only pay for damage the Hog deals, the
              Hog is never played, so the term pays zero forever and cannot
              bootstrap the behaviour it is meant to reward. Fix: force
              exploration.

  VALUATION   0% is correct. Against this opponent the Hog is a bad buy and the
              policy has worked that out. Fix: nothing, and forcing it would
              make the agent worse.

This project has run exactly this experiment once before and VALUATION won:
forcing Fireball dropped win rate 97% -> 23%. So the prior is not on the side
of intervening, and the measurement has to come before the fix.

DESIGN
------
Paired on `env.snapshot()`, so both arms get a bit-identical opening -- same
shuffled hand, same heuristic lane. Opponent is the C++ HeuristicOpponent at
1.5x elixir, where CLAUDE.md records there is headroom on both sides (at 1.0x
the policy wins ~100% and both arms saturate, pinning the delta at 0 by the
opponent rather than by the treatment).

The forced arm changes ONLY WHICH CARD is played, never where: the Hog goes to
the cell the net's own placement head chose for it. That keeps the comparison
about the card's value rather than about placement quality, which prove_hog.py
already measures separately (at ep 18,013: indistinguishable from chance).

`--force-prob` below 1.0 makes this an epsilon-exploration probe rather than a
hard override, which is the shape any actual fix would take.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import clash_royale_env as CE  # noqa: E402
from gym_wrapper import DEFAULT_DECK, WIN_CONDITION_ID  # noqa: E402
from model import MicroRoyaleNet  # noqa: E402

# BOARD_WIDTH lives on the CLASS, not the module -- same binding prove_hog.py
# uses. `clash_royale_env.BOARD_WIDTH` raises AttributeError.
E = CE.ClashRoyaleEnv

LSTM_HIDDEN = 256
NOOP = 4


def load(path):
    net = MicroRoyaleNet(num_ability_slots=0)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"] if "model" in blob else blob, strict=False)
    net.eval()
    return net


@torch.no_grad()
def play(net, env, force_prob, rng, stats):
    """One episode. Returns 1.0 win / 0.0 loss / 0.5 draw."""
    obs_l = env.get_observation_for_team(0)
    hx = torch.zeros(1, LSTM_HIDDEN)
    cx = torch.zeros(1, LSTM_HIDDEN)
    for _ in range(400):
        obs = torch.from_numpy(np.asarray(obs_l, np.float32)).view(1, -1)
        feats, embeds, spatial, hires = net.extract_features_hires(obs)
        mask = net.affordability_mask(obs)
        cl, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)

        card = int(cl.argmax(1))
        ids = net.hand_card_ids(obs)[0].tolist()
        if force_prob > 0.0 and WIN_CONDITION_ID in ids:
            slot = ids.index(WIN_CONDITION_ID)
            # Only force what the engine would actually accept. Forcing an
            # unaffordable slot makes playCard return false silently, which
            # would score as "we played the Hog" while nothing reached the
            # board -- the exact silent-failure mode the affordability mask
            # was added to remove.
            if bool(mask[0, slot]) and rng.random() < force_prob:
                card = slot
                stats["forced"] += 1

        pl = net.placement_given_card(hx2, embeds, torch.tensor([card]),
                                      obs, spatial, hires_map=hires)
        c = int(pl.argmax(1))
        if card != NOOP:
            stats["plays"] += 1
            played_id = net.hand_card_ids(obs)[0].tolist()[card]
            stats["by_card"][played_id] = stats["by_card"].get(played_id, 0) + 1
        res = env.step(card, float(c % E.BOARD_WIDTH), float(c // E.BOARD_WIDTH), 10)
        obs_l = res.observation
        hx, cx = hx2, cx2
        if res.done:
            break

    stats["wincon_damage"] += float(env.get_damage_dealt_by_card(WIN_CONDITION_ID, 0))
    towers = env.get_towers_destroyed(0) if hasattr(env, "get_towers_destroyed") else None
    del towers
    r = float(res.reward)
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join(HERE, "model_weights_selfplay.pth"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--force-prob", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"win condition = card id {WIN_CONDITION_ID}")
    net = load(args.weights)
    root = CE.ClashRoyaleEnv(DEFAULT_DECK, DEFAULT_DECK, 3600)
    root.set_opponent_elixir_multiplier(args.opp_elixir)

    arms = {"baseline": 0.0, f"forced@{args.force_prob}": args.force_prob}
    scores = {k: [] for k in arms}
    stats = {k: {"forced": 0, "plays": 0, "wincon_damage": 0.0, "by_card": {}}
             for k in arms}

    for i in range(args.n):
        root.reset()
        base = root.snapshot()
        for name, prob in arms.items():
            rng = random.Random(args.seed * 100003 + i)   # same draws per pair
            scores[name].append(play(net, base.snapshot(), prob, rng, stats[name]))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n}", flush=True)

    a, b = list(arms)
    A = np.array(scores[a])
    B = np.array(scores[b])
    print("\n" + "=" * 74)
    print(f"FORCED WIN-CONDITION USAGE  --  {os.path.basename(args.weights)}, "
          f"opp {args.opp_elixir}x, n={args.n} paired")
    print("=" * 74)
    for name in arms:
        s = stats[name]
        wc = s["wincon_damage"] / max(args.n, 1)
        hog = s["by_card"].get(WIN_CONDITION_ID, 0)
        print(f"  {name:<16} win {np.mean(scores[name]):.3f}   "
              f"plays/ep {s['plays']/args.n:5.1f}   "
              f"hog plays {hog:4d} ({100*hog/max(s['plays'],1):4.1f}%)   "
              f"wincon dmg/ep {wc:7.1f}")

    delta = float(B.mean() - A.mean())
    diff = B - A
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    better = int((diff > 0).sum())
    worse = int((diff < 0).sum())
    print(f"\n  delta (forced - baseline)  {delta:+.4f}  "
          f"95% CI [{delta-1.96*se:+.4f}, {delta+1.96*se:+.4f}]")
    print(f"  {better} better / {worse} worse / {len(diff)-better-worse} tied")
    if better + worse:
        from math import comb
        n = better + worse
        k = min(better, worse)
        p = sum(comb(n, j) for j in range(k + 1)) * 2 / (2 ** n)
        print(f"  exact sign test p = {min(p,1.0):.4g}")
    print("\n  A NEGATIVE delta means the policy was RIGHT to avoid the card and")
    print("  no exploration fix is warranted. A positive one means it is stuck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
