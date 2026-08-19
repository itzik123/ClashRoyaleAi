"""Is the dead-card deadlock CAUSAL, or is the policy right to decline them?

THE QUESTION
------------
Cannon / Fireball / Giant are played on ~2% of plays and their placement head
returns one fixed cell -- (11,0), our own back row -- in 54-91% of states. Two
readings fit that:

  (a) VALUATION. The cards are genuinely bad in this matchup, the policy learned
      that correctly, and the frozen head is a harmless consequence of a card
      that never gets sampled.
  (b) DEADLOCK. The head froze first (it gets zero gradient once the card stops
      being chosen -- see train.py's `mb_placed`), which MAKES the card
      worthless, which keeps it unplayed, which keeps the head frozen.

These predict opposite things, so one experiment separates them. Three arms,
each pair handed a bit-identical opening by `env.snapshot()`:

  A  control   -- greedy policy, untouched
  B  advisor   -- force the dead cards in, placed where `tactics` says
  C  own-cell  -- force the dead cards in at the POLICY'S OWN chosen cell

C is the control that makes B interpretable, and CLAUDE.md demands it: a
forced-placement A/B that omits it cannot tell "the card is good" from "this
particular cell is good". Under (a) both B and C lose to A. Under (b) B beats
both A and C, and C is no better than A.

Thresholds are set A PRIORI from card stats, never tuned on the outcome:
Fireball fires when the advisor sees at least a 3-cost squad's worth of HP
(Minions are 3x230=690), the Cannon goes down when at least a Musketeer's worth
(721) is approaching. Tuning these on win rate would be optional stopping.

    python_ai/venv/Scripts/python.exe python_ai/tactical_ab.py --n 200
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
import tactics  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from policy_io import load_net  # noqa: E402
from search_ab_test import LSTM_HIDDEN, outcome_score  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
SKIP = 10

FIREBALL_MIN_CATCH = 690.0   # a 3-cost Minions squad: 3 x 230 HP
CANNON_MIN_THREAT = 721.0    # one Musketeer
# Elixir that must REMAIN after an override plays. The ungated first run of this
# experiment forced ~27 extra elixir of cards per episode into an agent already
# spending ~105 against ~98 of income, and both forced arms lost -- it was
# measuring bankruptcy, not placement. See tactics.TacticalOverride.
RESERVE = 4.0


@torch.no_grad()
def play(net, env, device, mode):
    """mode: 'control' | 'advisor' | 'owncell'."""
    hx = torch.zeros(1, LSTM_HIDDEN, device=device)
    cx = torch.zeros(1, LSTM_HIDDEN, device=device)
    obs = env.get_observation_for_team(0)
    reward, steps, done = 0.0, 0, False
    forced = {"cannon": 0, "fireball": 0}
    override = tactics.TacticalOverride(
        net._placement_legal.numpy(),
        reserve=RESERVE,
        fireball_min_catch=FIREBALL_MIN_CATCH,
        cannon_min_cover=CANNON_MIN_THREAT)

    while not done and steps < 400:
        t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        mask = net.affordability_mask(t)
        feats, embeds, sp = net.extract_features(t)
        logits, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)

        slot = int(logits.argmax(-1).item())
        place = net.placement_given_card(hx2, embeds, torch.tensor([slot], device=device), t, sp)
        cell = int(place.argmax(-1).item())
        x, y = float(cell % tactics.BOARD_W), float(cell // tactics.BOARD_W)

        if mode != "control":
            hand = net.hand_card_ids(t)[0].tolist()
            aff = mask[0].tolist()
            new_slot, nx, ny = override(obs, hand, aff, (slot, x, y))
            if new_slot != slot:
                forced["fireball" if hand[new_slot] == tactics.FIREBALL_ID
                       else "cannon"] += 1
                slot = new_slot
                if mode == "advisor":
                    x, y = nx, ny
                else:
                    # own-cell control: same card, same moment, but the cell the
                    # POLICY's own frozen head would have chosen. Isolates the
                    # placement from the decision to play at all.
                    p = net.placement_given_card(
                        hx2, embeds, torch.tensor([slot], device=device), t, sp)
                    c = int(p.argmax(-1).item())
                    x, y = float(c % tactics.BOARD_W), float(c // tactics.BOARD_W)

        hx, cx = hx2, cx2
        r = env.step(slot, x, y, SKIP)
        obs, reward, done = r.observation, float(r.reward), r.done
        steps += 1
    return reward, forced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    here = os.path.dirname(os.path.abspath(__file__))
    net = load_net(os.path.join(here, args.net), device)

    modes = ["control", "advisor", "owncell"]
    scores = {m: [] for m in modes}
    forced_tot = {m: {"cannon": 0, "fireball": 0} for m in modes}

    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.set_opponent_elixir_multiplier(args.opp_elixir)
        root.reset()
        base = root.snapshot()
        for m in modes:
            r, f = play(net, base.snapshot(), device, m)
            scores[m].append(outcome_score(r))
            for k in f:
                forced_tot[m][k] += f[k]
        if (i + 1) % 20 == 0:
            line = "  ".join(f"{m} {np.mean(scores[m]):.3f}" for m in modes)
            print(f"  pair {i+1:>4}: {line}", flush=True)

    print(f"\n{args.n} paired openings, {args.net}, opponent {args.opp_elixir}x elixir\n")
    a = np.array(scores["control"])
    for m in modes:
        s = np.array(scores[m])
        print(f"  {m:<9} win rate {s.mean():.3f}   forced: "
              f"cannon {forced_tot[m]['cannon']:>4}  fireball {forced_tot[m]['fireball']:>4}")

    for m in ["advisor", "owncell"]:
        b = np.array(scores[m])
        d = b - a
        # paired: only discordant pairs carry information
        better = int((d > 0).sum())
        worse = int((d < 0).sum())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"\n  {m} - control: {d.mean():+.4f}  95% CI "
              f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]")
        print(f"    discordant: {better} better / {worse} worse / "
              f"{len(d)-better-worse} tied")
        if better + worse:
            from math import comb
            n_, k_ = better + worse, min(better, worse)
            p = sum(comb(n_, i) for i in range(k_ + 1)) * 2 / (2 ** n_)
            print(f"    exact McNemar p = {min(1.0, p):.4g}")


if __name__ == "__main__":
    main()
