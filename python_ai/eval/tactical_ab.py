"""Are the rarely played cards a deadlock, or is the policy right to decline them?

Two readings fit a card that is rarely played and whose placement head returns
one fixed cell:

  (a) valuation: the card is bad in this matchup and the frozen head is a harmless consequence;
  (b) deadlock: the head froze first (no gradient once the card stops being chosen, see `mb_placed` in rl/ppo.py), which makes the card worthless, which keeps it unplayed.

Three arms on bit-identical openings via `env.snapshot()`:

  A  control   greedy policy, untouched
  B  advisor   force the cards in, placed where `tactics` says
  C  own-cell  force the cards in at the policy's own chosen cell

C separates "the card is good" from "this cell is good". Under (a) B and C both
lose to A; under (b) B beats A and C, and C is no better than A.

Thresholds come from card stats, never tuned on the outcome.

    python_ai/venv/Scripts/python.exe python_ai/eval/tactical_ab.py --n 200
"""
import argparse
import os
import sys

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.search.search import outcome_score  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
SKIP = 10

FIREBALL_MIN_CATCH = 690.0   # a 3-cost Minions squad: 3 x 230 HP
CANNON_MIN_THREAT = 721.0    # one Musketeer
# Elixir that must remain after an override. Without it the forced arms measure
# bankruptcy, not placement; see tactics.TacticalOverride.
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
                    # Same card and moment, at the cell the policy's own head
                    # would choose.
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
    here = python_ai.PACKAGE_DIR
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
        # Paired: only discordant pairs carry information.
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
