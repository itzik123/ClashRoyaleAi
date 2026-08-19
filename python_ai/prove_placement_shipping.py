"""Placement dynamism under the SHIPPING configuration (search ON).

`prove_placement.py` scores the placement head's PROPOSED cell at every state,
which is the right question for the head itself. This asks a different one: with
decision-time search enabled, search overrides ~12.8% of decisions, so the cells
that actually reach the board are not the head's argmax. The success criterion
is about the agent we ship, so measure what it ACTUALLY plays.

Reports, per card, over cells actually placed:
  * modal cell and modal share -- the dynamism metric. CLAUDE.md's rule: a good
    head is sharp but MOVES ITS MODE with the board; a broken one returns one
    cell regardless of it.
  * distinct cells used.
  * the x mod 4 histogram against the 27.8/27.8/22.2/22.2 null that 18 columns
    imply -- the cheap detector for the 2026-08-09 checkerboard artifact, which
    `Entropy/Placement_Measured` provably cannot see.

Modal share is read next to distinct-cell count deliberately: modal share
degenerates on a near-uniform distribution (the argmax of a flat map is
arbitrary but deterministic), so a high cell count alongside a low modal share
is what distinguishes "healthy and state-dependent" from "dissolved".
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
from policy_io import load_net  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from search_ab_test import (  # noqa: E402
    HAND_SIZE, LSTM_HIDDEN, _greedy_from_logits, _policy_head, _search_action,
)

CE = clash_royale_env.ClashRoyaleEnv
BOARD_W = 18
WATCH = {25: "Cannon", 7: "Fireball", 2: "Giant"}


class Cfg:
    def __init__(self, horizon, k_cards=3, k_cells=2, terminal_weight=10.0):
        self.horizon = horizon
        self.k_cards = k_cards
        self.k_cells = k_cells
        self.terminal_weight = terminal_weight


@torch.no_grad()
def run(net, use_search, cfg, episodes, opp_elixir, max_ticks, max_steps=400):
    placed = {cid: Counter() for cid in WATCH}
    all_x = []
    for ep in range(episodes):
        env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks)
        env.set_opponent_elixir_multiplier(opp_elixir)
        env.reset()
        hid = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
        done, steps = False, 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(
                np.asarray(env.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)
            cl, emb, sp, _, hid_next = _policy_head(net, obs_t, hid)
            gi, gx, gy, _ = _greedy_from_logits(net, obs_t, cl, emb, sp, hid_next)
            action = (gi, gx, gy)
            if use_search:
                action, _, _ = _search_action(net, env, obs_t, cl, emb, sp,
                                              hid_next, action, cfg,
                                              torch.device("cpu"))
            slot, x, y = action
            if slot < HAND_SIZE:                       # a real placement
                hand = net.hand_card_ids(obs_t)[0].tolist()
                cid = int(hand[slot])
                if cid in WATCH:
                    placed[cid][(int(x), int(y))] += 1
                all_x.append(int(x))
            hid = hid_next
            r = env.step(slot, x, y)
            done = r.done
            steps += 1
        print(f"    ep {ep + 1}/{episodes}", end="\r", flush=True)
    return placed, all_x


def report(label, placed, all_x):
    print(f"\n=== {label} ===")
    print(f"{'card':<10}{'plays':>7}{'modal cell':>13}{'modal share':>13}{'cells':>8}")
    for cid, name in WATCH.items():
        c = placed[cid]
        n = sum(c.values())
        if n == 0:
            print(f"{name:<10}{0:>7}{'--':>13}{'--':>13}{'--':>8}")
            continue
        cell, cnt = c.most_common(1)[0]
        print(f"{name:<10}{n:>7}{str(cell):>13}{cnt / n:>12.1%}{len(c):>8}")

    if all_x:
        h = Counter(x % 4 for x in all_x)
        tot = len(all_x)
        obs = np.array([h.get(i, 0) for i in range(4)], dtype=float)
        exp = np.array([5, 5, 4, 4], dtype=float) / 18.0 * tot
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        print(f"\n  x mod 4 over {tot} placements: "
              f"{[f'{v / tot:.1%}' for v in obs]}")
        print(f"  null (18 columns)              : ['27.8%', '27.8%', '22.2%', '22.2%']")
        print(f"  chi2 = {chi2:.1f} on 3 df (critical 7.81 at p=0.05) -> "
              f"{'PHASE BIAS' if chi2 > 7.81 else 'no phase bias'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights_selfplay.pth")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--max-ticks", type=int, default=3600)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    net = load_net(os.path.join(here, args.weights), torch.device("cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    cfg = Cfg(args.horizon)

    print(f"\ncollecting {args.episodes} episodes per arm, opponent {args.opp_elixir}x")
    g_placed, g_x = run(net, False, cfg, args.episodes, args.opp_elixir, args.max_ticks)
    report(f"greedy (no search)", g_placed, g_x)
    s_placed, s_x = run(net, True, cfg, args.episodes, args.opp_elixir, args.max_ticks)
    report(f"SHIPPING: search horizon={args.horizon}", s_placed, s_x)


if __name__ == "__main__":
    main()
