"""Placement dynamism under the shipping configuration (search on).

`prove_placement.py` scores the head's proposed cell. With search enabled,
search overrides some decisions, so this measures the cells that actually reach
the board, per card:
  * modal cell and modal share: a good head is sharp but moves its mode with the board
  * distinct cells used: read with modal share, since modal share degenerates on a near-uniform map
  * the x mod 4 histogram against the 27.8/27.8/22.2/22.2 null of 18 columns: the cheap detector for the old checkerboard artifact
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402

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
            cl, emb, sp, _, hid_next = policy_head(net, obs_t, hid)
            gi, gx, gy, _ = greedy_from_logits(net, obs_t, cl, emb, sp, hid_next)
            action = (gi, gx, gy)
            if use_search:
                action, _, _ = search_action(net, env, obs_t, cl, emb, sp,
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

    here = python_ai.PACKAGE_DIR
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
