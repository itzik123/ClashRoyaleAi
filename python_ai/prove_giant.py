"""Is the cured net's GIANT placement any good? Engine-scored, paired.

    python_ai/venv/Scripts/python.exe python_ai/prove_giant.py \
        --a model_weights_selfplay.pth --b model_weights_cured.pth --episodes 12

WHY THIS EXISTS. `prove_placement.py` scores Cannon and Fireball only. The Giant
is the third card the coverage hole starved, its modal share improved from 55.6%
to ~12% under the advisor target, and **nobody has ever checked whether that
made it any better** -- modal share is a dynamism metric, not a value one. A
head can spread its mass and still place badly; this project has that exact
result on record for the entropy coverage term.

THE ORACLE. Enemy tower damage over 600 ticks after injecting a Giant, minus
the same state with no Giant. That is the metric `tactics.best_giant_cell` was
itself validated on (bridge on the weaker lane 535.6 vs the policy's own cell
3.3, n=913, p = 3.0e-87), so this scores the net against the advisor on the
advisor's own terms.

600 ticks because the Giant crosses ~15 tiles at 0.06 tiles/tick post-speed-fix.
Injection costs no elixir, so the rest of the match is untouched and every arm
faces one identical state.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
import tactics  # noqa: E402
from expert_iteration import load_net  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from prove_placement import paired_report  # noqa: E402

CE = E.ClashRoyaleEnv
GIANT = tactics.GIANT_ID
GIANT_FLIGHT = 600
NOOP = 4


def giant_value(env, x, y, baseline):
    """Enemy tower damage a Giant at (x,y) causes, over its whole advance."""
    s = env.snapshot()
    before = s.get_tower_damage_dealt(0)
    s.inject(GIANT, float(x), float(y), 0)
    for _ in range(GIANT_FLIGHT // 30):
        s.step(NOOP, 0.0, 0.0, 30)
        if s.is_game_over():
            break
    return (s.get_tower_damage_dealt(0) - before) - baseline


def giant_baseline(env):
    s = env.snapshot()
    before = s.get_tower_damage_dealt(0)
    for _ in range(GIANT_FLIGHT // 30):
        s.step(NOOP, 0.0, 0.0, 30)
        if s.is_game_over():
            break
    return s.get_tower_damage_dealt(0) - before


@torch.no_grad()
def giant_cell(net, obs_t, hid):
    """The net's proposed Giant cell, plus the advanced hidden state.

    ONE LSTM step per net per timestep -- stepping it per query would run the
    reference at double clock (prove_placement.step_net documents this trap).
    """
    feats, emb, sp = net.extract_features(obs_t)
    lg, _, _, _, hid = net.step_lstm_and_card(
        feats, hid, net.affordability_mask(obs_t))
    hand = net.hand_card_ids(obs_t)[0].tolist()
    cell = None
    if GIANT in hand:
        slot = torch.tensor([hand.index(GIANT)])
        c = int(net.placement_given_card(hid[0], emb, slot, obs_t, sp).argmax(-1))
        cell = (c % tactics.BOARD_W, c // tactics.BOARD_W)
    gi = int(lg.argmax(-1).item())
    gp = net.placement_given_card(hid[0], emb, torch.tensor([gi]), obs_t, sp)
    gc = int(gp.argmax(-1).item())
    return cell, (gi, float(gc % tactics.BOARD_W), float(gc // tactics.BOARD_W)), hid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="model_weights_selfplay.pth")
    ap.add_argument("--b", default="model_weights_cured.pth")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    def _load(p):
        return load_net(p if os.path.isabs(p) else os.path.join(here, p), dev)

    nets = {"a": _load(args.a), "b": _load(args.b)}
    ref = nets["b"]          # the cured net draws the trajectory
    legal = ref._placement_legal[GIANT].numpy().astype(bool)
    legal_idx = np.nonzero(legal)[0]
    rng = np.random.default_rng(0)

    arms = ["a", "b", "advisor", "random"]
    vals = {k: [] for k in arms}
    cells = {k: Counter() for k in arms}

    for ep in range(args.episodes):
        env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        env.set_opponent_elixir_multiplier(args.opp_elixir)
        env.reset()
        hid = {k: (torch.zeros(1, 256), torch.zeros(1, 256)) for k in nets}
        obs = env.get_observation_for_team(0)

        for _t in range(300):
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            props, greedy = {}, {}
            for k, net in nets.items():
                c, g, hid[k] = giant_cell(net, obs_t, hid[k])
                props[k] = c
                greedy[k] = g
            gx, gy, _ = tactics.best_giant_cell(obs, legal=legal)
            props["advisor"] = (int(gx), int(gy))

            if props["b"] is not None and props["a"] is not None:
                base = giant_baseline(env)
                r = int(rng.choice(legal_idx))
                props["random"] = (r % tactics.BOARD_W, r // tactics.BOARD_W)
                for k in arms:
                    c = props[k]
                    vals[k].append(giant_value(env, c[0], c[1], base))
                    cells[k][c] += 1

            gi, tx, ty = greedy["b"]
            res = env.step(gi, tx, ty, 10)
            obs = res.observation
            if res.done:
                break
        print(f"  ep{ep}: {len(vals['b'])} scored states", flush=True)

    print("\n" + "=" * 72)
    print("GIANT -- enemy tower damage over 600 ticks "
          f"(n={len(vals['b'])})")
    label = {"a": args.a, "b": args.b, "advisor": "advisor (tactics)",
             "random": "random legal cell"}
    for k in arms:
        if vals[k]:
            c, n = cells[k].most_common(1)[0]
            print(f"    {label[k]:34s} {np.mean(vals[k]):8.1f}   "
                  f"modal {c} {n / max(1, len(vals[k])):.1%}  "
                  f"cells {len(cells[k])}")

    print()
    for base_arm, other in (("random", "b"), ("a", "b"), ("b", "advisor")):
        if vals[base_arm] and vals[other]:
            paired_report("giant", vals[base_arm], vals[other],
                          label[base_arm], label[other],
                          np.random.default_rng(0))


if __name__ == "__main__":
    main()
