"""Paired A/B for the inference-time solvency gate.

Both arms get a bit-identical opening via env.snapshot(). Reports the
bankruptcy rate (what the gate is designed to move, thousands of samples per
episode) alongside win rate (what matters, and far noisier).

    python_ai/venv/Scripts/python.exe python_ai/eval/gate_ab.py --n 120
"""
import argparse
import os
import sys
from math import comb

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
from python_ai.engine_constants import BOARD_W  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv
SKIP = 10
CHEAPEST = 3.0


@torch.no_grad()
def play(net, env, device, gate):
    hx = torch.zeros(1, LSTM_HIDDEN, device=device)
    cx = torch.zeros(1, LSTM_HIDDEN, device=device)
    obs = env.get_observation_for_team(0)
    reward, steps, done = 0.0, 0, False
    elix, blocked = [], 0

    while not done and steps < 400:
        o = np.asarray(obs, dtype=np.float32)
        t = torch.tensor(o, device=device).unsqueeze(0)
        mask = net.affordability_mask(t)

        if gate is not None:
            costs = net.hand_costs_from_obs(t)[0].tolist()
            allow = torch.tensor([gate.mask(o, costs)], dtype=torch.bool, device=device)
            newmask = mask & allow
            # The no-op column is legal in both masks, so this only ever
            # removes card slots.
            blocked += int((mask & ~newmask).sum())
            mask = newmask

        feats, embeds, sp = net.extract_features(t)
        logits, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
        gi = int(logits.argmax(-1).item())
        place = net.placement_given_card(hx2, embeds, torch.tensor([gi], device=device), t, sp)
        cell = int(place.argmax(-1).item())

        elix.append(tactics.own_elixir(o))
        hx, cx = hx2, cx2
        r = env.step(gi, float(cell % BOARD_W), float(cell // BOARD_W), SKIP)
        obs, reward, done = r.observation, float(r.reward), r.done
        steps += 1

    e = np.array(elix)
    return dict(score=outcome_score(reward),
                bankrupt=float((e < CHEAPEST).mean()),
                mean_elixir=float(e.mean()),
                tower_lost=float(env.get_tower_damage_dealt(1)),
                spent=float(env.get_elixir_spent(0)),
                blocked=blocked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--reserve", type=float, default=4.0)
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    net = load_net(os.path.join(here, args.net), device)
    gate = tactics.SolvencyGate(reserve=args.reserve)

    rows = {"off": [], "on": []}
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.set_opponent_elixir_multiplier(args.opp_elixir)
        root.reset()
        base = root.snapshot()
        rows["off"].append(play(net, base.snapshot(), device, None))
        rows["on"].append(play(net, base.snapshot(), device, gate))
        if (i + 1) % 20 == 0:
            a = np.mean([r["score"] for r in rows["off"]])
            b = np.mean([r["score"] for r in rows["on"]])
            print(f"  pair {i+1:>4}: off {a:.3f}  on {b:.3f}", flush=True)

    print(f"\n{args.n} paired openings, {args.net}, opponent {args.opp_elixir}x, "
          f"reserve {args.reserve}\n")
    keys = [("bankrupt <3 elixir", "bankrupt", "{:.1%}"),
            ("mean elixir", "mean_elixir", "{:.2f}"),
            ("elixir spent / ep", "spent", "{:.0f}"),
            ("tower HP lost / ep", "tower_lost", "{:.0f}"),
            ("win rate", "score", "{:.3f}")]
    print(f"{'metric':<22}{'gate off':>12}{'gate on':>12}")
    print("-" * 46)
    for label, k, fmt in keys:
        a = np.array([r[k] for r in rows["off"]], dtype=float)
        b = np.array([r[k] for r in rows["on"]], dtype=float)
        print(f"{label:<22}{fmt.format(a.mean()):>12}{fmt.format(b.mean()):>12}")
    print(f"{'card slots blocked':<22}{'--':>12}"
          f"{np.mean([r['blocked'] for r in rows['on']]):>12.0f}")

    rng = np.random.default_rng(0)
    for label, k, _ in keys:
        a = np.array([r[k] for r in rows["off"]], dtype=float)
        b = np.array([r[k] for r in rows["on"]], dtype=float)
        d = b - a
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better, worse = int((d > 0).sum()), int((d < 0).sum())
        m = better + worse
        p = (sum(comb(m, i) for i in range(min(better, worse) + 1)) * 2 / (2 ** m)
             if m else 1.0)
        print(f"\n  {label}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {better}+/{worse}-  sign p={min(1.0, p):.3g}")


if __name__ == "__main__":
    main()
