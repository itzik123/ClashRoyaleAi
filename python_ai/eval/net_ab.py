"""Paired win-rate comparison of two nets on bit-identical openings.

    python_ai/venv/Scripts/python.exe python_ai/eval/net_ab.py \
        --a model_weights_selfplay.pth --b model_weights_cured.pth --n 200

`env.snapshot()` gives both nets the same opening (hand, opponent), so far
fewer episodes are needed than unpaired (~1,568 per arm to resolve 5 points).
Greedy on both sides, with no search, override or gate: this measures the
network alone.
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

import clash_royale_env as E  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.eval.match_outcome import score_from_towers  # noqa: E402
from python_ai.engine_constants import BOARD_W  # noqa: E402

CE = E.ClashRoyaleEnv


@torch.no_grad()
def play(net, env, max_steps=400):
    """Greedy episode. Returns 1.0 win / 0.5 draw / 0.0 loss."""
    hid = (torch.zeros(1, 256), torch.zeros(1, 256))
    obs = env.get_observation_for_team(0)
    for _ in range(max_steps):
        ot = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        feats, emb, sp = net.extract_features(ot)
        lg, _, _, _, hid = net.step_lstm_and_card(
            feats, hid, net.affordability_mask(ot))
        gi = int(lg.argmax(-1).item())
        cell = int(net.placement_given_card(
            hid[0], emb, torch.tensor([gi]), ot, sp).argmax(-1).item())
        r = env.step(gi, float(cell % BOARD_W), float(cell // BOARD_W), 10)
        obs = r.observation
        if r.done:
            break
    # Scored by TimeoutRules, including the weakest-tower tie-break; see
    # match_outcome.py.
    return score_from_towers(env, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline net")
    ap.add_argument("--b", required=True, help="candidate net")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    def _load(p):
        return load_net(p if os.path.isabs(p) else os.path.join(here, p), dev)

    nets = {"A": _load(args.a), "B": _load(args.b)}
    for n in nets.values():
        for p in n.parameters():
            p.requires_grad_(False)

    rows = {"A": [], "B": []}
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.set_opponent_elixir_multiplier(args.opp_elixir)
        root.reset()
        base = root.snapshot()
        for k in ("A", "B"):
            rows[k].append(play(nets[k], base.snapshot()))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1:>4}: A {np.mean(rows['A']):.3f}  "
                  f"B {np.mean(rows['B']):.3f}", flush=True)

    a = np.array(rows["A"])
    b = np.array(rows["B"])
    d = b - a
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    better = int((d > 0).sum())
    worse = int((d < 0).sum())
    # Exact sign test on discordant pairs; ties say nothing about which net is
    # better.
    from math import comb
    n_disc = better + worse
    if n_disc:
        k = min(better, worse)
        p = min(1.0, 2 * sum(comb(n_disc, j) for j in range(k + 1)) / 2 ** n_disc)
    else:
        p = 1.0

    print(f"\n{args.n} paired openings, opponent {args.opp_elixir}x, greedy both sides")
    print(f"  A  {args.a}: {a.mean():.4f}")
    print(f"  B  {args.b}: {b.mean():.4f}")
    print(f"  paired delta {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {better} better / {worse} worse / {len(d) - n_disc} tied"
          f"   sign test p = {p:.4g}")


if __name__ == "__main__":
    main()
