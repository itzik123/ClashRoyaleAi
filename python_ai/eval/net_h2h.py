"""Head-to-head between two nets, with the sides swapped.

    python_ai/venv/Scripts/python.exe python_ai/eval/net_h2h.py \
        --a model_weights_selfplay.pth --b model_weights_cured.pth --n 150

Separates degradation from specialization when a league-trained net scores
worse against the C++ heuristic: if B beats A head-to-head while losing to the
heuristic, it specialized; if it loses both, it degraded. Each pairing is
played from both sides, since side assignment alone can bias a one-sided
result. stepSelfPlay never runs the heuristic, so this is purely net vs net.
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
def act(net, env, team, hid):
    obs = torch.tensor(np.asarray(env.get_observation_for_team(team),
                                  dtype=np.float32)).unsqueeze(0)
    feats, emb, sp = net.extract_features(obs)
    lg, _, _, _, hid = net.step_lstm_and_card(
        feats, hid, net.affordability_mask(obs))
    gi = int(lg.argmax(-1).item())
    cell = int(net.placement_given_card(
        hid[0], emb, torch.tensor([gi]), obs, sp).argmax(-1).item())
    return gi, float(cell % BOARD_W), float(cell // BOARD_W), hid


def duel(net0, net1, env, max_steps=400):
    """net0 as team 0, net1 as team 1. Returns net0's score."""
    h0 = (torch.zeros(1, 256), torch.zeros(1, 256))
    h1 = (torch.zeros(1, 256), torch.zeros(1, 256))
    for _ in range(max_steps):
        # Both decide from the same board, then the engine applies both.
        g0, x0, y0, h0 = act(net0, env, 0, h0)
        g1, x1, y1, h1 = act(net1, env, 1, h1)
        r = env.step_self_play(g0, x0, y0, g1, x1, y1, 10)
        if r.done:
            break
    # Scored by TimeoutRules, including the weakest-tower tie-break; see
    # match_outcome.py.
    return score_from_towers(env, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    def _load(p):
        return load_net(p if os.path.isabs(p) else os.path.join(here, p), dev)

    A, B = _load(args.a), _load(args.b)
    for n in (A, B):
        for p in n.parameters():
            p.requires_grad_(False)

    scores = []          # B's score, averaged over both side assignments
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.reset()
        base = root.snapshot()
        # B on team 0, then B on team 1, from the same opening.
        b_as_0 = duel(B, A, base.snapshot())
        a_as_0 = duel(A, B, base.snapshot())
        scores.append(0.5 * (b_as_0 + (1.0 - a_as_0)))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1:>4}: B score {np.mean(scores):.3f}", flush=True)

    s = np.array(scores)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(s, len(s), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n{args.n} side-swapped pairings, greedy both sides, no heuristic")
    print(f"  B ({args.b}) scores {s.mean():.4f} against A ({args.a})")
    print(f"  95% CI [{lo:.4f}, {hi:.4f}]   (0.500 = evenly matched)")
    verdict = ("B is STRONGER" if lo > 0.5 else
               "B is WEAKER" if hi < 0.5 else "no difference resolved")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
