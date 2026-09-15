"""Does the entropy floor pull a card's placement mass off its optimum?

THE HYPOTHESIS. The placement entropy coefficient is pinned at its 0.01 floor,
so there is a constant pressure toward a FLATTER placement distribution. Flatter
means closer to uniform over the card's LEGAL cells -- and the uniform
distribution has a fixed centre of mass set purely by the legal mask's geometry.
So the entropy term does not push in a neutral direction: it pulls every card's
mass toward its own legal region's centroid.

For a card whose optimum sits near that centroid the pull is negligible. For a
card whose optimum sits at the EDGE of its legal region the pull is large and
one-directional, because there is no legal space on the far side to balance it.

The Log is the second case by construction: rolling spells have been capped at
the own half plus the river (y <= 17.5) since 2026-08-29, and The Log's value is
concentrated at y ~ 17, i.e. on that boundary. Fireball's legal region runs the
whole board and its optimum sits comfortably inside.

WHAT THIS MEASURES, per card, on the bank's high-opportunity states:

    uniform_y   centre of mass of a UNIFORM distribution over the legal cells
                -- where entropy pressure alone would put the mass
    optimum_y   centre of mass of the CATCH MAP -- where the value actually is
    policy_y    centre of mass of the policy's actual placement distribution

`pull` is optimum_y - uniform_y: the distance and direction the entropy term
fights the value signal over. `progress` is how far the policy has travelled
from its optimum toward the uniform centroid, 0.0 meaning it still sits on the
value and 1.0 meaning entropy has won completely.

This is arithmetic on the mask and the value map. It needs no training, and it
tests the MECHANISM rather than correlating two time series.

    ... -m python_ai.eval.entropy_pull_geometry --bank <bank.npz> \\
        --weights model_weights_phase7.pth
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
from python_ai.engine_constants import BOARD_H, BOARD_W  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.eval.measure_deck_matchups import log_catch_map  # noqa: E402
from python_ai.eval.probe_card_discrimination import HI_PCTL, load_bank  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402

VALUE_MAP = {"Fireball": lambda o: tactics.spell_catch_map(o),
             "The Log": log_catch_map}


def row_com(weights_grid):
    """Centre of mass along y for a (BOARD_H, BOARD_W) non-negative grid."""
    rows = weights_grid.sum(axis=1)
    tot = rows.sum()
    if tot <= 0:
        return float("nan")
    return float((rows * np.arange(BOARD_H)).sum() / tot)


@torch.no_grad()
def measure(net, seqs, card_name):
    vmap = VALUE_MAP[card_name]
    deck = list(gym_wrapper.DEFAULT_DECK)
    cid = next(c for c in deck if E.get_card_info(c)["name"] == card_name)
    got = []
    for seq in seqs:
        hx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        cx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        for row in seq:
            o32 = np.asarray(row, dtype=np.float32)
            t = torch.tensor(o32).unsqueeze(0)
            mask = net.affordability_mask(t)
            feats, embeds, spatial_f = net.extract_features(t)
            _, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
            hand = net.hand_card_ids(t)[0].tolist()
            slot = next((k for k, c in enumerate(hand) if c == cid), None)
            if slot is None or not bool(mask[0, slot]):
                continue
            v = vmap(o32)
            if float(v.max()) <= 1e-6:
                continue
            idx = torch.tensor([slot])
            pl = net.placement_given_card(hx, embeds, idx, t, spatial_f)
            legal = net.placement_mask(t, idx)[0].numpy().reshape(BOARD_H, BOARD_W)
            cellp = torch.softmax(pl.masked_fill(~net.placement_mask(t, idx),
                                                 float("-inf")), dim=-1)[0]
            got.append((float(v.max()), legal.astype(np.float64),
                        v.astype(np.float64),
                        cellp.numpy().reshape(BOARD_H, BOARD_W)))
    if not got:
        return None
    opp = np.array([g[0] for g in got])
    keep = opp >= np.percentile(opp, HI_PCTL)
    got = [g for g, k in zip(got, keep) if k]

    u, o, p, legal_n = [], [], [], []
    for _best, legal, val, cellp in got:
        u.append(row_com(legal))                 # uniform over legal cells
        o.append(row_com(val * legal))           # where the value is
        p.append(row_com(cellp))                 # where the policy is
        legal_n.append(legal.sum())
    u, o, p = np.array(u), np.array(o), np.array(p)
    # A handful of states have NO catch value on any legal cell (measured: 1.8%
    # for The Log), so `optimum_y` is undefined there. nanmean rather than mean:
    # six such states out of 334 turned every reported figure into nan.
    pull = o - u
    prog = np.where(np.abs(pull) > 1e-6, (o - p) / pull, np.nan)
    ok = np.isfinite(o)
    return {"n": len(got), "n_scored": int(ok.sum()),
            "uniform_y": float(np.nanmean(u)), "optimum_y": float(np.nanmean(o)),
            "policy_y": float(np.nanmean(p)), "pull": float(np.nanmean(pull)),
            "progress": float(np.nanmean(prog)),
            "legal_cells": float(np.mean(legal_n))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cards", default="The Log,Fireball")
    args = ap.parse_args()
    seqs = load_bank(args.bank)[0]
    p = args.weights if os.path.isabs(args.weights) else os.path.join(
        python_ai.PACKAGE_DIR, args.weights)
    net = load_net(p, torch.device("cpu"), verbose=False)
    print(f"weights: {os.path.basename(p)}\n")
    print(f"{'card':<10}{'legal':>8}{'uniform_y':>11}{'optimum_y':>11}"
          f"{'policy_y':>10}{'pull':>8}{'progress':>10}")
    print("-" * 68)
    rows = {}
    for c in [x.strip() for x in args.cards.split(",") if x.strip()]:
        r = measure(net, seqs, c)
        if not r:
            print(f"{c:<10}  (never live on high-opportunity states)")
            continue
        rows[c] = r
        print(f"{c:<10}{r['legal_cells']:>8.0f}{r['uniform_y']:>11.2f}"
              f"{r['optimum_y']:>11.2f}{r['policy_y']:>10.2f}"
              f"{r['pull']:>8.2f}{r['progress']:>10.2f}")
    print("""
READ: `pull` is how far the value sits from where entropy alone would put the
mass -- the distance the two terms fight over, in board rows. `progress` is the
fraction of that distance the policy has already travelled toward the uniform
centroid: 0.0 = still on the value, 1.0 = entropy has won outright.

The hypothesis predicts a LARGE pull and a HIGH progress for The Log, and a
small pull for Fireball. If The Log's pull is small, or its progress is no
higher than Fireball's, the entropy floor is not the explanation.""")


if __name__ == "__main__":
    main()
