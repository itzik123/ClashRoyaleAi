"""Where did a card's placement mass move between two checkpoints?

`place_q` says a card's aim got worse, not how: mass sliding onto a bad region,
spreading out, or staying put while the boards changed. This scores two
checkpoints on the same frozen bank as probe_card_discrimination and reports,
for the high-opportunity states:

  * expected catch as a share of the best cell        (place_q)
  * placement entropy                                 (sharpness)
  * mean row / column of the placement mass           (where it sits)
  * the modal cell and how often it repeats           (collapse detector)

    ... -m python_ai.eval.diff_placement_mass --bank <bank.npz> \\
        --a stage_checkpoints/stage5_ep00041173.pth --b model_weights_phase7.pth \\
        --card "The Log"
"""
import argparse
import os
import sys
from collections import Counter

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
from python_ai.eval.probe_card_discrimination import (  # noqa: E402
    HI_PCTL, load_bank,
)
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402

VALUE_MAP = {"Fireball": lambda o: tactics.spell_catch_map(o),
             "The Log": log_catch_map}


@torch.no_grad()
def analyse(net, seqs, card_name):
    """Placement statistics for `card_name` on the bank's HIGH-opportunity states."""
    vmap = VALUE_MAP[card_name]
    deck = list(gym_wrapper.DEFAULT_DECK)
    cid = next(c for c in deck if E.get_card_info(c)["name"] == card_name)

    rows = []
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
            best = float(v.max())
            if best <= 1e-6:
                continue
            idx = torch.tensor([slot])
            pl = net.placement_given_card(hx, embeds, idx, t, spatial_f)
            pm = net.placement_mask(t, idx)
            pl = pl.masked_fill(~pm, float("-inf"))
            cellp = torch.softmax(pl, dim=-1)[0].numpy()
            rows.append((best, cellp, v.reshape(-1)))

    if not rows:
        return None
    opp = np.array([r[0] for r in rows])
    keep = opp >= np.percentile(opp, HI_PCTL)
    rows = [r for r, k in zip(rows, keep) if k]

    qs, ents, ys, xs, modes = [], [], [], [], []
    for best, cellp, vflat in rows:
        qs.append(float((cellp * vflat).sum() / best))
        p = cellp[cellp > 0]
        ents.append(float(-(p * np.log(p)).sum() / np.log(len(cellp))))
        grid = cellp.reshape(BOARD_H, BOARD_W)
        ys.append(float((grid.sum(1) * np.arange(BOARD_H)).sum()))
        xs.append(float((grid.sum(0) * np.arange(BOARD_W)).sum()))
        modes.append(int(cellp.argmax()))
    counts = Counter(modes)
    top_cell, top_n = counts.most_common(1)[0]
    return {
        "n": len(rows),
        "place_q": float(np.mean(qs)),
        "entropy_frac": float(np.mean(ents)),
        "mean_y": float(np.mean(ys)),
        "mean_x": float(np.mean(xs)),
        "modal_cell": (top_cell // BOARD_W, top_cell % BOARD_W),
        "modal_share": top_n / len(rows),
        "distinct_modes": len(counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--a", required=True, help="earlier checkpoint")
    ap.add_argument("--b", required=True, help="later checkpoint")
    ap.add_argument("--card", default="The Log")
    args = ap.parse_args()

    seqs, _decks = load_bank(args.bank)[:2]
    dev = torch.device("cpu")
    out = {}
    for label, path in (("A", args.a), ("B", args.b)):
        p = path if os.path.isabs(path) else os.path.join(python_ai.REPO_ROOT, path)
        if not os.path.exists(p):
            p = os.path.join(python_ai.PACKAGE_DIR, path)
        out[label] = analyse(load_net(p, dev, verbose=False), seqs, args.card)
        print(f"  {label}: {os.path.basename(path)}")

    a, b = out["A"], out["B"]
    if not a or not b:
        raise SystemExit("card never live on the bank's high-opportunity states")
    print(f"\n{args.card} on {a['n']} high-opportunity bank states")
    print(f"{'':<22}{'A (earlier)':>14}{'B (later)':>14}{'delta':>10}")
    for k, fmtq in (("place_q", "{:.3f}"), ("entropy_frac", "{:.3f}"),
                    ("mean_y", "{:.2f}"), ("mean_x", "{:.2f}"),
                    ("modal_share", "{:.3f}")):
        av, bv = a[k], b[k]
        print(f"  {k:<20}" + fmtq.format(av).rjust(14)
              + fmtq.format(bv).rjust(14) + f"{bv - av:>+10.3f}")
    print(f"  {'modal_cell (y,x)':<20}{str(a['modal_cell']):>14}{str(b['modal_cell']):>14}")
    print(f"  {'distinct modes':<20}{a['distinct_modes']:>14}{b['distinct_modes']:>14}")
    print("\nREAD: entropy_frac up with place_q down = the head got FLATTER (lost "
          "sharpness).\n      entropy_frac flat/down with place_q down = mass MOVED "
          "to worse cells.\n      modal_share up = collapsing onto one cell "
          "regardless of the board.")


if __name__ == "__main__":
    main()
