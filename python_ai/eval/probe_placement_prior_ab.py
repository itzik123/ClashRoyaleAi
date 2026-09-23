"""Per-card placement diagnostics, paired across two checkpoints.

Reports each card's modal share (how often its conditional placement argmax
repeats across states), not its entropy: a good head is sharp but moves its
mode with the board, a broken one returns one cell regardless. Entropy can flag
the healthiest card and clear dead ones.

Every hand slot is evaluated at every decision, not only the played one, so a
card the policy has stopped playing is still visible. Both checkpoints run on
the same env seeds.

    python_ai/venv/Scripts/python.exe python_ai/eval/probe_placement_prior_ab.py \
        --weights-a _runs/prior_control/model_weights.pth \
        --weights-b _runs/prior_treat/model_weights.pth --episodes 30
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import human_prior as HP  # noqa: E402
from python_ai.engine_constants import card_name  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402


def _short(cid):
    return card_name(cid).rsplit("_", 1)[0]


def collect(weights, episodes, seeds, max_steps=400):
    """-> {card_id: dict of per-card placement statistics}."""
    net = load_net(weights, "cpu", verbose=True)
    env = gym_wrapper.MicroRoyaleEnv()
    H = net.__class__.LSTM_HIDDEN if hasattr(net.__class__, "LSTM_HIDDEN") else 256

    modes = defaultdict(Counter)      # card -> Counter(argmax cell)
    ents = defaultdict(list)          # card -> [H(place|card)/Hmax]
    dists = defaultdict(lambda: np.zeros(HP.N_CELLS, dtype=np.float64))
    plays = Counter()
    steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        hx, cx = torch.zeros(1, H), torch.zeros(1, H)
        for _t in range(max_steps):
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mask = net.affordability_mask(t)
                feats, embeds, spatial = net.extract_features(t)
                logits, _, _, _, (hx, cx) = net.step_lstm_and_card(
                    feats, (hx, cx), mask)
                idx = logits.argmax(-1)
                hand_ids = net.hand_card_ids(t)[0].tolist()

                # Every slot, not just the chosen one.
                for slot, cid in enumerate(hand_ids):
                    if cid < 0:
                        continue
                    si = torch.tensor([slot], dtype=torch.long)
                    pl = net.placement_given_card(hx, embeds, si, t, spatial)[0]
                    finite = torch.isfinite(pl)
                    if not bool(finite.any()):
                        continue
                    p = torch.softmax(pl[finite], -1).numpy()
                    modes[cid][int(pl.argmax().item())] += 1
                    hmax = float(np.log(int(finite.sum())))
                    h = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
                    ents[cid].append(h / max(1e-9, hmax))
                    full = np.zeros(HP.N_CELLS, dtype=np.float64)
                    full[finite.numpy()] = p
                    dists[cid] += full

                place = net.placement_given_card(hx, embeds, idx, t, spatial)
                cell = place.argmax(-1)
            x, y = net.cell_to_xy(cell)
            slot = int(idx.item())
            if slot < net.hand_size and hand_ids[slot] >= 0:
                plays[hand_ids[slot]] += 1
            steps += 1
            obs, _r, term, trunc, _i = env.step({
                "card_index": np.array([slot]),
                "target_x": x.numpy().reshape(1, 1),
                "target_y": y.numpy().reshape(1, 1),
                "activate_ability_slot1": np.zeros(1, dtype=np.int64),
                "activate_ability_slot2": np.zeros(1, dtype=np.int64),
            })
            if term or trunc:
                break
    return {"modes": modes, "ents": ents, "dists": dists,
            "plays": plays, "steps": steps}


def prior_kl(card_id, mean_dist):
    """KL(policy || human prior) for one card over the legal cells, or None where
    the prior has nothing to say.

    The direct treatment-effect measure: if the term does anything, this falls
    in the treated arm.
    """
    env = E.ClashRoyaleEnv(list(gym_wrapper.DEFAULT_DECK),
                           list(gym_wrapper.DEFAULT_DECK), max_ticks=3600)
    env.seed(0)
    env.reset()
    legal = np.array([env.is_valid_placement(card_id, float(c % HP.BOARD_W),
                                             float(c // HP.BOARD_W), 0)
                      for c in range(HP.N_CELLS)], dtype=bool)
    saved = HP.HUMAN_PRIOR_COEF
    HP.HUMAN_PRIOR_COEF = 1.0            # read the surface regardless of the gate
    HP._cache.clear()
    lg = HP.logits_for(card_id, legal)
    HP.HUMAN_PRIOR_COEF = saved
    HP._cache.clear()
    if lg is None:
        return None
    q = np.exp(lg[legal] - lg[legal].max())
    q /= q.sum()
    p = mean_dist[legal]
    s = p.sum()
    if s <= 0:
        return None
    p = p / s
    return float((p * np.log(np.clip(p, 1e-12, None) / np.clip(q, 1e-12, None))).sum())


def report(name, res, deck):
    print(f"\n=== {name} === ({res['steps']} decision steps)")
    print(f"  {'card':<12} {'slot-obs':>9} {'plays':>6} {'modal cell':>11} "
          f"{'modal share':>12} {'H/Hmax':>8} {'KL||prior':>10}")
    out = {}
    for cid in deck:
        m = res["modes"][cid]
        n = sum(m.values())
        if not n:
            print(f"  {_short(cid):<12} {0:>9}   (never in hand)")
            continue
        cell, cnt = m.most_common(1)[0]
        share = cnt / n
        h = float(np.mean(res["ents"][cid]))
        kl = prior_kl(cid, res["dists"][cid] / n)
        out[cid] = {"share": share, "h": h, "kl": kl, "n": n,
                    "plays": res["plays"][cid]}
        print(f"  {_short(cid):<12} {n:>9} {res['plays'][cid]:>6} "
              f"{f'({cell % HP.BOARD_W},{cell // HP.BOARD_W})':>11} "
              f"{share:>11.1%} {h:>8.3f} "
              f"{('%.3f' % kl) if kl is not None else '-':>10}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-a", required=True, help="control arm")
    ap.add_argument("--weights-b", default=None, help="treatment arm")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    deck = list(gym_wrapper.DEFAULT_DECK)
    seeds = list(range(args.seed, args.seed + args.episodes))

    a = report("A (control)", collect(args.weights_a, args.episodes, seeds), deck)
    if not args.weights_b:
        return 0
    b = report("B (treatment)", collect(args.weights_b, args.episodes, seeds), deck)

    print("\n=== PAIRED DELTA (B - A) ===")
    print(f"  {'card':<12} {'d modal share':>14} {'d H/Hmax':>10} {'d KL||prior':>12}")
    for cid in deck:
        if cid not in a or cid not in b:
            continue
        dk = ("%+.3f" % (b[cid]["kl"] - a[cid]["kl"])
              if a[cid]["kl"] is not None and b[cid]["kl"] is not None else "-")
        print(f"  {_short(cid):<12} {b[cid]['share'] - a[cid]['share']:>+13.1%} "
              f"{b[cid]['h'] - a[cid]['h']:>+10.3f} {dk:>12}")
    print("\n  Healthy treatment: modal share DOWN and KL||prior DOWN on the five")
    print("  cards that previously had no target. Modal share UP would mean the")
    print("  prior is sharpening a card onto one cell rather than spreading it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
