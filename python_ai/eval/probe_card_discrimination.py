"""Is a card played more, or played when it is worth playing?

P(play | in hand) cannot tell the two apart: a policy that plays Fireball more
everywhere scores the same as one that learned when. So this measures
discrimination:

    p_hi = P(play C | C in hand, affordable, the board offers C a lot)
    p_lo = P(play C | C in hand, affordable, the board offers C nothing)

Both rising together is systemic drift; p_hi rising with p_lo flat is the
policy learning the condition. Each card is split on what it answers, the same
quantities as measure_deck_matchups.py: Fireball on best spell catch, The Log
on best corridor catch, Cannon on threat HP on our half.

Every number is a mean over states of the card slot's exact softmax
probability, not a count of sampled plays, which for a rarely played card would
need many thousands of decisions.

Checkpoints are compared on one frozen bank of observation sequences, collected
once. Each checkpoint replays the sequences and builds its own hidden state
(teacher forcing); storing one net's (hx, cx) would feed another its memory.

    # collect once
    ... -m python_ai.eval.probe_card_discrimination --collect --episodes 16 \\
        --weights model_weights_phase5.pth --bank bank.npz

    # score any checkpoint against it
    ... -m python_ai.eval.probe_card_discrimination --bank bank.npz \\
        --weights model_weights_phase6.pth --csv trend.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.eval.measure_deck_matchups import log_catch_map  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402

#: The watched cards and the board quantity each answers. By name, resolved
#: against DEFAULT_DECK at run time.
WATCHED = {
    "Fireball": "fb",
    "The Log": "log",
    "Cannon": "threat",
}

#: High / low opportunity percentiles. Percentiles, since the three quantities
#: have different units and absolute cuts would shift with the deck pool.
HI_PCTL, LO_PCTL = 70.0, 30.0


def _opportunity(obs):
    """The three board quantities, for one observation."""
    o = np.asarray(obs, dtype=np.float32)
    return {
        "fb": float(tactics.spell_catch_map(o).max()),
        "log": float(log_catch_map(o).max()),
        "threat": float(tactics.threat_level(o)),
    }


def build_net(weights):
    path = (weights if os.path.isabs(weights)
            else os.path.join(python_ai.PACKAGE_DIR, weights))
    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    ep = None
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        load_state_dict_flexible(net, sd, os.path.basename(path))
        ep = ck.get("episodes_completed") if isinstance(ck, dict) else None
    else:
        raise SystemExit(f"{path} not found")
    net.eval()
    return net, ep


@torch.no_grad()
def collect_bank(net, *, episodes, stage, seed0, max_steps=400):
    """Roll the policy against the deck pool and store observation SEQUENCES."""
    from python_ai.opponents import deck_pool

    decks = deck_pool.load_pool()
    seqs, deck_names = [], []
    for i in range(episodes):
        deck = decks[i % len(decks)]        # round-robin: every deck represented
        env = gym_wrapper.MicroRoyaleEnv({
            "ai_deck": list(gym_wrapper.DEFAULT_DECK),
            "opp_deck": list(deck.card_ids),
            "opponent": "teacher",
            "teacher_stage": int(stage),
        })
        env.game.seed(seed0 + i)
        obs, _ = env.reset()
        hx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        cx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        rows = []
        for _t in range(max_steps):
            rows.append(np.asarray(obs, dtype=np.float16))
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = net.affordability_mask(t)
            feats, embeds, spatial_f = net.extract_features(t)
            logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
            idx = logits.argmax(-1)
            place = net.placement_given_card(hx, embeds, idx, t, spatial_f)
            x, y = net.cell_to_xy(place.argmax(-1))
            obs, _r, term, trunc, _i = env.step({
                "card_index": np.array([int(idx.item())]),
                "target_x": x.numpy().reshape(1, 1),
                "target_y": y.numpy().reshape(1, 1),
                "activate_ability_slot1": np.zeros(1, dtype=np.int64),
                "activate_ability_slot2": np.zeros(1, dtype=np.int64),
            })
            if term or trunc:
                break
        seqs.append(np.stack(rows))
        deck_names.append(deck.name)
        print(f"  bank: episode {i + 1}/{episodes} ({deck.name}), "
              f"{len(rows)} steps", flush=True)
    return seqs, deck_names


def save_bank(path, seqs, deck_names):
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    flat = np.concatenate(seqs, axis=0)
    opp = np.array([[_opportunity(o)[k] for k in ("fb", "log", "threat")]
                    for o in flat.astype(np.float32)], dtype=np.float32)
    np.savez_compressed(path, obs=flat, lengths=lengths, opp=opp,
                        decks=np.array(deck_names))
    print(f"wrote {path}: {len(seqs)} episodes, {len(flat)} states, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


def load_bank(path):
    z = np.load(path, allow_pickle=False)
    obs, lengths, opp, decks = z["obs"], z["lengths"], z["opp"], z["decks"]
    seqs, i = [], 0
    for n in lengths:
        seqs.append(obs[i:i + n])
        i += n
    return seqs, opp, [str(d) for d in decks]


@torch.no_grad()
def score(net, seqs):
    """Per-state probability of each watched card, by teacher-forced replay.

    Returns (probs, in_hand, quality), each (n_states, n_watched):
      probs       softmax probability of the card's hand slot, 0 where absent
      in_hand     True where the card is in hand and affordable
      quality     expected value under the placement distribution / the best cell's value; nan for Cannon, which has no per-cell value map
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    watch_ids = [next(c for c in deck if E.get_card_info(c)["name"] == n)
                 for n in WATCHED]
    # Cards with a per-cell value map. Cannon's value is a defensive
    # interaction over time, so it reports nan rather than a made-up proxy.
    vmap = {"Fireball": lambda o: tactics.spell_catch_map(o),
            "The Log": log_catch_map}
    P, H, Q = [], [], []
    for seq in seqs:
        hx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        cx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        for row in seq:
            o32 = np.asarray(row, dtype=np.float32)
            t = torch.tensor(o32).unsqueeze(0)
            mask = net.affordability_mask(t)
            feats, embeds, spatial_f = net.extract_features(t)
            logits, _, _, _, (hx, cx) = net.step_lstm_and_card(
                feats, (hx, cx), mask)
            probs = torch.softmax(logits, dim=-1)[0]
            hand = net.hand_card_ids(t)[0].tolist()
            p_row, h_row, q_row = [], [], []
            for j, (name, cid) in enumerate(zip(WATCHED, watch_ids)):
                slot = next((k for k, c in enumerate(hand) if c == cid), None)
                if slot is None or not bool(mask[0, slot]):
                    p_row.append(0.0)
                    h_row.append(False)
                    q_row.append(np.nan)
                    continue
                p_row.append(float(probs[slot]))
                h_row.append(True)
                if name not in vmap:
                    q_row.append(np.nan)
                    continue
                # Placement quality: the share of the achievable value the
                # placement distribution expects to collect. A usage rise with
                # flat quality is hollow.
                idx = torch.tensor([slot])
                pl = net.placement_given_card(hx, embeds, idx, t, spatial_f)
                pmask = net.placement_mask(t, idx)
                pl = pl.masked_fill(~pmask, float("-inf"))
                cellp = torch.softmax(pl, dim=-1)[0].numpy()
                v = vmap[name](o32).reshape(-1)
                best = float(v.max())
                q_row.append(float((cellp * v).sum() / best)
                             if best > 1e-6 else np.nan)
            P.append(p_row)
            H.append(h_row)
            Q.append(q_row)
    return np.array(P), np.array(H), np.array(Q)


def analyse(probs, in_hand, quality, opp, decks, lengths_per_state):
    """The report: marginal, conditional, and the discrimination ratio."""
    keys = list(WATCHED.values())
    out = {}
    for j, (name, key) in enumerate(WATCHED.items()):
        col = keys.index(key)
        q = opp[:, col]
        live = in_hand[:, j]
        if live.sum() < 20:
            out[name] = None
            continue
        p = probs[live, j]
        qual = quality[live, j]
        qq = q[live]
        hi_cut = np.percentile(qq, HI_PCTL)
        lo_cut = np.percentile(qq, LO_PCTL)
        hi = qq >= hi_cut
        lo = qq <= lo_cut
        p_hi = float(p[hi].mean()) if hi.any() else float("nan")
        p_lo = float(p[lo].mean()) if lo.any() else float("nan")
        out[name] = {
            "n_states": int(live.sum()),
            "marginal": float(p.mean()),
            "p_hi": p_hi,
            "p_lo": p_lo,
            "ratio": (p_hi / p_lo) if p_lo > 1e-9 else float("nan"),
            "hi_cut": float(hi_cut),
            "lo_cut": float(lo_cut),
            # Quality on high-opportunity states only; on an empty board there
            # is no good cell to find.
            "quality_hi": (float(np.nanmean(qual[hi]))
                           if hi.any() and not np.all(np.isnan(qual[hi]))
                           else float("nan")),
        }
    # Per-deck marginal: a rise concentrated in decks that offer the card
    # something is the mechanism working.
    per_deck = {}
    state_deck = np.concatenate([[d] * n for d, n in
                                 zip(decks, lengths_per_state)])
    for j, name in enumerate(WATCHED):
        d = {}
        for dk in sorted(set(decks)):
            sel = (state_deck == dk) & in_hand[:, j]
            if sel.sum() >= 10:
                d[dk] = float(probs[sel, j].mean())
        per_deck[name] = d
    return out, per_deck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights_phase5.pth")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--stage", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--csv", default="")
    ap.add_argument("--per-deck", action="store_true")
    args = ap.parse_args()

    net, ep = build_net(args.weights)

    if args.collect:
        print(f"collecting a bank from {args.weights} (episode {ep}) "
              f"at teacher rung {args.stage}...")
        seqs, decks = collect_bank(net, episodes=args.episodes,
                                   stage=args.stage, seed0=args.seed)
        save_bank(args.bank, seqs, decks)
        return

    seqs, opp, decks = load_bank(args.bank)
    lengths = [len(s) for s in seqs]
    t0 = time.time()
    probs, in_hand, quality = score(net, seqs)
    res, per_deck = analyse(probs, in_hand, quality, opp, decks, lengths)

    print(f"\n{os.path.basename(args.weights)}  episode {ep}   "
          f"({len(probs)} states, {time.time() - t0:.0f}s)")
    print(f"{'card':<10}{'n':>6}{'marginal':>11}{'p_lo':>10}{'p_hi':>10}"
          f"{'hi/lo':>8}{'place_q':>9}   what 'opportunity' means")
    print("-" * 88)
    for name, r in res.items():
        if r is None:
            print(f"{name:<10}   (too few states)")
            continue
        qh = r["quality_hi"]
        qs = "     n/a" if qh != qh else f"{qh:>9.3f}"
        print(f"{name:<10}{r['n_states']:>6}{r['marginal']:>11.5f}"
              f"{r['p_lo']:>10.5f}{r['p_hi']:>10.5f}{r['ratio']:>8.2f}{qs}"
              f"   {WATCHED[name]} >= {r['hi_cut']:.0f} vs <= {r['lo_cut']:.0f}")
    print("\nhi/lo > 1 means the policy prefers the card WHEN THE BOARD OFFERS IT")
    print("something. A marginal that rises with a FLAT ratio is systemic drift.")

    if args.per_deck:
        print()
        for name, d in per_deck.items():
            top = sorted(d.items(), key=lambda kv: -kv[1])[:5]
            print(f"  {name:<10} " + "  ".join(f"{k[:16]} {v:.4f}"
                                               for k, v in top))

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["wall", "episode", "card", "n", "marginal",
                            "p_lo", "p_hi", "ratio", "quality_hi"])
            for name, r in res.items():
                if r is None:
                    continue
                w.writerow([time.strftime("%H:%M:%S"), ep, name, r["n_states"],
                            f"{r['marginal']:.6f}", f"{r['p_lo']:.6f}",
                            f"{r['p_hi']:.6f}", f"{r['ratio']:.4f}",
                            f"{r['quality_hi']:.5f}"])
        print(f"\nappended to {args.csv}")


if __name__ == "__main__":
    main()
