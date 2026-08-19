"""Does the high-resolution branch actually buy placement accuracy?

    python_ai/venv/Scripts/python.exe python_ai/prove_hires.py --episodes 12

THE QUESTION, AND WHY IT NEEDED ITS OWN HARNESS
-----------------------------------------------
`distill_tactics.py` fit the advisor's exact cell into the placement head with
the trunk frozen and measured cross-entropy falling 180.9 -> 21.4 while
exact-cell argmax match never left **0.0%**. That was read as the head being
unable to REPRESENT an exact cell, and it is the premise the whole
"high-resolution skip connection" plan rests on.

`test_python_ai.py` tested that premise directly and it is too strong: on
14 boards differing only in which column holds one enemy, the coarse head fits
14/14. Nearest-upsample + 3x3 conv lets a fine cell mix neighbouring pooled
cells, so sub-block position is recoverable in principle.

So the honest question is not "can it?" but "does it, at the scale and on the
target that actually failed?" -- hundreds of correlated rollout states, a target
that moves in both axes, two cards, with the alive cards anchored. That is what
this measures, as a controlled A/B:

  * ONE collection, shared by both arms, so the states and targets are identical
  * both arms start from the SAME checkpoint and the same seed
  * identical epochs, batch size, learning rate, clipping and anchor
  * the ONLY difference is whether `place_hires`/`place_ctx_hi` are trainable.
    The control leaves them at their zero init, which is bit-exactly the
    architecture that measured 0.0%.

WHAT IS REPORTED, AND WHY IT IS NOT JUST ACCURACY
-------------------------------------------------
Held-out exact-cell match is the headline, split from train match because a head
with a fresh branch has more capacity and could simply memorise.

Modal share and top-1 probability are printed together, never apart: CLAUDE.md
records that modal share degenerates on a near-uniform map (the argmax of a flat
map is arbitrary but deterministic, so a dissolved head reports ~99% modal
share). A real fix moves the mode WITH the board; a dissolved head reports a
high modal share at a top-1 probability near 1/612 = 0.0016.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import distill_tactics as D  # noqa: E402
import tactics  # noqa: E402
from policy_io import load_net  # noqa: E402

CANNON, FIREBALL = tactics.CANNON_ID, tactics.FIREBALL_ID
NAME = {CANNON: "Cannon", FIREBALL: "Fireball"}


def soft_target_logits(score_map, T):
    """Advisor score map -> target LOGITS at temperature T.

    Standardized over the legal cells first, so T means the same thing for the
    Cannon's coverage sums and the Fireball's caught-damage sums, which live on
    completely different scales. CLAUDE.md's expert-iteration entry is explicit
    that temperature must be read off the TARGET's own entropy rather than
    tuned on the outcome -- `--target-entropy` prints that table.
    """
    finite = torch.isfinite(score_map)
    out = torch.full_like(score_map, float("-inf"))
    for i in range(score_map.shape[0]):
        m = finite[i]
        v = score_map[i][m]
        std = v.std()
        if not torch.isfinite(std) or std < 1e-8:
            # A flat surface carries no preference. A uniform target over the
            # legal cells is the honest encoding of that, and it is also what
            # keeps such rows from dominating the loss with arbitrary noise.
            out[i][m] = 0.0
        else:
            out[i][m] = (v - v.mean()) / std / T
    return out


def fit(net, frozen, obs, hxs, targets, idx, args, train_hires,
        maps=None, soft_T=None):
    """Fit the placement pathway on `idx`. Mirrors distill_tactics.main()."""
    trainable = []
    for name, p in net.named_parameters():
        train_it = name.startswith(("place_ctx", "place_up", "card_id_embed"))
        if name.startswith(("place_hires", "place_ctx_hi")):
            # place_ctx_hi also matches the "place_ctx" prefix above, so it has
            # to be decided here rather than left to prefix order -- otherwise
            # the control arm would silently train half the branch and the A/B
            # would compare two treatments.
            train_it = train_hires
        p.requires_grad_(train_it)
        if train_it:
            trainable.append(p)
    opt = torch.optim.Adam(trainable, lr=args.lr)
    n = len(idx)

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        tot = seen = 0
        for i in range(0, n, args.batch):
            rows = idx[perm[i:i + args.batch]]
            ob = torch.tensor(obs[rows])
            hx = torch.tensor(hxs[rows])
            B = ob.shape[0]
            feats, embeds, sp = net.extract_features(ob)
            with torch.no_grad():
                _f2, e2, s2 = frozen.extract_features(ob)

            loss = torch.zeros(())
            for cid in D.TARGET_CARDS:
                slot = D.slot_of(net, ob, cid)
                m = slot >= 0
                if not bool(m.any()):
                    continue
                logits = net.placement_given_card(
                    hx[m], embeds[m], slot[m].clamp(min=0), ob[m], sp[m])
                if soft_T is None:
                    tgt = torch.tensor(targets[cid][rows][m.numpy()], dtype=torch.long)
                    loss = loss + F.cross_entropy(logits, tgt)
                else:
                    # KL to the advisor's whole surface, not to its argmax.
                    # masked_kl is reused rather than F.kl_div because both
                    # sides carry -inf on illegal cells and torch evaluates
                    # 0 * (-inf - -inf) = nan there.
                    tgt_logits = soft_target_logits(
                        torch.tensor(maps[cid][rows][m.numpy()]), soft_T)
                    loss = loss + D.masked_kl(logits, tgt_logits)

            if args.anchor > 0:
                anchor = torch.zeros(())
                for s in range(net.hand_size):
                    si = torch.full((B,), s, dtype=torch.long)
                    new = net.placement_given_card(hx, embeds, si, ob, sp)
                    with torch.no_grad():
                        old = frozen.placement_given_card(hx, e2, si, ob, s2)
                    keep = torch.ones(B, dtype=torch.bool)
                    for cid in D.TARGET_CARDS:
                        keep &= D.slot_of(net, ob, cid) != s
                    if bool(keep.any()):
                        anchor = anchor + D.masked_kl(new[keep], old[keep])
                loss = loss + args.anchor * anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
            tot += loss.item() * B
            seen += B
        print(f"    epoch {epoch}: loss {tot / max(1, seen):.4f}", flush=True)
    return net


@torch.no_grad()
def score(net, obs, hxs, targets, idx, label):
    """Exact-cell match, modal share and top-1 probability, per card."""
    out = {}
    for cid in D.TARGET_CARDS:
        hits = near = tot = 0
        modes, top1, dists = [], [], []
        for i in range(0, len(idx), 256):
            rows = idx[i:i + 256]
            ob = torch.tensor(obs[rows])
            hx = torch.tensor(hxs[rows])
            slot = D.slot_of(net, ob, cid)
            m = slot >= 0
            if not bool(m.any()):
                continue
            feats, embeds, sp = net.extract_features(ob)
            logits = net.placement_given_card(
                hx[m], embeds[m], slot[m].clamp(min=0), ob[m], sp[m])
            tgt = torch.tensor(targets[cid][rows][m.numpy()], dtype=torch.long)
            cells = logits.argmax(-1)
            hits += int((cells == tgt).sum())
            # Chebyshev tile distance to the advisor's cell. Exact match is a
            # harsh and partly unfair metric for the Cannon: `best_building_cell`
            # maximises a coverage score that has many near-ties, so its argmax
            # hops between cells that are worth the same, and a head landing one
            # tile away scores zero while having learned the geometry. Distance
            # says how wrong it is; exact match only says whether it is wrong.
            d = torch.maximum((cells % 18 - tgt % 18).abs(),
                              (cells // 18 - tgt // 18).abs())
            near += int((d <= 2).sum())
            dists += d.tolist()
            tot += int(m.sum())
            modes += cells.tolist()
            top1 += logits.softmax(-1).max(-1).values.tolist()
        if tot == 0:
            continue
        counts = np.bincount(modes, minlength=612)
        modal = counts.max() / max(1, len(modes))
        out[cid] = (hits / tot, modal, float(np.mean(top1)), tot,
                    int(counts.argmax()), near / tot, float(np.mean(dists)))
        cell = out[cid][4]
        print(f"    {label:9s} {NAME[cid]:8s}  exact {hits / tot:6.1%}  "
              f"within-2 {near / tot:6.1%}  mean dist {np.mean(dists):5.2f}   "
              f"modal ({cell % 18},{cell // 18}) {modal:5.1%}   "
              f"top-1 p {np.mean(top1):.4f}   n={tot}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of states, taken as a CONTIGUOUS TAIL. "
                         "Random rows would leak: states inside one episode are "
                         "highly correlated, so a shuffled split scores a "
                         "neighbour of every training state.")
    ap.add_argument("--out-treatment", default="model_weights_hires.pth")
    ap.add_argument("--out-soft", default="model_weights_hires_soft.pth")
    ap.add_argument("--soft-T", type=float, default=0.5,
                    help="temperature for the soft arm, applied to the "
                         "STANDARDIZED advisor score. Read the printed target "
                         "entropy before changing it: near 100%% of max is a "
                         "uniform target that trains on nothing.")
    ap.add_argument("--out-control", default="model_weights_hires_control.pth",
                    help="the coarse-only arm. Saved because the engine-scored "
                         "follow-up (prove_placement.py) needs all three nets: "
                         "the seed, this, and the treatment. Without it a later "
                         "comparison would conflate 'distillation happened' "
                         "with 'the branch helped'.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    dev = torch.device("cpu")

    seed_path = os.path.join(here, args.net)
    print(f"\ncollecting {args.episodes} episodes at {args.opp_elixir}x "
          f"from {args.net} ...", flush=True)
    t0 = time.time()
    collector = load_net(seed_path, dev)
    obs, hxs, targets, maps = D.collect(collector, args.episodes,
                                        args.opp_elixir, dev, want_maps=True)
    n = len(obs)
    cut = int(n * (1.0 - args.holdout))
    train_idx, test_idx = np.arange(cut), np.arange(cut, n)
    print(f"collected {n} states in {time.time() - t0:.0f}s "
          f"({len(train_idx)} train / {len(test_idx)} held out)\n", flush=True)

    # Report the target's own entropy before fitting anything to it. A target
    # at ~95% of maximum entropy is near-uniform: it trains, it looks busy, and
    # it carries no signal. That mistake cost an expert-iteration run once
    # already (CLAUDE.md, "Temperature is the knob").
    for cid in D.TARGET_CARDS:
        tl = soft_target_logits(torch.tensor(maps[cid][:256]), args.soft_T)
        p = tl.softmax(-1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        n_legal = torch.isfinite(tl).sum(-1).float()
        print(f"  soft target T={args.soft_T}: {NAME[cid]:8s} entropy "
              f"{float((ent / n_legal.log()).mean()):.1%} of max "
              f"({float(n_legal.mean()):.0f} legal cells)")
    print()

    arms = (("control", False, None),
            ("treatment", True, None),
            ("soft", True, args.soft_T))
    files = {"control": args.out_control, "treatment": args.out_treatment,
             "soft": args.out_soft}
    results = {}
    for label, train_hires, soft_T in arms:
        print(f"--- {label}: place_hires trainable={train_hires}  "
              f"target={'argmax' if soft_T is None else f'soft T={soft_T}'}")
        torch.manual_seed(0)
        np.random.seed(0)
        net = load_net(seed_path, dev, verbose=False)
        frozen = load_net(seed_path, dev, verbose=False)
        for p in frozen.parameters():
            p.requires_grad_(False)
        t = time.time()
        fit(net, frozen, obs, hxs, targets, train_idx, args, train_hires,
            maps=maps, soft_T=soft_T)
        print(f"    fitted in {time.time() - t:.0f}s")
        score(net, obs, hxs, targets, train_idx, "train")
        results[label] = score(net, obs, hxs, targets, test_idx, "HELD-OUT")
        out = os.path.join(here, files[label])
        torch.save({"model": net.state_dict()}, out)
        print(f"    saved {out}")
        print()

    print("=" * 72)
    print("HELD-OUT (the headline)")
    cols = [a[0] for a in arms]
    for cid in D.TARGET_CARDS:
        got = {k: results[k].get(cid) for k in cols}
        if not all(got.values()):
            continue
        print(f"  {NAME[cid]}")
        for metric, i, fmt in (("exact cell", 0, "6.1%"),
                               ("within 2", 5, "6.1%"),
                               ("mean dist", 6, "6.2f"),
                               ("top-1 p", 2, "6.4f"),
                               ("modal share", 1, "6.1%")):
            row = "   ".join(f"{k} {got[k][i]:{fmt}}" for k in cols)
            print(f"    {metric:12s} {row}")


if __name__ == "__main__":
    main()
