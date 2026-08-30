"""The pre-registered kill criterion: does conditioning earn its place?

Compares, on HELD-OUT episodes, the log-likelihood a human placement receives
under three models:

    uniform       1/|legal|                       -- knows nothing
    marginal      P(cell | card)                  -- knows where the card goes
    conditional   P(cell | card, context) + backoff

If `conditional` does not beat `marginal` out of sample, the context features
are not carrying information and the honest thing is to ship the marginal alone
rather than a key that looks sophisticated and predicts nothing.

Split by EPISODE, never by placement. Placements inside one match are heavily
correlated -- same player, same opponent, same board -- so a per-placement split
leaks the test set into training and inflates every number.
"""
from __future__ import annotations

import argparse

import numpy as np

import clash_royale_env as E

from .extract_prior import BOARD_W, N_CELLS, build_counts
from .prior import BACKOFF_CONTEXT, BACKOFF_LANE, BACKOFF_MARGINAL, probabilities

N_SPLITS = 5
HOLDOUT = 0.2


def legal_masks(deck):
    """{card_slot: (N_CELLS,) bool} from the engine's own predicate."""
    env = E.ClashRoyaleEnv(list(deck), list(deck), max_ticks=3600)
    env.seed(0)
    env.reset()
    out = {}
    for i, cid in enumerate(deck):
        m = np.zeros(N_CELLS, dtype=bool)
        for c in range(N_CELLS):
            m[c] = env.is_valid_placement(cid, float(c % BOARD_W),
                                          float(c // BOARD_W), 0)
        out[i] = m
    return out


def _marginal_counts(counts):
    """Collapse the context axis, keeping the array shape so `probabilities`
    can be reused unchanged -- one code path for both models."""
    m = counts.sum(axis=1, keepdims=True)
    return np.repeat(m, counts.shape[1], axis=1)


def evaluate(table, n_splits=N_SPLITS, holdout=HOLDOUT, seed=0):
    deck = table["deck"]
    masks = legal_masks(deck)
    episodes = np.unique(table["episode"])
    rng = np.random.default_rng(seed)

    rows = []
    for s in range(n_splits):
        perm = rng.permutation(episodes)
        n_test = max(1, int(round(holdout * len(episodes))))
        test_eps = set(perm[:n_test].tolist())
        te = np.array([e in test_eps for e in table["episode"]])
        tr = ~te
        counts = build_counts(table, tr)
        marg = _marginal_counts(counts)

        ll = {"uniform": [], "marginal": [], "conditional": []}
        hit = {"marginal": [], "conditional": []}
        levels = []
        for idx in np.nonzero(te)[0]:
            cs = int(table["card_slot"][idx])
            ctx = int(table["context"][idx])
            cell = int(table["cell"][idx])
            legal = masks[cs]
            if not legal[cell]:
                continue
            ll["uniform"].append(-np.log(legal.sum()))
            pm, _ = probabilities(marg, cs, ctx, legal)
            pc, lv = probabilities(counts, cs, ctx, legal)
            if pm is None or pc is None:
                continue
            ll["marginal"].append(np.log(max(pm[cell], 1e-12)))
            ll["conditional"].append(np.log(max(pc[cell], 1e-12)))
            hit["marginal"].append(int(pm.argmax() == cell))
            hit["conditional"].append(int(pc.argmax() == cell))
            levels.append(lv)
        rows.append({
            "n": len(ll["conditional"]),
            "uniform": float(np.mean(ll["uniform"])) if ll["uniform"] else np.nan,
            "marginal": float(np.mean(ll["marginal"])),
            "conditional": float(np.mean(ll["conditional"])),
            "top1_marginal": float(np.mean(hit["marginal"])),
            "top1_conditional": float(np.mean(hit["conditional"])),
            "levels": np.bincount(levels, minlength=4),
        })
    return rows


def report(rows):
    u = np.array([r["uniform"] for r in rows])
    m = np.array([r["marginal"] for r in rows])
    c = np.array([r["conditional"] for r in rows])
    d = c - m
    print("=" * 74)
    print("HELD-OUT LOG-LIKELIHOOD PER PLACEMENT  (higher is better)")
    print("=" * 74)
    print(f"  splits {len(rows)}   held-out placements/split ~{int(np.mean([r['n'] for r in rows]))}")
    print(f"  uniform      {u.mean():+.4f} +/- {u.std():.4f}")
    print(f"  marginal     {m.mean():+.4f} +/- {m.std():.4f}   (vs uniform {m.mean()-u.mean():+.4f})")
    print(f"  conditional  {c.mean():+.4f} +/- {c.std():.4f}   (vs marginal {d.mean():+.4f})")
    print()
    print(f"  paired delta conditional-marginal: {d.mean():+.4f} +/- {d.std():.4f}"
          f"   wins {int((d>0).sum())}/{len(d)} splits")
    tm = np.array([r["top1_marginal"] for r in rows])
    tc = np.array([r["top1_conditional"] for r in rows])
    print(f"  top-1 cell:  marginal {tm.mean():.3f}   conditional {tc.mean():.3f}")
    lv = np.sum([r["levels"] for r in rows], axis=0).astype(float)
    lv /= max(1.0, lv.sum())
    print(f"  backoff used: context {lv[BACKOFF_CONTEXT]:.1%}  "
          f"lane {lv[BACKOFF_LANE]:.1%}  marginal {lv[BACKOFF_MARGINAL]:.1%}")
    print()
    print("=" * 74)
    if d.mean() > 0 and (d > 0).sum() >= 0.8 * len(d):
        print("VERDICT: conditioning EARNS ITS PLACE -- ship the conditional prior.")
    else:
        print("VERDICT: conditioning does NOT beat the marginal out of sample.")
        print("         Ship P(cell|card) alone; the context key is not paying.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    args = ap.parse_args()
    d = np.load(args.table, allow_pickle=True)
    table = {k: d[k] for k in ("episode", "card_slot", "context", "cell", "deck")}
    report(evaluate(table, n_splits=args.splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
