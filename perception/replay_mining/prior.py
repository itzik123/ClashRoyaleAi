"""Counts -> a served placement surface.

What is served is the marginal P(cell | card), a measured result. A
context-conditional prior with a backoff chain was tested on held-out episodes
(`evaluate_prior.py`: 5 episode-level splits, 221 episodes, 10,718 placements,
mean held-out log-likelihood per placement):

    key            K    sigma=1.0    vs marginal
    uniform        -      -5.5667      -0.9299
    MARGINAL       1      -4.6368      +0.0000
    threat         2      -4.6455      -0.0087
    phase          2      -4.6629      -0.0261
    lane           3      -4.6293      +0.0075
    half           3      -4.6576      -0.0208
    full          18      -4.7782      -0.1414

The prior itself is a large win: +0.930 nats over uniform (a factor of 2.53 on
a real placement's likelihood, consistent across splits), top-1 cell 0.081
against a uniform-over-legal 0.0038. The context key is not: 18 buckets leave
~74 placements each against a 612-cell grid, and only `lane` beats the
marginal, by about one split-to-split sigma. The context machinery stays so the
comparison can be re-run with more data; changing `SERVE_KEY` needs a new
measurement.

sigma is 1.0 for the same reason: 2.0 and 3.0 measured worse at every key.

Smoothing is a Gaussian blur plus Laplace: a per-card histogram is sparse over
612 cells, human intent is spatially smooth, and the KL in
`advisor_target.masked_kl_elementwise` needs strictly positive mass on every
legal cell.
"""
from __future__ import annotations

import numpy as np

from .context import N_HALF, N_PHASE, decode
from .extract_prior import BOARD_H, BOARD_W, N_CELLS

MIN_SAMPLES = 30
BLUR_SIGMA = 1.0
LAPLACE = 0.5

#: The key the prior is served on. "marginal" is the measured answer;
#: "context" and "lane" are kept so the comparison can be re-run.
SERVE_KEY = "marginal"

BACKOFF_CONTEXT, BACKOFF_LANE, BACKOFF_MARGINAL, BACKOFF_NONE = 0, 1, 2, 3


def _lane_sibling_contexts(ctx: int) -> list[int]:
    """Every context sharing this one's lane: the middle backoff level."""
    lane, _, _ = decode(ctx)
    return [(lane * N_HALF + h) * N_PHASE + p
            for h in range(N_HALF) for p in range(N_PHASE)]


# The blur and geometry stamp live in python_ai/advisors/human_prior.py, where
# training consumes them; perception may import python_ai, so there is one
# implementation of each.
from python_ai.advisors.human_prior import blur as _blur  # noqa: F401


def select_counts(counts, card_slot: int, ctx: int, min_samples: int = MIN_SAMPLES,
                  key_mode: str = None):
    """-> (counts over cells, backoff level actually used). `key_mode` selects the
    starting level of the backoff chain; the default "marginal" collapses the
    context axis immediately.
    """
    key_mode = SERVE_KEY if key_mode is None else key_mode
    c = counts[card_slot]
    if key_mode == "marginal":
        marg = c.sum(axis=0)
        if marg.sum() >= min_samples:
            return marg.astype(np.float64), BACKOFF_MARGINAL
        return None, BACKOFF_NONE
    if key_mode == "lane":
        lane = c[_lane_sibling_contexts(ctx)].sum(axis=0)
        if lane.sum() >= min_samples:
            return lane.astype(np.float64), BACKOFF_LANE
        marg = c.sum(axis=0)
        if marg.sum() >= min_samples:
            return marg.astype(np.float64), BACKOFF_MARGINAL
        return None, BACKOFF_NONE
    exact = c[ctx]
    if exact.sum() >= min_samples:
        return exact.astype(np.float64), BACKOFF_CONTEXT
    lane = c[_lane_sibling_contexts(ctx)].sum(axis=0)
    if lane.sum() >= min_samples:
        return lane.astype(np.float64), BACKOFF_LANE
    marg = c.sum(axis=0)
    if marg.sum() >= min_samples:
        return marg.astype(np.float64), BACKOFF_MARGINAL
    return None, BACKOFF_NONE


def probabilities(counts, card_slot, ctx, legal=None,
                  min_samples=MIN_SAMPLES, sigma=BLUR_SIGMA, laplace=LAPLACE,
                  key_mode=None):
    """-> (p over N_CELLS summing to 1 on `legal`, backoff level) or (None, ...).
    `legal` is applied after smoothing, so a placement near a legality boundary
    still contributes to its legal neighbours.
    """
    raw, level = select_counts(counts, card_slot, ctx, min_samples, key_mode)
    if raw is None:
        return None, level
    p = _blur(raw, sigma) + laplace
    if legal is not None:
        legal = np.asarray(legal, dtype=bool).reshape(-1)
        if not legal.any():
            return None, BACKOFF_NONE
        p = np.where(legal, p, 0.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return None, BACKOFF_NONE
    return p / total, level


def logits(counts, card_slot, ctx, legal, T=1.0, **kw):
    """Target logits shaped for `advisor_target`: -inf off `legal`."""
    p, level = probabilities(counts, card_slot, ctx, legal, **kw)
    if p is None:
        return None, level
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    out = np.full(N_CELLS, float("-inf"), dtype=np.float32)
    out[legal] = (np.log(p[legal]) / max(1e-6, T)).astype(np.float32)
    return out, level
