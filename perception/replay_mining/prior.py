"""Counts -> a served placement surface.

WHAT IS SERVED IS THE MARGINAL, P(cell | card), AND THAT IS A MEASURED RESULT
RATHER THAN A SIMPLIFICATION. The design called for a context-conditional prior
with a backoff chain; `evaluate_prior.py` tested it on held-out episodes and the
conditioning does not pay for itself. Mean held-out log-likelihood per
placement, 5 episode-level splits, 221 episodes / 10,718 placements:

    key            K    sigma=1.0    vs marginal
    uniform        -      -5.5667      -0.9299
    MARGINAL       1      -4.6368      +0.0000
    threat         2      -4.6455      -0.0087
    phase          2      -4.6629      -0.0261
    lane           3      -4.6293      +0.0075
    half           3      -4.6576      -0.0208
    full          18      -4.7782      -0.1414

Two readings, and the second is the one that decides the module:

  - The PRIOR ITSELF IS A LARGE WIN. +0.930 nats over uniform is a factor of
    2.53 on the likelihood of a real human placement, consistent across all
    five splits. Top-1 cell is 0.081 against a uniform-over-legal 0.0038 -- 21x.
  - THE CONTEXT KEY IS NOT. Splitting 10.7k placements 18 ways leaves ~74 per
    bucket against a 612-cell grid, and the variance costs more than the bias
    buys. Only `lane` beats the marginal at all, by +0.0075 against a
    split-to-split spread of ~0.03 -- about one sigma, i.e. nothing.

So the context machinery stays (it is what produced the measurement, and the
table records a context per placement so this can be revisited with more data),
but `SERVE_KEY` is "marginal" and changing it needs a new measurement, not an
opinion.

sigma is 1.0 for the same reason: 2.0 and 3.0 were both measured worse at every
key, and top-1 collapses from 0.081 to 0.034 by sigma=2.

Smoothing is a Gaussian blur plus Laplace. A per-card histogram is sparse over
612 cells, a human's intent is spatially smooth ("the Cannon goes about here"),
and the blur guarantees strictly positive mass on every legal cell -- which the
KL in `advisor_target.masked_kl_elementwise` needs, since a zero there is a
-inf log.
"""
from __future__ import annotations

import numpy as np

from .context import N_HALF, N_PHASE, decode
from .extract_prior import BOARD_H, BOARD_W, N_CELLS

MIN_SAMPLES = 30
BLUR_SIGMA = 1.0
LAPLACE = 0.5

#: Which key the prior is SERVED on. "marginal" is the measured answer; see the
#: module docstring. "context" and "lane" are kept so the comparison can be
#: re-run, not because either is recommended.
SERVE_KEY = "marginal"

BACKOFF_CONTEXT, BACKOFF_LANE, BACKOFF_MARGINAL, BACKOFF_NONE = 0, 1, 2, 3


def _lane_sibling_contexts(ctx: int) -> list[int]:
    """Every context sharing this one's lane -- the middle backoff level."""
    lane, _, _ = decode(ctx)
    return [(lane * N_HALF + h) * N_PHASE + p
            for h in range(N_HALF) for p in range(N_PHASE)]


# The blur and the geometry stamp live in python_ai/advisors/human_prior.py,
# which is where they are CONSUMED during training. perception may import
# python_ai (that is the allowed direction), so importing them keeps exactly one
# implementation of each -- the alternative is the second copy of an engine-side
# constant this project forbids.
from python_ai.advisors.human_prior import blur as _blur  # noqa: F401


def select_counts(counts, card_slot: int, ctx: int, min_samples: int = MIN_SAMPLES,
                  key_mode: str = None):
    """-> (counts over cells, backoff level actually used).

    `key_mode` selects the STARTING level of the backoff chain. At the default
    "marginal" the context axis is collapsed immediately, which is what the
    held-out measurement says to do.
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

    `legal` is applied AFTER smoothing: blurring first and masking second keeps
    a human placement near a legal/illegal boundary contributing to its legal
    neighbours, which masking first would throw away.
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
