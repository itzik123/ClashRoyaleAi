"""The mined human placement prior, served as advisor-shaped target logits.

WHAT THIS IS. `perception/replay_mining/` mined 10,718 labelled placements from
221 episodes of expert 2.6 Hog Cycle play and reduced them to a per-card
distribution over the 612 placement cells. This module serves that distribution
into training through the same seam the hand-written advisor uses.

WHY IT IS A MARGINAL AND NOT A CONDITIONAL. The design called for
P(cell | card, context) with a backoff chain. `evaluate_prior.py` measured it on
held-out episodes and the conditioning does not pay for itself: mean held-out
log-likelihood per placement over 5 episode-level splits was -4.6368 for the
marginal against -4.7782 for an 18-way context key, and only a 3-way lane key
beat the marginal at all -- by +0.0075 against a split-to-split spread of ~0.03.
Splitting 10.7k placements 18 ways leaves ~74 per bucket against 612 cells, and
the variance costs more than the bias buys.

The MARGINAL, though, is a large win: +0.930 nats over uniform, a factor of 2.53
on the likelihood of a real human placement, consistent on all five splits, with
top-1 cell 0.081 against a uniform-over-legal 0.0038.

WHAT IT IS FOR. Five of the eight cards in DEFAULT_DECK -- Musketeer, Ice Golem,
Skeletons, Ice Spirit, The Log -- have no advisor rule, so the coverage term
gives them an ENTROPY bonus and nothing else. `advisor_target`'s own docstring
says why that fails: entropy is a marginal objective, it says "be spread out"
and never "be here", and the argmax of a flat map is an arbitrary constant. This
replaces that uniform pull with one measured to be 2.53x better than uniform at
predicting what a strong human actually does.

WHAT IT IS NOT. At the marginal level the prior teaches WHERE humans put a card,
not WHEN or in response to WHAT. It is strictly more than uniform and strictly
less than a state-conditional rule, which is exactly why it is a FALLBACK: where
`tactics` has an engine-validated rule, that rule wins. The prior comes from a
game whose physics this engine does not reproduce -- the same divergence probe
that produced this data established that in the strongest possible terms -- so
it must never override a rule measured against this engine.

IT SPEAKS UNCONDITIONALLY, AND THAT NEEDS DEFENDING, because `advisor_target`
is emphatic that a source which always answers "teaches a CONSTANT" -- the exact
pathology the coverage term exists to repair. Measured over 495 decision states
carrying a live threat, the coverage-slot target rate is:

    prior OFF   23.4%   Cannon 0.94, Fireball 0.95, Hog 0.02, other five 0.00
    prior ON   100.0%   every card, every row

The defence is that the rows it newly covers were not previously getting a
state-DEPENDENT target. They were getting the ENTROPY BONUS, which is itself
state-independent -- it pulls every board toward the same uniform map. So this
replaces one state-independent pull with a strictly better one, and introduces
state-independence exactly nowhere it did not already exist.

What it does change is STRENGTH: a human marginal is much sharper than uniform,
so it pulls harder. That is what HUMAN_PRIOR_COEF is for, and it is why the
measurement to watch in the A/B is per-card MODAL SHARE rather than entropy --
CLAUDE.md records that an aggregate entropy provably cannot see a per-card
collapse, and flagged the healthiest card in the deck while clearing three dead
ones.
"""
import os
from pathlib import Path

import numpy as np

import clash_royale_env as E

from python_ai import PACKAGE_DIR

#: The mined artifact. Lives with its CONSUMER: `perception` produces it, but
#: python_ai must not import perception -- the dependency runs the other way.
ARTIFACT_PATH = Path(PACKAGE_DIR) / "advisors" / "data" / "hog26_placement_prior.npz"

BOARD_H = E.ClashRoyaleEnv.BOARD_HEIGHT
BOARD_W = E.ClashRoyaleEnv.BOARD_WIDTH
N_CELLS = BOARD_H * BOARD_W

#: Measured on held-out episodes; see the module docstring. Both are pinned by
#: `perception/tests/test_placement_prior.py`, which fails if either moves
#: without the measurement being redone.
BLUR_SIGMA = 1.0
LAPLACE = 0.5

#: How hard the prior pulls RELATIVE to a hand-written advisor target, which
#: carries weight 1.0. The overall scale is still ADVISOR_COVERAGE_COEF, so this
#: is a ratio and not a second global knob.
#:
#: DEFAULT 0 -- OFF. The A/B that has to justify this term runs both arms from
#: byte-identical code and flips only this variable, which is the arrangement
#: PLACEMENT_COVERAGE_COEF's own comment mandates and the only way the
#: comparison attributes a difference to the term rather than to two scripts.
HUMAN_PRIOR_COEF = float(os.environ.get("CLASH_HUMAN_PRIOR_COEF", 0.0))

_NEG_INF = float("-inf")
_cache = {}


def enabled():
    return HUMAN_PRIOR_COEF > 0.0 and ARTIFACT_PATH.exists()


def geometry_stamp():
    """The arena this prior is only valid against.

    Lives here rather than in the extractor so there is ONE definition; the
    extractor imports it. A prior built against a different board would keep
    serving plausible-looking cells for an arena that no longer exists, and
    CLAUDE.md records the arena moving twice.
    """
    CE = E.ClashRoyaleEnv
    return {
        "board_h": BOARD_H, "board_w": BOARD_W,
        "arena_center_x": E.ARENA_CENTER_X, "arena_bridge_y": E.ARENA_BRIDGE_Y,
        "left_lane_x": E.ARENA_LEFT_LANE_X, "right_lane_x": E.ARENA_RIGHT_LANE_X,
        "left_bridge_x": E.ARENA_LEFT_BRIDGE_X,
        "right_bridge_x": E.ARENA_RIGHT_BRIDGE_X,
        "king_y0": E.arena_king_y(0), "princess_y0": E.arena_princess_y(0),
        "n_contexts": 18, "n_cells": N_CELLS,
        "num_card_ids": CE.NUM_CARD_IDS,
    }


def _blur_matrix(n, sigma):
    """(n, n) edge-clamped Gaussian smoothing operator along one axis.

    Row-normalised, so an output cell near the wall averages only over cells
    that exist. The alternative -- padding by replicating the edge value --
    duplicates mass that is not there, which matters when the result is about
    to become a probability distribution.

    A matrix rather than a per-row convolution because this runs on the rollout
    path: the first version was slow enough to time out an offline sweep.
    """
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2)
    k[np.abs(d) > 3 * sigma] = 0.0
    return k / k.sum(axis=1, keepdims=True)


def blur(flat, sigma=BLUR_SIGMA):
    """Separable Gaussian over the board grid. scipy-free by design: this is on
    the training path and perception's venv does not carry scipy either."""
    if sigma <= 0:
        return np.asarray(flat, dtype=np.float64)
    m = np.asarray(flat, dtype=np.float64).reshape(BOARD_H, BOARD_W)
    a, b = _blur_matrix(BOARD_H, sigma), _blur_matrix(BOARD_W, sigma)
    return (a @ m @ b.T).reshape(-1)


def _load():
    """{card_id: (N_CELLS,) float64 smoothed counts}, or {} if unavailable.

    Smoothing is done ONCE at load rather than per call: the surface is a
    marginal, so it does not depend on the observation and there is nothing to
    recompute per step. Legality is applied per call, because it does depend on
    the card.
    """
    if _cache:
        return _cache["surfaces"]
    if not ARTIFACT_PATH.exists():
        _cache["surfaces"] = {}
        return _cache["surfaces"]
    d = np.load(ARTIFACT_PATH, allow_pickle=True)

    keys = [str(k) for k in d["geometry_keys"]]
    vals = dict(zip(keys, [float(v) for v in d["geometry_vals"]]))
    live = geometry_stamp()
    for k, v in live.items():
        if k not in vals or abs(vals[k] - float(v)) > 1e-6:
            raise ValueError(
                f"{ARTIFACT_PATH.name} was built against a different arena "
                f"({k}: prior {vals.get(k)!r} vs engine {v!r}). Re-run "
                f"perception/replay_mining/extract_prior.py; do not reinterpret "
                f"a stale prior.")

    counts = np.asarray(d["counts"], dtype=np.float64)   # (cards, contexts, cells)
    deck = [int(c) for c in d["deck"]]
    surfaces = {}
    for i, cid in enumerate(deck):
        marginal = counts[i].sum(axis=0)                 # SERVE_KEY == "marginal"
        if marginal.sum() <= 0:
            continue
        surfaces[cid] = blur(marginal) + LAPLACE
    _cache["surfaces"] = surfaces
    return surfaces


def prior_cards():
    """Card ids this prior can speak for; EMPTY when the prior is disabled.

    The `enabled()` check is load-bearing, not defensive. `advisor_target`
    derives both `build_legal_table` and `slot_weights_for` from this set, and
    `slot_weights_for` gives ADVISOR_SLOT_WEIGHT to every card in it. Reporting
    the eight deck cards while the prior is OFF therefore made the coverage draw
    uniform over all eight -- measured [5, 5, 5, 5] on a hand where the old
    behaviour is [1, 5, 1, 1] -- which silently changed the CONTROL arm of the
    very A/B this gate exists to keep clean.
    """
    if not enabled():
        return set()
    try:
        return set(_load())
    except ValueError:
        return set()


def board_is_quiet(obs):
    """True when there is nothing on the board to react to.

    The one gate the prior keeps. `validate_pipeline`'s advisor section asserts
    that a target source DECLINES on an empty board, "or it teaches a constant",
    and that check is backed by the measured 2026-08-14 placement collapse -- it
    is the project's invariant, not a stylistic preference, so the prior honours
    it rather than arguing with it.

    The cost of honouring it is close to zero. A "quiet" board here is the empty
    one at episode start, where nothing should be placed at all; declining there
    removes a negligible number of training rows. What it buys is that the prior
    no longer speaks on states that carry no information, which is the strongest
    form of the "a source that always answers teaches a constant" objection.
    """
    from python_ai.advisors import tactics
    return float(tactics.enemy_hp_map(obs).sum()) <= 0.0


def logits_for(card_id, legal, obs=None, T=None):
    """(N_CELLS,) float32 target logits for `card_id`, or None.

    The surface itself has NO state dependence -- it is a marginal, so there is
    nothing to read off the board. `obs` is used only by the quiet-board gate
    above, and passing it is what keeps this source inside the invariant
    `validate_pipeline` enforces. A caller that omits it gets the ungated
    surface, which is correct for offline inspection and wrong for training;
    `advisor_target` always passes it.

    Legality is applied AFTER smoothing. Blurring first and masking second lets
    a human placement near a legal/illegal boundary contribute to its legal
    neighbours; masking first would discard that.
    """
    if not enabled():
        return None
    if obs is not None and board_is_quiet(obs):
        return None
    surfaces = _load()
    surf = surfaces.get(int(card_id))
    if surf is None:
        return None
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    if not legal.any():
        return None
    p = np.where(legal, surf, 0.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0.0:
        return None
    p = p / total
    out = np.full(N_CELLS, _NEG_INF, dtype=np.float32)
    # Temperature divides a LOG-PROBABILITY here, not a standardized score as in
    # advisor_target._standardize. The prior is already a distribution, so its
    # own log is the natural logit and T=1 reproduces it exactly.
    out[legal] = (np.log(p[legal]) / max(1e-6, 1.0 if T is None else T)).astype(np.float32)
    return out
