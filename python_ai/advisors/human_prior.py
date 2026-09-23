"""The mined human placement prior, served as advisor-shaped target logits.

`perception/replay_mining/` reduced 10,718 labelled placements from 221
episodes of expert 2.6 Hog Cycle play to a per-card distribution over the 612
cells. A marginal, not a context-conditional: on held-out episodes conditioning
did not pay for its variance, while the marginal beats uniform by 0.93 nats per
placement.

It fills in for cards without an advisor rule, which otherwise get only the
entropy bonus. It teaches where humans put a card, not when, so a rule always
wins where one exists. It speaks on every non-quiet row, but those rows
previously got the entropy bonus, which is just as state-independent; the
difference is that the prior pulls harder, hence HUMAN_PRIOR_COEF, and per-card
modal share is the thing to watch.
"""
import os
from pathlib import Path

import numpy as np

import clash_royale_env as E

from python_ai import PACKAGE_DIR

#: Lives with its consumer: python_ai must not import perception.
ARTIFACT_PATH = Path(PACKAGE_DIR) / "advisors" / "data" / "hog26_placement_prior.npz"

BOARD_H = E.ClashRoyaleEnv.BOARD_HEIGHT
BOARD_W = E.ClashRoyaleEnv.BOARD_WIDTH
N_CELLS = BOARD_H * BOARD_W

#: Measured on held-out episodes; pinned by
#: `perception/tests/test_placement_prior.py`.
BLUR_SIGMA = 1.0
LAPLACE = 0.5

#: How hard the prior pulls relative to a rule target (weight 1.0); the overall
#: scale is still ADVISOR_COVERAGE_COEF. Off by default: the A/B that must
#: justify it flips only this variable.
HUMAN_PRIOR_COEF = float(os.environ.get("CLASH_HUMAN_PRIOR_COEF", 0.0))

_NEG_INF = float("-inf")
_cache = {}


def enabled():
    return HUMAN_PRIOR_COEF > 0.0 and ARTIFACT_PATH.exists()


def geometry_stamp():
    """The arena this prior is valid against. One definition, which the extractor
    imports; a prior from another arena would serve plausible cells on a board
    that no longer exists.
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
    """(n, n) edge-clamped Gaussian smoothing along one axis, row-normalised so
    cells near a wall average only over cells that exist. A matrix for speed on
    the rollout path.
    """
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2)
    k[np.abs(d) > 3 * sigma] = 0.0
    return k / k.sum(axis=1, keepdims=True)


def blur(flat, sigma=BLUR_SIGMA):
    """Separable Gaussian over the board grid, without scipy (not in either venv's
    training path).
    """
    if sigma <= 0:
        return np.asarray(flat, dtype=np.float64)
    m = np.asarray(flat, dtype=np.float64).reshape(BOARD_H, BOARD_W)
    a, b = _blur_matrix(BOARD_H, sigma), _blur_matrix(BOARD_W, sigma)
    return (a @ m @ b.T).reshape(-1)


def _load():
    """{card_id: (N_CELLS,) float64 smoothed counts}, or {} if unavailable.

    Smoothed once at load: the surface does not depend on the observation.
    Legality is applied per call.
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
        marginal = counts[i].sum(axis=0)                 # the marginal over contexts
        if marginal.sum() <= 0:
            continue
        surfaces[cid] = blur(marginal) + LAPLACE
    _cache["surfaces"] = surfaces
    return surfaces


def prior_cards():
    """Card ids this prior can speak for; empty when disabled.

    The `enabled()` check matters: `advisor_target` weights the coverage draw
    by this set, so reporting cards while the prior is off would change the
    A/B's control arm.
    """
    if not enabled():
        return set()
    try:
        return set(_load())
    except ValueError:
        return set()


def board_is_quiet(obs):
    """True when there is nothing on the board to react to.

    The prior declines there, honouring the invariant `validate_pipeline`
    checks for every target source (a source that speaks on an empty board
    teaches a constant). It costs almost nothing: such states are the empty
    board at episode start.
    """
    from python_ai.advisors import tactics
    return float(tactics.enemy_hp_map(obs).sum()) <= 0.0


def logits_for(card_id, legal, obs=None, T=None):
    """(N_CELLS,) float32 target logits for `card_id`, or None.

    The surface has no state dependence; `obs` only feeds the quiet-board gate
    (omit it only for offline inspection). Legality is applied after smoothing,
    so placements near a boundary still inform their legal neighbours.
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
    # T divides a log-probability here (the prior is already a distribution),
    # so T=1 reproduces it.
    out[legal] = (np.log(p[legal]) / max(1e-6, 1.0 if T is None else T)).astype(np.float32)
    return out
