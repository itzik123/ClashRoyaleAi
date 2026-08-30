"""Episodes -> a table of (episode, card, context, cell) placements.

A TABLE, not a histogram. Counts are one `np.add.at` away and can be rebuilt for
any subset, which is what makes the held-out evaluation in `evaluate_prior.py`
possible without re-reading 1.3 GB. The table for the whole corpus is ~12k rows.

Two things this refuses to do quietly, both learned the hard way:

  - A placement that maps OUTSIDE our board is DROPPED and counted, never
    truncated. Their arena is 32 rows and ours is 34, and the transform is
    fitted on the four towers, which are interior; extrapolated to the deep
    edge it puts the deepest human placements at y slightly below 0, where
    `int(-0.87) == 0` silently relocates them to the back row. Measured before
    this guard existed: that artifact alone made (8, 0) the Musketeer's modal
    cell with 7.6% of its mass.
  - A prior is stamped with the arena it was built against and the loader
    refuses a mismatch, the way `bc_pretrain.load_dataset` refuses a dataset
    recorded against a different `observation_size()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import clash_royale_env as E

from .context import N_CONTEXTS, context_from_replay
from .events import _engine_card_names, ego_events
from .geometry import fit_transform
from .katacr_format import load_episode, load_unit_labels

BOARD_H, BOARD_W = E.ClashRoyaleEnv.BOARD_HEIGHT, E.ClashRoyaleEnv.BOARD_WIDTH
N_CELLS = BOARD_H * BOARD_W


def cell_from_xy(x: float, y: float):
    """Engine coordinates -> cell index, or None if outside the board.

    TRUNCATION, matching `bc_pretrain._cell_from_xy` and the engine's own
    `static_cast<int>(position.y)`. Never `round()`: round(15.5) = 16 is one row
    past the last legal own-half row, and the resulting target can never be
    matched because the mask makes that cell -inf.

    Deliberately a separate implementation from bc_pretrain's rather than an
    import -- that module pulls in gym_wrapper and through it gymnasium, which
    perception's venv does not have. Agreement is pinned by test instead, the
    same arrangement `advisor_target._standardize` has with
    `prove_hires.soft_target_logits`.
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    col, row = int(x), int(y)
    if not (0 <= col < BOARD_W and 0 <= row < BOARD_H):
        return None
    if x < 0.0 or y < 0.0:          # int() truncates toward zero, so -0.9 -> 0
        return None
    return row * BOARD_W + col


@dataclass
class Census:
    per_card: dict = field(default_factory=dict)
    out_of_board: dict = field(default_factory=dict)
    illegal: dict = field(default_factory=dict)
    residuals: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def extract(paths, labels_py, deck, drop_illegal=True):
    """-> (table dict of arrays, Census)."""
    idx2unit = load_unit_labels(labels_py)
    names = _engine_card_names()
    deck = list(deck)
    card_slot = {cid: i for i, cid in enumerate(deck)}
    env = E.ClashRoyaleEnv(deck, deck, max_ticks=3600)
    env.seed(0)
    env.reset()

    ep_i, cards, ctxs, cells = [], [], [], []
    cen = Census()
    for e_idx, p in enumerate(paths):
        try:
            ep = load_episode(p, idx2unit)
            T = fit_transform(ep)
        except Exception as exc:                       # noqa: BLE001
            cen.failed.append((Path(p).name, str(exc)))
            continue
        cen.episodes.append(Path(p).name)
        cen.residuals.append(T.residual_tiles)
        for ev in ego_events(ep, T):
            cid = names.get(ev.card)
            if cid is None or cid not in card_slot:
                continue
            cen.per_card[ev.card] = cen.per_card.get(ev.card, 0) + 1
            cell = cell_from_xy(ev.x, ev.y)
            if cell is None:
                cen.out_of_board[ev.card] = cen.out_of_board.get(ev.card, 0) + 1
                continue
            col, row = cell % BOARD_W, cell // BOARD_W
            if not env.is_valid_placement(cid, float(col), float(row), 0):
                cen.illegal[ev.card] = cen.illegal.get(ev.card, 0) + 1
                if drop_illegal:
                    continue
            frame = int(round((ev.t - ep.seconds_at(0)) * ep.fps))
            frame = int(np.clip(frame, 0, ep.n_frames - 1))
            ep_i.append(e_idx)
            cards.append(card_slot[cid])
            ctxs.append(context_from_replay(ep, frame, T))
            cells.append(cell)

    table = {
        "episode": np.asarray(ep_i, dtype=np.int32),
        "card_slot": np.asarray(cards, dtype=np.int32),
        "context": np.asarray(ctxs, dtype=np.int32),
        "cell": np.asarray(cells, dtype=np.int32),
        "deck": np.asarray(deck, dtype=np.int32),
    }
    return table, cen


def geometry_stamp() -> dict:
    """What the prior is only valid against. Any change here invalidates it."""
    CE = E.ClashRoyaleEnv
    return {
        "board_h": BOARD_H, "board_w": BOARD_W,
        "arena_center_x": E.ARENA_CENTER_X, "arena_bridge_y": E.ARENA_BRIDGE_Y,
        "left_lane_x": E.ARENA_LEFT_LANE_X, "right_lane_x": E.ARENA_RIGHT_LANE_X,
        "left_bridge_x": E.ARENA_LEFT_BRIDGE_X,
        "right_bridge_x": E.ARENA_RIGHT_BRIDGE_X,
        "king_y0": E.arena_king_y(0), "princess_y0": E.arena_princess_y(0),
        "n_contexts": N_CONTEXTS, "n_cells": N_CELLS,
        "num_card_ids": CE.NUM_CARD_IDS,
    }


def save_table(path, table, cen):
    stamp = geometry_stamp()
    np.savez_compressed(
        path, **table,
        geometry_keys=np.asarray(list(stamp), dtype=object),
        geometry_vals=np.asarray([stamp[k] for k in stamp], dtype=np.float64),
        episodes=np.asarray(cen.episodes, dtype=object),
        residuals=np.asarray(cen.residuals, dtype=np.float32),
    )


def build_counts(table, mask=None):
    """(n_cards, N_CONTEXTS, N_CELLS) int32 counts over a row subset."""
    n_cards = len(table["deck"])
    out = np.zeros((n_cards, N_CONTEXTS, N_CELLS), dtype=np.int32)
    sel = slice(None) if mask is None else mask
    np.add.at(out, (table["card_slot"][sel], table["context"][sel],
                    table["cell"][sel]), 1)
    return out
