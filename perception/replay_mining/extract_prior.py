"""Episodes -> a table of (episode, card, context, cell) placements.

A table, not a histogram: counts are one `np.add.at` away for any subset, which
makes held-out evaluation in `evaluate_prior.py` possible without re-reading
the corpus.

Two refusals:

  - A placement mapping outside our board is dropped and counted, never
    truncated. Their arena is 32 rows to our 34, and the transform is fitted
    on the interior towers; extrapolated to the deep edge it puts the deepest
    placements slightly below y=0, where `int(-0.87) == 0` would move them to
    the back row.
  - A prior is stamped with the arena it was built against and the loader
    refuses a mismatch, as `bc_pretrain.load_dataset` does for
    `observation_size()`.
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

    Truncation, matching `bc_pretrain._cell_from_xy` and the engine's
    `static_cast<int>(position.y)`. Never `round()`: round(15.5) = 16 is past
    the last legal own-half row, a target the mask makes -inf.

    A separate implementation because bc_pretrain pulls in gymnasium, which
    perception's venv lacks; agreement is pinned by test, as
    `advisor_target._standardize` is with `prove_hires.soft_target_logits`.
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


#: The arena stamp comes from python_ai/advisors/human_prior.py, which
#: validates it at load time, so a prior cannot be stamped with one set of
#: constants and checked against another.
from python_ai.advisors.human_prior import geometry_stamp  # noqa: E402


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
