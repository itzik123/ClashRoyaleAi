"""Reading the KataCR replay format, and mapping it onto our engine.

An episode is an lzma-compressed pickled dict with three parallel lists at
~5 fps:

    state[i]  = {'time': int seconds, 'elixir': int|None,
                 'cards': [next, slot1..slot4] as CLASSIFIER indices,
                 'unit_infos': [{'xy','cls','bel','body','bar1','bar2'}]}
    action[i] = {'xy': ndarray|None, 'card_id': 0..4}   # 0 = no action
    reward[i] = float

`cls` indexes KataCR's `label_list.unit_list`; `cards` indexes the CARD
CLASSIFIER's own label set, which is NOT `card_list.py`'s 126-card list -- it is
the sorted directory names of the 2.6-deck classification dataset, with 'empty'
at index 1 (`classification/train.py:EMPTY_CARD_INDEX = 1`).

That mapping is VERIFIED, not assumed: see `tests/test_katacr_card_mapping.py`,
which re-derives it from an episode's own elixir ledger.
"""
from __future__ import annotations

import lzma
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

# Sorted directory names of the 2.6 card-classification dataset. 'empty' lands
# at index 1 by alphabetical order, which is what EMPTY_CARD_INDEX = 1 pins.
KATACR_CARD_CLASSES = [
    "cannon", "empty", "fireball", "hog-rider", "ice-golem", "ice-spirit",
    "ice-spirit-evolution", "musketeer", "skeletons", "skeletons-evolution",
    "the-log",
]
EMPTY_CARD_INDEX = 1

# Their canonical elixir costs, used ONLY by the mapping test as an independent
# oracle -- never to drive the engine, which prices cards itself.
KATACR_CARD_ELIXIR = {
    "cannon": 3, "fireball": 4, "hog-rider": 4, "ice-golem": 2,
    "ice-spirit": 1, "musketeer": 4, "skeletons": 1, "the-log": 2,
    "ice-spirit-evolution": 1, "skeletons-evolution": 1,
}

# Detected classes that are scenery rather than deployed bodies. Our engine's
# unit-count channels do not include towers, so neither may theirs.
NON_BODY_CLASSES = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar",
    "bar-level", "clock", "text", "emote", "elixir", "skeleton-king-bar",
    "background-items", "ruler",
}


@dataclass
class Episode:
    """One parsed match. `fps` is measured, not assumed."""
    path: Path
    state: list[dict[str, Any]]
    action: list[dict[str, Any]]
    reward: np.ndarray
    idx2unit: dict[int, str]

    # Filled by load_episode: seconds = _sec_per_frame * i + _sec_at_zero, and
    # frames at/after n_frames are not gameplay.
    _sec_per_frame: float = 0.2
    _sec_at_zero: float = 0.0
    _n_valid: int = 0

    @property
    def n_frames(self) -> int:
        """Gameplay frames only.

        The recorded `time` field is a clean monotonic second counter for the
        whole match and then RESETS on the final frame or two -- the victory
        screen, where the clock OCR reads a different part of the UI. Deriving
        the frame rate from the endpoints therefore reported 641 fps on one
        episode and 1.2e9 on another, which silently collapsed every timestamp
        to zero. The tail is trimmed here instead.
        """
        return self._n_valid

    @property
    def fps(self) -> float:
        return 1.0 / self._sec_per_frame

    def seconds_at(self, i: int) -> float:
        """Frame index -> match seconds, from a robust fit rather than from
        `state['time']` directly, which is quantised to whole seconds and so
        cannot express a sub-second placement."""
        return self._sec_at_zero + i * self._sec_per_frame

    def card_name_at(self, frame: int, slot: int) -> str:
        """slot is KataCR's 1..4; index 0 of `cards` is the NEXT card."""
        idx = int(self.state[frame]["cards"][slot])
        if 0 <= idx < len(KATACR_CARD_CLASSES):
            return KATACR_CARD_CLASSES[idx]
        return "empty"


def _timebase(state, tol_seconds: float = 3.0, k: int = 100):
    """Robust (seconds-per-frame, offset, n_valid) from the recorded clock.

    The slope is a median of long-baseline differences rather than a least
    squares fit, so an isolated OCR blip cannot move it; `n_valid` then trims
    the trailing frames that disagree with that line by more than `tol`, which
    is what removes the victory screen.
    """
    t = np.asarray([s["time"] for s in state], dtype=np.float64)
    n = len(t)
    k = min(k, max(1, n - 1))
    slopes = (t[k:] - t[:-k]) / k
    spf = float(np.median(slopes)) if len(slopes) else 0.2
    if not np.isfinite(spf) or spf <= 1e-6:
        spf = 0.2                      # documented dataset rate: 5 fps
    off = float(np.median(t - spf * np.arange(n)))
    resid = np.abs(t - (spf * np.arange(n) + off))
    good = np.nonzero(resid <= tol_seconds)[0]
    n_valid = int(good[-1]) + 1 if len(good) else n
    return spf, off, n_valid


def load_episode(path: str | Path, idx2unit: dict[int, str]) -> Episode:
    raw = lzma.open(str(path)).read()
    d = np.load(BytesIO(raw), allow_pickle=True).item()
    spf, off, n_valid = _timebase(d["state"])
    ep = Episode(Path(path), d["state"], d["action"],
                 np.asarray(d["reward"], dtype=np.float32), idx2unit)
    ep._sec_per_frame, ep._sec_at_zero, ep._n_valid = spf, off, n_valid
    return ep


def load_unit_labels(label_list_py: str | Path) -> dict[int, str]:
    """Import KataCR's label_list.py for `idx2unit` without vendoring it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_katacr_labels",
                                                  str(label_list_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.idx2unit)
