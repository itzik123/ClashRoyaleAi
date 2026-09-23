"""Hand identity against a template pool of our own eight cards.

CRBAB's `CardDetector` hashes each slot crop to 8x8 greyscale and matches it
against its stock icons with a Hungarian assignment. Its pool is already the
deck (eight cards plus five `blank` entries), so off-deck reads are rare; what
remains is within-deck confusion, from two causes:

1. The templates are from the wrong domain. A stock icon is 252x313 and frames
   the whole character; the live slot crop is 61x73, zoomed tighter, and
   carries the magenta cost badge. Corresponding cells of the two 8x8 grids
   hold different content.
2. 8x8 is 64 numbers for a 4,453-pixel crop, applied to eight cards that must
   be told apart from each other.

So these templates are cut from this game's own hand slots, in the live loop's
368x652 `CARD_CONFIG` geometry (`tools/build_icon_templates.py --frames`), and
matched at 48x40.

Contrast-normalised greyscale, not colour: the game renders an unaffordable
card in true greyscale, so a colour feature is absent exactly when the card is
unaffordable. Mean/std normalisation is invariant to that desaturation and the
brightness wash.

Edge cases:

- Empty slot. Between a play and the next card sliding in, the slot shows a
  blue card-back. The cost badge alone cannot decide this, because a dimmed
  card's badge is greyscale too. Over 1,872 in-match slots:

  | bucket | share | best correlation |
  |---|---|---|
  | has badge | 80.3% | median 0.997, p05 0.856 |
  | no badge, greyscale (dimmed card) | 15.1% | median 0.785, p05 0.513 |
  | no badge, coloured (card-back) | 4.6% | median 0.559, max 0.571 |

  A slot holds a card if it shows a badge or is greyscale; correlation then decides identity. The last two rows overlap in correlation, so both signals are needed.
- A card mid-drag leaves a partial render, which lands in the coloured-no-badge
  bucket and reads `blank`. `readers/hand.HandStabiliser` turns a run of those
  back into one play.
- A duplicate cannot occur in a real hand (eight distinct cards, strict FIFO).
  `assign="hungarian"`, the default, forbids it; `assign="argmax"` is kept for
  diagnosis. On 120 labelled crops they tie on accuracy (96.7%) and differ on
  sequence invariants: impossible cycle transitions 16.7% for argmax against
  5.3% for hungarian, duplicates 0.6% against 0.4%.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clashroyalebuildabot.constants import (  # noqa: E402
    CARD_CONFIG,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.namespaces.cards import Card, Cards  # noqa: E402
from readers.hand import ICON_SHAPE, has_cost_badge, normalise_icon  # noqa: E402

#: A pool is per deck, and the default is the deck we play.
#: `deck_pool_hog26` is `gym_wrapper.DEFAULT_DECK` (2.6 Hog Cycle), cut from
#: `assets/recordings/2026-09-02 19-33-22.mp4`. `deck_pool_giant` is the July
#: recordings' deck, which `tests/` and the match_practice_01 benchmark score
#: against. Kept separate: a slot holds one of eight cards, and a sixteen-card
#: pool would give away the design's one advantage over open-set recognition.
DEFAULT_POOL = _ROOT / "config" / "templates" / "deck_pool_hog26"

#: Below this best correlation a slot is reported `blank` rather than
#: guessed. In-match correlations are bimodal (real matches 0.75-0.95,
#: animation frames below 0.35); this sits in the gap.
MIN_SCORE = 0.45

#: A selected card is lifted out of its slot, and a rigid template misses
#: it. Tapping a card raises its art by a stable 11 px of a 73 px slot, and
#: correlation against the unshifted template collapses from ~0.99 to 0.26-0.39
#: while the card stays legible. A card is selected precisely when it is about
#: to be played, so these are the frames placement depends on most. The window
#: is searched only when the direct match is poor.
LIFT_SEARCH = (-14, 3)
SEARCH_TRIGGER = 0.75

#: Mean std across the colour axis below which a slot is rendered in true
#: greyscale. CRBAB uses the same statistic and threshold for "is this card
#: affordable"; it is one question, computed once per slot.
GREY_STD_THRESHOLD = 5.0


class _Window(NamedTuple):
    """One slot's pixels and the two cheap facts derived from them."""

    tall: np.ndarray
    base: int
    height: int
    badge: bool
    colour: float
    present: bool


@dataclass(frozen=True)
class SlotRead:
    card: Card
    score: float
    runner_up: float
    has_badge: bool

    @property
    def margin(self) -> float:
        return self.score - self.runner_up


class DeckPoolMissing(FileNotFoundError):
    """The restricted deck template pool has not been built."""


def load_pool(directory: Path | str = DEFAULT_POOL) -> dict[str, np.ndarray]:
    """Templates keyed by CRBAB card name, from a built pool directory. The pool's
    `icons.json` keys by simulator id; `live/unit_to_card.hand_card_id_for` is
    the one mapping between the two.
    """
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415

    directory = Path(directory)
    index_path = directory / "icons.json"
    if not index_path.exists():
        raise DeckPoolMissing(
            f"no restricted deck pool at {directory}. Build it with:\n"
            f"  perception/.venv/Scripts/python.exe "
            f"perception/tools/build_icon_templates.py \\\n"
            f"      --frames <frame dir> --out {directory} --cluster\n"
            f"then name the clusters with --label."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))

    by_sim_id: dict[int, str] = {}
    for card in vars(Cards).values():
        name = getattr(card, "name", None)
        if not name or name == "blank":
            continue
        by_sim_id.setdefault(hand_card_id_for(name), name)

    pool: dict[str, np.ndarray] = {}
    for sim_id, meta in index.items():
        image = cv2.imread(str(directory / meta["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise DeckPoolMissing(f"could not read {directory / meta['file']}")
        name = by_sim_id.get(int(sim_id))
        if name is None:
            raise DeckPoolMissing(
                f"pool holds simulator id {sim_id} with no CRBAB card")
        pool[name] = image
    return pool


class DeckHandDetector:
    """Drop-in for `CardDetector`: `run(image) -> (cards, ready)`. `cards[0]` is
    the next-card preview and `cards[1:5]` the hand.
    """

    def __init__(
        self,
        deck: list[Card],
        pool_dir: Path | str = DEFAULT_POOL,
        assign: str = "hungarian",
        min_score: float = MIN_SCORE,
    ):
        if assign not in ("argmax", "hungarian"):
            raise ValueError(f"assign must be argmax or hungarian, got {assign!r}")
        self.deck = list(deck)
        self.assign = assign
        self.min_score = min_score

        pool = load_pool(pool_dir)
        missing = [c.name for c in self.deck if c.name not in pool]
        if missing:
            raise DeckPoolMissing(
                f"the deck pool at {pool_dir} has no template for {missing}. "
                "It was built from a recording of a DIFFERENT deck -- rebuild "
                "it from footage of this one, or the loop will read every one "
                "of those cards as whichever of the eight it least mismatches."
            )

        self._names = [c.name for c in self.deck]
        self._cards = {c.name: c for c in self.deck}
        # (n_cards, 48*40), unit-normalised so a dot product is the correlation
        # and the match is one matmul.
        stack = []
        for name in self._names:
            vec = normalise_icon(pool[name]).ravel()
            stack.append(vec / (np.linalg.norm(vec) + 1e-6))
        self._templates = np.asarray(stack, dtype=np.float32)

    def _vec(self, crop_bgr: np.ndarray) -> np.ndarray:
        vec = normalise_icon(crop_bgr).ravel()
        return vec / (np.linalg.norm(vec) + 1e-6)

    def _best_over_lift(self, tall: np.ndarray, base: int, height: int):
        """Best (scores, dy) over the vertical lift window. `tall` is the slot
        crop grown up by -LIFT_SEARCH[0] rows and down by LIFT_SEARCH[1];
        `base` is where the unshifted slot begins inside it.
        """
        direct = self._templates @ self._vec(tall[base:base + height])
        if float(direct.max()) >= SEARCH_TRIGGER:
            return direct, 0

        best, best_dy = direct, 0
        for dy in range(LIFT_SEARCH[0], LIFT_SEARCH[1]):
            if dy == 0:
                continue
            top = base + dy
            if top < 0 or top + height > tall.shape[0]:
                continue
            scores = self._templates @ self._vec(tall[top:top + height])
            if float(scores.max()) > float(best.max()):
                best, best_dy = scores, dy
        return best, best_dy

    def _slot_windows(self, image):
        """Per slot: grown crop, base row, height, and whether a card is there.
        One PIL crop per slot; each lift offset is a numpy slice. `present` is
        the badge-or-greyscale rule.
        """
        out = []
        for x0, y0, x1, y1 in CARD_CONFIG:
            top = max(0, y0 + LIFT_SEARCH[0])
            bottom = min(SCREENSHOT_HEIGHT, y1 + LIFT_SEARCH[1])
            tall = np.ascontiguousarray(
                np.array(image.crop((x0, top, x1, bottom)).convert("RGB"))[:, :, ::-1])
            base = y0 - top
            height = y1 - y0
            crop = tall[base:base + height]
            badge = has_cost_badge(crop)
            colour = float(np.mean(np.std(crop.astype(np.float32), axis=2)))
            present = badge or colour < GREY_STD_THRESHOLD
            out.append(_Window(tall, base, height, badge, colour, present))
        return out

    def _read_window(self, window: _Window, gate_presence: bool) -> SlotRead:
        if gate_presence and not window.present:
            return SlotRead(Cards.BLANK, 0.0, 0.0, window.badge)
        tall, base, height, badge = (window.tall, window.base,
                                     window.height, window.badge)
        scores, _dy = self._best_over_lift(tall, base, height)
        order = np.argsort(scores)[::-1]
        best, second = float(scores[order[0]]), float(scores[order[1]])
        if best < self.min_score:
            return SlotRead(Cards.BLANK, best, second, badge)
        return SlotRead(self._cards[self._names[order[0]]], best, second, badge)

    def _read(self, crop_bgr: np.ndarray, gate_presence: bool) -> SlotRead:
        """One fixed crop, no lift search. For diagnostics and tests."""
        badge = has_cost_badge(crop_bgr)
        colour = float(np.mean(np.std(crop_bgr.astype(np.float32), axis=2)))
        if gate_presence and not (badge or colour < GREY_STD_THRESHOLD):
            return SlotRead(Cards.BLANK, 0.0, 0.0, badge)
        scores = self._templates @ self._vec(crop_bgr)
        order = np.argsort(scores)[::-1]
        best, second = float(scores[order[0]]), float(scores[order[1]])
        if best < self.min_score:
            return SlotRead(Cards.BLANK, best, second, badge)
        return SlotRead(self._cards[self._names[order[0]]], best, second, badge)

    def run(self, image):
        """`image` is a PIL image already at CRBAB's 368x652."""
        if image.size != (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT):
            image = image.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
        windows = self._slot_windows(image)

        if self.assign == "argmax":
            reads = [self._read_window(windows[0], gate_presence=False)]
            reads += [self._read_window(w, gate_presence=True)
                      for w in windows[1:]]
        else:
            reads = self._hungarian(windows)

        ready = self._detect_if_ready(windows[1:])
        return [r.card for r in reads], ready

    def _hungarian(self, windows) -> list[SlotRead]:
        """A bijection over the slots that actually hold a card. Empty slots are
        left out of the assignment, so it cannot be spent on a slot with no
        card. Its value shows on the dimmed cards, where the matcher is least
        certain (module docstring).
        """
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        live = [i for i, w in enumerate(windows) if i == 0 or w.present]
        if not live:
            return [SlotRead(Cards.BLANK, 0.0, 0.0, w.badge) for w in windows]
        scores = np.stack([self._best_over_lift(windows[i].tall, windows[i].base,
                                                windows[i].height)[0]
                           for i in live])
        rows, cols = linear_sum_assignment(-scores)

        reads = [SlotRead(Cards.BLANK, 0.0, 0.0, w.badge) for w in windows]
        for r, c in zip(rows, cols):
            row = scores[r]
            order = np.argsort(row)[::-1]
            best = float(row[c])
            second = float(row[order[1]]) if len(order) > 1 else 0.0
            card = self._cards[self._names[c]] if best >= self.min_score else Cards.BLANK
            reads[live[r]] = SlotRead(card, best, second,
                                      windows[live[r]].badge)
        return reads

    @staticmethod
    def _detect_if_ready(hand_windows):
        """Which hand slots are affordable: CRBAB's `mean(std over the colour
        axis) > 5`, which works because an unaffordable card is rendered in
        true greyscale.
        """
        return [i for i, w in enumerate(hand_windows)
                if w.colour > GREY_STD_THRESHOLD]
