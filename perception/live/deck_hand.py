"""Hand identity against a template pool restricted to OUR OWN EIGHT CARDS.

WHAT THIS REPLACES, AND WHAT IT DOES NOT
----------------------------------------
The live loop reads its hand from the vendored CRBAB `CardDetector`: every
slot crop is squashed to an **8x8 greyscale hash** (64 numbers) and compared
against the stock `images/cards/*.jpg` icons, with a Hungarian assignment
picking a bijection.

**Its candidate pool was ALREADY restricted to the deck** -- `CardDetector`
loads only the eight cards it is handed, plus five `blank` entries, so it
matches against thirteen templates and not the 139 icons in the images
directory. Shrinking the pool is therefore not available as a fix; it is
already done, and the measurement agrees: off-deck reads are **0.5%** of
in-match frames, because a card outside the deck is not a candidate at all.

The error that remains is **within-deck confusion**, and it has two causes,
both of which this module addresses and neither of which is pool size:

1. **The templates are from the wrong domain.** A stock icon is 252x313 and
   frames the whole character with background; the live slot crop is 61x73,
   zoomed much tighter, and carries the magenta elixir-cost badge that the
   stock icon does not have at all. Corresponding cells of the two 8x8 grids
   therefore hold different content, and no matcher recovers that.
2. **8x8 greyscale is 64 numbers for a 4,453-pixel crop** -- a 70x
   downsample, applied to eight cards that must be told apart from each
   other rather than from the world.

So the templates here are **cut from this game's own hand slots**, in the
live loop's own 368x652 `CARD_CONFIG` geometry, by
`tools/build_icon_templates.py --frames`, and matched at 48x40 rather than
8x8.

WHY GREYSCALE SURVIVES, AND COLOUR DOES NOT
-------------------------------------------
Colour is the obvious thing to add and it is a trap. An unaffordable slot is
not merely darkened -- the game renders it in **actual greyscale**. Measured
on `assets/live/match_practice_01` frame 372, all four slots are fully
desaturated at once, and at frame 453 only the Giant (cost 5) is, the other
three being affordable. A colour feature is therefore *absent exactly when
the card is unaffordable*, which is a large fraction of every match, so a
colour matcher would fail in bursts correlated with low elixir.

Contrast-normalised greyscale -- subtract the mean, divide by the standard
deviation -- is invariant to both the desaturation and the brightness
wash, which is why `tools/build_icon_templates.py` clusters in it and why the
clusters come out clean.

THE THREE EDGE CASES
--------------------
- **Empty slot.** Between playing a card and the next one sliding in, the
  slot shows a blue card-back with a crown, not a card. It is reported
  `blank`.

  **The cost badge ALONE cannot decide this, and using it alone was a real
  defect in the first version of this module.** The badge test asks for
  MAGENTA, and an unaffordable card is rendered in true greyscale *including
  its badge* -- so every dimmed card failed the test and was reported empty.
  Measured over 1,872 in-match slots: 80.3% carry a badge, **15.1% are dimmed
  cards with no magenta left**, and only 4.6% are genuinely empty. The gate
  was throwing away one slot in seven, exactly when elixir was low.

  The two no-badge causes separate cleanly on COLOUR, because they differ in
  it by construction -- a dimmed card is greyscale, the card-back is
  saturated blue:

  | bucket | share | best correlation |
  |---|---|---|
  | has badge | 80.3% | median 0.997, p05 0.856 |
  | no badge, greyscale (dimmed card) | 15.1% | median 0.785, p05 0.513 |
  | no badge, coloured (card-back) | 4.6% | median 0.559, **max 0.571** |

  So a slot holds a card if it shows a badge OR is greyscale, and correlation
  then decides identity. Note the third row's ceiling of 0.571 against the
  second row's p05 of 0.513: the two OVERLAP, so a correlation threshold on
  its own could not have separated them either. Both signals are needed.
- **A card mid-drag** lifts out of its slot, leaving a partial render. Those
  land in the coloured-no-badge bucket and read `blank` rather than being
  forced onto whichever card they least mismatch. That is the honest answer,
  and `readers/hand.HandStabiliser` is what turns a run of them back into one
  play.
- **A duplicate** cannot happen in a real hand: eight distinct cards in a
  strict FIFO. `assign="hungarian"` forbids it by construction and is the
  DEFAULT; `assign="argmax"` permits one and is kept for diagnosis. On the
  full 120-crop set they tie on labelled accuracy (96.7%) and separate on the
  sequence invariants: impossible cycle transitions **16.7% for argmax
  against 5.3% for hungarian**, duplicates 0.6% against 0.4%.

  That separation only appeared once unaffordable cards were read at all --
  measured against the badge-gated sample the two were indistinguishable. The
  constraint pays off precisely on the slots the matcher is least sure of,
  which is the population the first sample had excluded.
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

#: A POOL IS PER DECK, AND THE DEFAULT IS THE DECK WE PLAY.
#:
#: `deck_pool_hog26` is `gym_wrapper.DEFAULT_DECK` -- the 2.6 Hog Cycle -- cut
#: from `assets/recordings/2026-09-02 19-33-22.mp4`. `deck_pool_giant` is the
#: July recordings' deck and is what `tests/` and the match_practice_01
#: benchmark score against. They are separate directories rather than one
#: merged pool on purpose: a hand slot holds one of EIGHT cards, and widening
#: the candidate set to sixteen would give away the one advantage this design
#: has over open-set recognition.
DEFAULT_POOL = _ROOT / "config" / "templates" / "deck_pool_hog26"

#: A slot whose best correlation falls below this is reported `blank` rather
#: than guessed. Correlation of contrast-normalised crops: 1.0 is perfect.
#: The measured in-match distribution is strongly bimodal -- real matches sit
#: at 0.75-0.95 and the animation frames well below 0.35 -- so this sits in
#: the empty band rather than on either mode.
MIN_SCORE = 0.45

#: A SELECTED CARD IS LIFTED OUT OF ITS SLOT, AND A RIGID TEMPLATE MISSES IT.
#:
#: Tapping a card raises it and draws a highlight bar under it. Measured on
#: `assets/live/match_practice_01`, the art translates up by a sharp, stable
#: **11 px** in a 73 px slot -- 15% of the slot -- and correlation against the
#: unshifted template collapses from ~0.99 to 0.26-0.39, i.e. below MIN_SCORE,
#: while the card stays perfectly legible to a human. All four residual errors
#: of the first version of this module were exactly this, and every one of
#: them recovers to 0.90-0.99 at dy = -11.
#:
#: That matters more than four crops suggests: a card is selected precisely
#: when it is ABOUT TO BE PLAYED, so these are the frames whose identity the
#: placement pipeline most depends on.
#:
#: The window is searched only when the direct match is poor, so a clean slot
#: -- the overwhelming majority -- pays one correlation, exactly as before.
LIFT_SEARCH = (-14, 3)
SEARCH_TRIGGER = 0.75

#: Mean standard deviation across the colour axis, below which a slot is being
#: rendered in TRUE greyscale. CRBAB uses the same statistic and the same 5 to
#: answer "is this card affordable"; it is one physical question -- is there
#: any colour here -- so it is computed once per slot and used for both.
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
    """Templates keyed by CRBAB card NAME, from a built pool directory.

    The pool's `icons.json` keys by SIMULATOR id, because that is what
    `readers/hand.py` and the bridge speak. The live loop speaks CRBAB card
    objects. `live/unit_to_card.hand_card_id_for` is the one mapping between
    them and is used here rather than a second table.
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
    """Drop-in for `CardDetector`: `run(image) -> (cards, ready)`.

    `cards[0]` is the small next-card preview and `cards[1:5]` the hand, which
    is the layout every caller in this repo already slices by.
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
        # (n_cards, 48*40), unit-normalised so a dot product IS the
        # correlation and the whole match is one matmul.
        stack = []
        for name in self._names:
            vec = normalise_icon(pool[name]).ravel()
            stack.append(vec / (np.linalg.norm(vec) + 1e-6))
        self._templates = np.asarray(stack, dtype=np.float32)

    def _vec(self, crop_bgr: np.ndarray) -> np.ndarray:
        vec = normalise_icon(crop_bgr).ravel()
        return vec / (np.linalg.norm(vec) + 1e-6)

    def _best_over_lift(self, tall: np.ndarray, base: int, height: int):
        """Best (scores, dy) over the vertical lift window.

        `tall` is the slot crop grown upward by -LIFT_SEARCH[0] rows and
        downward by LIFT_SEARCH[1]; `base` is the row at which the unshifted
        slot begins inside it.
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

        One PIL crop per slot; every lift offset is then a numpy slice of it.
        `present` is the badge-or-greyscale rule the module docstring derives.
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
        """One fixed crop, no lift search. Kept for diagnostics and tests."""
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
        """A bijection over the slots that actually hold a card.

        Empty slots are excluded from the assignment rather than left in it to
        be matched against a `blank` template, so the bijection is over the
        slots that really hold a card and cannot be spent on one that does
        not.

        This is measurably worth having, but only once dimmed cards are read
        at all. With them excluded the two assignment modes scored
        identically; with them included, `hungarian` cuts impossible cycle
        transitions from 16.7% to 5.3% and duplicates from 0.6% to 0.4%, at
        equal labelled accuracy -- the constraint has errors to correct
        exactly where the matcher is least certain.
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
        """Which hand slots are affordable.

        CRBAB tests `mean(std over the colour axis) > 5`, i.e. "is this crop
        coloured at all", which works precisely BECAUSE the game renders an
        unaffordable card in true greyscale. Kept verbatim: it is a different
        question from identity and it is measurably right.
        """
        return [i for i, w in enumerate(hand_windows)
                if w.colour > GREY_STD_THRESHOLD]
