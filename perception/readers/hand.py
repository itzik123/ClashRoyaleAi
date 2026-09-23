"""Our four hand slots and the next-card slot, by template matching.

The slots are fixed and show one of our own eight known icons, so this is
1-of-8 classification against templates cropped from the recording itself. It
produces two things nothing else can:

  1. `incoming_card` for bridge/sim_driver.py: knowing which card slid into a
     vacated slot collapses the simulator's candidate pool onto our real
     cycle far faster than play failures alone.
  2. A free cross-check on the elixir bar: unaffordable slots are dimmed, so
     the set of dimmed slots brackets the elixir value from both sides (see
     readers/elixir.cross_check).

Dimming is measured relative to the slot's own history: icons differ in
absolute brightness, so a fixed threshold would mark dark cards permanently
unaffordable. `is_dimmed` uses the ratio to the brightest that card has been
seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Below this fraction of a card's own observed maximum brightness, the slot is
# dimmed. The game dims by roughly half; 0.75 is clear of compression noise.
DIM_RATIO_THRESHOLD = 0.75

# Correlation of contrast-normalised crops (1.0 perfect, 0.0 unrelated),
# calibrated on this batch.
MIN_ICON_SCORE = 0.30
MIN_ICON_MARGIN = 0.05

# Icon crops are normalised to this (h, w) before matching; must equal what
# tools/build_icon_templates.py clusters at.
ICON_SHAPE = (48, 40)


def normalise_icon(image: np.ndarray) -> np.ndarray:
    """Grey, resized, contrast-normalised. See HandReader._read_slot.

    Public because three places must agree on it: this reader,
    `tools/build_icon_templates.py` (which clusters in it) and
    `live/deck_hand.py` (which matches in it).
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (ICON_SHAPE[1], ICON_SHAPE[0]),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


#: Kept for existing internal callers.
_normalise_icon = normalise_icon


def has_cost_badge(crop: np.ndarray) -> bool:
    """True if this slot actually holds a card. `crop` is BGR.

    Every card icon carries a magenta elixir-cost badge low-centre; nothing
    else in the tray does (not the blue card-back shown while the next card
    slides in, not the empty between-match tray). A better "is this a card"
    test than any brightness or variance heuristic.
    """
    h, w = crop.shape[:2]
    badge = crop[int(h * 0.62):, int(w * 0.2):int(w * 0.8)]
    if badge.size == 0:
        return False
    hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
    magenta = cv2.inRange(hsv, np.array([135, 90, 90]), np.array([175, 255, 255]))
    return float(magenta.mean()) / 255.0 > 0.04


class HandCalibrationMissing(NotImplementedError):
    """Slot ROIs or card icon templates are absent."""


@dataclass(frozen=True)
class SlotReading:
    card_sim_id: int
    confidence: float
    dimmed: bool
    brightness: float


@dataclass
class HandReading:
    slots: tuple[SlotReading, ...]
    next_card: SlotReading | None

    @property
    def hand(self) -> tuple[int, ...]:
        return tuple(s.card_sim_id for s in self.slots)

    @property
    def confidence(self) -> float:
        return min((s.confidence for s in self.slots), default=0.0)


@dataclass
class HandReader:
    """Reads the hand from calibrated slot ROIs and per-card icon templates.

    `icons` maps a simulator card id to a template cropped from this recording
    at this resolution. A rescaled template matches systematically worse in a
    way that looks like a hard frame rather than a setup error.
    """

    slot_rois: list[tuple[int, int, int, int]]
    next_roi: tuple[int, int, int, int] | None
    icons: dict[int, np.ndarray]

    _peak_brightness: dict[int, float] = field(default_factory=dict)
    _normalised: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slot_rois or len(self.slot_rois) != 4:
            raise HandCalibrationMissing(
                f"expected 4 hand slot ROIs, got {len(self.slot_rois or [])}. "
                "Run perception/tools/calibrate.py on a frame from this "
                "recording."
            )
        if not self.icons:
            raise HandCalibrationMissing(
                "no card icon templates. Crop them from this recording with "
                "perception/tools/build_icon_templates.py -- templates taken "
                "at another resolution match systematically worse and the "
                "failure looks like a hard frame, not a setup error."
            )

    def read(self, frame: np.ndarray) -> HandReading:
        slots = tuple(self._read_slot(frame, roi) for roi in self.slot_rois)
        next_card = self._read_slot(frame, self.next_roi) if self.next_roi else None
        return HandReading(slots=slots, next_card=next_card)

    def _read_slot(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> SlotReading:
        x, y, w, h = roi
        patch = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        # Matched on contrast-normalised greyscale, the representation the
        # templates were clustered in, not raw BGR: an unaffordable slot is
        # darker and lower-contrast, which moves raw pixels a long way, and
        # mean/std normalisation removes exactly that.
        probe = _normalise_icon(patch)
        scores: dict[int, float] = {}
        for card_id, template in self.icons.items():
            key = id(template)
            cached = self._normalised.get(key)
            if cached is None:
                cached = _normalise_icon(template)
                self._normalised[key] = cached
            scores[card_id] = float((probe * cached).mean())

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        margin = best_score - (ranked[1][1] if len(ranked) > 1 else 0.0)

        peak = max(self._peak_brightness.get(best, 0.0), brightness)
        self._peak_brightness[best] = peak
        dimmed = brightness < peak * DIM_RATIO_THRESHOLD

        if best_score < MIN_ICON_SCORE or margin < MIN_ICON_MARGIN:
            confidence = max(0.0, min(1.0, margin / MIN_ICON_MARGIN)) * 0.5
        else:
            confidence = min(1.0, 0.5 + margin)

        return SlotReading(
            card_sim_id=best,
            confidence=round(confidence, 4),
            dimmed=dimmed,
            brightness=round(brightness, 2),
        )


@dataclass
class HandStabiliser:
    """Debounces hand readings, and emits a play only on a settled change.

    Playing a card is an animation: the icon lifts out, the next slides in, and
    for a few frames the slot holds a blend that matches some template, so a
    naive before/after comparison reports a chain of plays where there was one.
    A hand state must be seen `hold` times in a row before it is believed;
    transients never survive that, and a real play persists until the next.
    """

    hold: int = 2
    deck: frozenset[int] | None = None
    """If given, readings containing an off-deck card or a duplicate are
    discarded outright. Our own deck is known in advance, so a hand that
    cannot exist is a misread, not information."""

    stable: tuple[int, ...] | None = None
    _pending: tuple[int, ...] | None = None
    _count: int = 0
    rejected: int = 0

    def push(self, hand: tuple[int, ...]) -> tuple[int, int, int] | None:
        """Feed one reading. Returns a play when the hand settles on a change."""
        if len(set(hand)) != len(hand):
            self.rejected += 1
            return None
        if self.deck is not None and not set(hand) <= self.deck:
            self.rejected += 1
            return None

        if hand != self._pending:
            self._pending, self._count = hand, 1
            return None
        self._count += 1
        if self._count < self.hold or hand == self.stable:
            return None

        previous, self.stable = self.stable, hand
        if previous is None:
            return None
        return infer_play_from_hand_change(previous, hand)


def infer_play_from_hand_change(
    before: tuple[int, ...], after: tuple[int, ...]
) -> tuple[int, int, int] | None:
    """Detect our own play from consecutive hand readings.

    Returns (slot_index, played_card, incoming_card), or None.

    This makes our side of the dataset self-labelling: a card vacating a slot
    is a placement, and its replacement is the queue front. It labels the card
    and the time; the tile still has to come from the board.

    None when more than one slot changed: frames were dropped between the
    readings, and pairing cards with slots would fabricate a label.
    """
    if len(before) != len(after):
        return None
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    if len(changed) != 1:
        return None
    index = changed[0]
    return index, before[index], after[index]
