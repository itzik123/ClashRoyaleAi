"""Our four hand slots and the next-card slot, by template matching.

The slots are at fixed positions and show one of eight known icons -- our own
deck, which we are told in advance. So this is a 1-of-8 classification against
templates cropped from the recording itself, not open-set recognition. That
is the easiest problem anywhere in this pipeline, and it is worth doing well
because it produces two things nothing else can:

  1. `incoming_card` for bridge/sim_driver.py. Knowing which card slid into a
     vacated slot is what collapses the simulator's candidate pool onto our
     real cycle -- see that module. Without it the pool can only be narrowed
     by play failures, which is far slower and sometimes not enough.

  2. A free cross-check on the elixir bar. An unaffordable slot is rendered
     dimmed, so the set of dimmed slots brackets the elixir value from both
     sides at no extra cost -- see readers/elixir.cross_check. Two
     independent readings of the same quantity, and neither needs a label.

DIMMING IS MEASURED RELATIVE TO THE SLOT'S OWN HISTORY
------------------------------------------------------
The absolute brightness of a card icon depends on the card -- a Skeleton Army
icon is darker than a Fireball icon before any dimming is applied. So an
absolute threshold misclassifies whole cards as permanently unaffordable.
What is stable is the RATIO between a slot's current brightness and the
brightest that same card has been observed at, which is what `is_dimmed`
uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Below this fraction of a card's own observed maximum brightness, the slot
# is treated as dimmed. The game's dimming is substantial (roughly half
# value); 0.75 sits well clear of compression noise and shimmer.
DIM_RATIO_THRESHOLD = 0.75

# Correlation of contrast-normalised crops: 1.0 is a perfect match, 0.0 is
# unrelated. Calibrated against measured scores on this batch.
MIN_ICON_SCORE = 0.30
MIN_ICON_MARGIN = 0.05

# Icon crops are normalised to this (h, w) before matching. Must match the
# shape tools/build_icon_templates.py clusters at, or the two representations
# are not comparable.
ICON_SHAPE = (48, 40)


def _normalise_icon(image: np.ndarray) -> np.ndarray:
    """Grey, resized, contrast-normalised. See HandReader._read_slot."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (ICON_SHAPE[1], ICON_SHAPE[0]),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


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

    `icons` maps a simulator card id to a template cropped from THIS
    recording at THIS resolution. Templates from elsewhere would need
    rescaling, and a rescaled template matches systematically worse in a way
    that looks like a hard frame rather than a setup error.
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

        # Matched on contrast-normalised greyscale, not raw BGR.
        #
        # The same representation the icon templates were CLUSTERED with, and
        # that is the point: clustering separated all eight cards perfectly,
        # so it is demonstrably sufficient, while raw-BGR matching on the same
        # icons flickered badly -- 0.790 mean confidence, 14% of hands
        # containing a duplicate or an off-deck card, and phantom "plays"
        # 0.8s apart in one slot, which the engine's own 20-tick slot cooldown
        # makes impossible.
        #
        # The reason is the affordability dimming: an unaffordable slot is
        # rendered darker AND lower contrast, which moves raw pixel values a
        # long way. Subtracting the mean and dividing by the standard
        # deviation removes exactly that, which is why the clusters were clean.
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

    WITHOUT THIS THE PLAY STREAM IS MOSTLY FICTION. Playing a card is an
    animation: the icon lifts out of the slot, the next card slides in, and
    for a few frames the slot holds a blend of the two. Every one of those
    intermediate frames matches SOME template, so a naive
    before/after comparison reports a chain of plays where there was one.

    Measured on a 326-second recording at 6fps: 69 "plays", of which 32 were
    closer together than the engine's own 20-tick (2s) hand-slot cooldown
    makes possible -- three changes in one slot inside 1.5s, in one case.

    A hand state has to be observed `hold` times in a row before it is
    believed. Transients never survive that; a real play does, because the
    new hand persists until the next play.
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

    This is what makes our side of the dataset self-labelling: a card
    vacating a slot IS a placement, and the card that replaces it is the
    queue front. No tap logger, no manual annotation, no extra tooling -- the
    label comes from the same frames the detector is being trained on.

    What it does NOT give is WHERE the card was placed, which still has to
    come from the board. So this labels the card and the time for free, and
    leaves only the tile to be recovered.

    Returns None when more than one slot changed: two simultaneous changes
    mean frames were dropped between the readings, and guessing which card
    went with which slot would fabricate a label.
    """
    if len(before) != len(after):
        return None
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    if len(changed) != 1:
        return None
    index = changed[0]
    return index, before[index], after[index]
