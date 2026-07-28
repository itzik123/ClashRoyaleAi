"""Detecting the opponent's placements. The expensive stage.

STATUS: the two sides are deliberately asymmetric, and only one of them is
finished. That is not an oversight -- it is the cheapest correct ordering.

OUR SIDE IS DONE AND NEEDS NO MODEL
-----------------------------------
A card leaving one of our hand slots IS a placement, and the card that
replaces it is the next queue entry. readers/hand.infer_play_from_hand_change
extracts both, exactly, from two consecutive frames. So our card identity and
our timing are free and perfectly labelled -- no tap logger, no annotation
pass, no model. Only the TILE has to be recovered from the board.

That is the whole basis of the semi-supervised plan: our own plays generate a
correctly labelled dataset just by playing, and the opponent's side is the
same visual problem with the board mirrored.

THE OPPONENT'S SIDE NEEDS RECORDED DATA, AND THERE IS NONE
-----------------------------------------------------------
Detecting an opponent placement means finding the moment and screen position
where a unit appears, then classifying it. Both halves need frames from the
real game to build or train anything at all, and no recording exists yet.

So `OpponentPlacementDetector.detect` raises NotImplementedError rather than
shipping a plausible-looking heuristic. A frame-differencing stub would
"work" on any input and be wrong in ways that only show up as divergence
three stages downstream -- which is exactly the failure this whole design is
built to avoid.

What IS specified here is the interface, the sampling rate, and the
suppression logic, all of which are testable and independent of the model.

WHY THE DETECTION RATE IS A PARAMETER
-------------------------------------
A match produces roughly 20 placements a minute. Running a detector at 30fps
to catch 20 events is three orders of magnitude of waste. The rate is
configurable and decimation happens once, in FrameSource.sample_every, on
presentation time rather than frame index.

The floor is set by tile accuracy, not by event count: a Hog Rider covers
about a tile every 0.9 s, so at 4 fps a unit has already moved ~0.3 tiles
before its first observation -- inside the 1.5-tile budget, but not by much
if the unit is fast. 6 fps is the default for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts import PlacementEvent

# Detector rate, in frames per second. See the docstring's last section.
DEFAULT_DETECT_FPS = 6.0

# Two detections of the same card within this many ticks and this many tiles
# are treated as one placement. Multi-unit cards spawn several units at once
# and slow-decimated frames can see the same spawn twice; both must collapse
# to a single event or the opponent-elixir model double-charges and the
# missed-placement alarm inverts into a false-positive machine.
SUPPRESS_WITHIN_TICKS = 12
SUPPRESS_WITHIN_TILES = 3.5


class DetectorNotTrainedError(NotImplementedError):
    """No detector exists yet. See this module's docstring."""


@dataclass
class Detection:
    """A raw detection, before suppression and mapping."""

    tick: int
    wall_time_ms: float
    card_real_name: str
    tile_x: int
    tile_y: int
    confidence: float
    is_evolution: bool = False


@dataclass
class PlacementSuppressor:
    """Collapses repeat detections of one placement into a single event.

    Stateful and order-dependent: feed detections in tick order.
    """

    within_ticks: int = SUPPRESS_WITHIN_TICKS
    within_tiles: float = SUPPRESS_WITHIN_TILES
    _recent: list[Detection] = field(default_factory=list)

    def accept(self, detection: Detection) -> bool:
        self._recent = [
            d for d in self._recent
            if detection.tick - d.tick <= self.within_ticks
        ]
        for previous in self._recent:
            if previous.card_real_name != detection.card_real_name:
                continue
            distance = np.hypot(
                previous.tile_x - detection.tile_x,
                previous.tile_y - detection.tile_y,
            )
            if distance <= self.within_tiles:
                return False
        self._recent.append(detection)
        return True


class OpponentPlacementDetector:
    """Finds opponent placements on the board.

    Not implemented -- see the module docstring. The constructor accepts its
    real dependencies so the wiring around it can be built and tested now,
    and so that the day a model exists it drops in without any consumer
    changing.
    """

    def __init__(self, homography, geometry, model_path=None, detect_fps: float = DEFAULT_DETECT_FPS):
        self.homography = homography
        self.geometry = geometry
        self.model_path = model_path
        self.detect_fps = detect_fps
        self.suppressor = PlacementSuppressor()

    def detect(self, frame: np.ndarray, tick: int, wall_time_ms: float) -> list[Detection]:
        raise DetectorNotTrainedError(
            "no opponent placement detector exists yet.\n"
            "\n"
            "It needs recorded matches to build: both the appearance model and "
            "the spawn-moment detector are learned from frames of the real "
            "game, and none have been captured.\n"
            "\n"
            "Our OWN placements do not need this at all -- see "
            "readers/hand.infer_play_from_hand_change, which extracts card and "
            "timing exactly from consecutive hand readings. Start there; it is "
            "what produces the labelled data this detector will be trained on.\n"
            "\n"
            "See perception/README.md, 'What to record'."
        )

    def to_events(self, detections: list[Detection]) -> list[PlacementEvent]:
        """Map detections to events, applying suppression and the card map.

        Mirrors y into ClashEnv convention here, once, so no consumer has to
        remember to. PlacementEvent.tile_y is defined as already mirrored for
        team 1 -- see contracts.py.
        """
        import mapping  # imported here so this module works without the map

        events: list[PlacementEvent] = []
        for detection in detections:
            if not self.suppressor.accept(detection):
                continue
            entry = mapping.resolve(detection.card_real_name)  # raises on unknown
            events.append(PlacementEvent(
                tick=detection.tick,
                wall_time_ms=detection.wall_time_ms,
                card_sim_id=entry.sim_id_for(detection.is_evolution),
                card_real_name=entry.real_name,
                team=1,
                tile_x=detection.tile_x,
                tile_y=(self.geometry.height - 1) - detection.tile_y,
                confidence=detection.confidence,
                is_evolution=detection.is_evolution,
            ))
        return events
