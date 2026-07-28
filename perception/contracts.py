"""Data contract between the perception pipeline and everything downstream.

Dataclasses only -- no logic, no imports beyond the standard library. Every
other module in `perception/` depends on this one; this one depends on
nothing. That is deliberate: the contract has to be readable and diffable on
its own, without pulling in numpy/opencv, so the bridge and the trackers can
be unit-tested with no capture stack installed at all.

Two fields here look redundant and are not. Both exist because of a concrete
gap in the simulator, and both would be very easy to drop by accident:

  * `confidence`, on essentially everything. The engine has no
    representation of uncertainty whatsoever -- the observation vector is a
    dense float grid with no mask channel, no NaN, no "unknown" sentinel (see
    ClashEnv::extractObservationForTeam). A tile is either occupied or it is
    zero, and zero means "empty", never "I could not see". So uncertainty
    cannot be handed to the engine and cannot be represented inside it: it
    has to live in this layer, or it does not exist anywhere.

  * `wall_time_ms`, on PlacementEvent. The simulator applies a play on the
    exact tick it is told to, with no actuation delay model -- real input has
    roughly 100ms of it (touch -> render -> capture -> decode). Without a
    real-clock stamp taken at capture time, that offset can never be measured
    after the fact, only guessed. Keeping the raw timestamp costs 8 bytes and
    is the only thing that makes the calibration possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    """Which elixir-rate regime the match is currently in.

    NOT CONSUMED BY THE SIMULATOR. GameManager::step() applies a single
    `const float ELIXIR_REGEN_RATE = 0.035f` to both players for the entire
    match; there is no multiplier, no overtime, and no phase concept anywhere
    in the engine. (`oppElixirMultiplier` is a training-curriculum knob for
    handicapping the built-in opponent -- it is not a game phase, and driving
    it from here would silently corrupt the opponent model.)

    This field is produced, carried, logged, and asserted on -- and then
    ignored by bridge/sim_driver.py, on purpose. It is here so that the
    perception layer is already correct on the day the engine grows phases,
    and so the gap is visible in the type rather than buried in a comment.
    See perception/README.md, "Known gaps".
    """

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    OVERTIME = 4


class EventSource(Enum):
    """How a PlacementEvent was produced.

    Matters because the two sides of the board are labelled by completely
    different mechanisms with completely different error profiles: our own
    plays are read off the hand (a card vacating a slot is a near-certain
    placement), while the opponent's are detected from board pixels. Anything
    that aggregates confidence across both -- the divergence metric above
    all -- needs to be able to tell them apart.
    """

    OWN_HAND = "own_hand"          # inferred from our own hand changing
    OPPONENT_VISION = "opp_vision"  # detected on the board
    REPLAY_GROUND_TRUTH = "replay"  # read from a simulator replay JSON (tests)


# Sentinel for PlacementEvent.card_sim_id: a placement that was detected and
# localised, but whose card has no counterpart in the simulator's registry.
# Deliberately not None -- the field stays a plain int so consumers can never
# get a TypeError instead of the explicit branch they should have written.
# See mapping/card_map.json and bridge/sim_driver.py's own handling.
UNKNOWN_CARD_SIM_ID = -1


@dataclass(frozen=True)
class PlacementEvent:
    """One card being put on the board, by either player.

    Frozen: events are the append-only spine of the whole pipeline. Once a
    placement has been emitted and fed to the simulator, mutating it would
    desynchronise the estimator from its own history with no way to detect
    that it happened.
    """

    tick: int
    """Simulator ticks since match start. NOT seconds -- see timebase.py."""

    wall_time_ms: float
    """Capture-time wall clock, milliseconds. For latency calibration only;
    never fed to the engine. See this module's docstring."""

    card_sim_id: int
    """CardRegistry id, after mapping. UNKNOWN_CARD_SIM_ID if the card is
    real but absent from the simulator's roster."""

    card_real_name: str
    """Canonical real-game card name. Always populated, even when
    card_sim_id is UNKNOWN_CARD_SIM_ID -- it is the only thing that makes an
    unmapped placement diagnosable."""

    team: int
    """0 = us, 1 = opponent. Same convention as the engine."""

    tile_x: int
    tile_y: int
    """Board tile, already in ClashEnv convention: origin at the corner, y
    increasing toward the opponent, and ALREADY MIRRORED for team 1 the way
    ClashEnv::extractObservationForTeam mirrors it
    (y = BOARD_HEIGHT - 1 - rawY). Consumers never re-mirror."""

    confidence: float
    """[0,1]. For OWN_HAND events this is near 1.0 by construction."""

    source: EventSource = EventSource.OPPONENT_VISION

    is_evolution: bool = False
    """The evolved variant was played. Kept separate from card_sim_id rather
    than folded into it because the registry gives Evolutions their own ids
    (123-163) while reusing the base card's NAME verbatim -- id 1 and id 128
    are both "Archers". Name alone can therefore never disambiguate, so the
    flag has to be carried explicitly. See mapping/card_map.json."""


@dataclass
class ClockState:
    """Match clock, as believed right now."""

    tick: int
    seconds_remaining: float
    phase: Phase
    last_resync_tick: int
    """Tick of the last successful on-screen verification. A large gap
    between this and `tick` means the clock is free-running and its drift is
    unbounded -- consumers should discount accordingly."""

    drift_ms: float
    """Measured (predicted - observed) at the last resync. Signed: positive
    means the free-running clock was ahead of the screen."""

    confidence: float = 1.0


@dataclass
class CycleState:
    """A player's hand and the known part of their cycle order."""

    hand: tuple[int, ...]
    """Up to 4 simulator card ids. Shorter only before the deck is known."""

    next_card: int
    """The card shown in the "next" slot, or UNKNOWN_CARD_SIM_ID."""

    queue: tuple[int, ...]
    """Known cycle order behind `next_card`, nearest first. Empty when
    unknown. For the opponent this fills in gradually and is only fully
    determined once every one of their 8 cards has been seen once."""

    confidence: float = 1.0


@dataclass
class PerceptionFrame:
    """Everything perception believes at one instant.

    This is the unit the bridge consumes. `new_events` is the only field the
    simulator is actually driven by; the rest is state that either verifies
    it or is carried for downstream consumers and diagnostics.
    """

    clock: ClockState
    my_elixir: float
    opp_elixir: float
    """DERIVED, never observed -- the opponent's elixir bar is not on screen.
    Computed from a known start value, a known regen rate, and the cost of
    every detected placement. See track/opp_elixir.py."""

    my_cycle: CycleState
    opp_cycle: CycleState
    opp_known_deck: frozenset[int]
    new_events: list[PlacementEvent]

    sim_divergence: float = 0.0
    """0.0 = the simulator's predicted tower HP matches the screen. Grows as
    they disagree. This is the pipeline's single quality number: a missed
    placement, a misclassified card, and a mislocalised tile all surface
    here, with no hand-labelled ground truth required. See
    readers/towers.py -- which is VALIDATION ONLY and never feeds the
    engine."""

    flags: tuple[str, ...] = ()
    """Machine-readable anomaly markers raised this frame, e.g.
    "opp_elixir_negative" (a certain sign that a placement was missed) or
    "unmapped_placement". Consumers gate on these rather than re-deriving
    the same conditions from the numeric fields."""


@dataclass(frozen=True)
class BoardGeometry:
    """Board dimensions and landmarks, in simulator tile coordinates.

    Populated from the live engine where possible rather than hardcoded --
    see geometry.py. Exists as a dataclass so calibration code can be handed
    a geometry without importing the engine at all (which is what lets the
    homography tests run with no .pyd present).
    """

    width: int
    height: int
    river_y_start: float
    river_y_end: float
    left_bridge: tuple[float, float]
    right_bridge: tuple[float, float]
    own_king: tuple[float, float]
    opp_king: tuple[float, float]
    own_princess_left: tuple[float, float]
    own_princess_right: tuple[float, float]
    opp_princess_left: tuple[float, float]
    opp_princess_right: tuple[float, float]


@dataclass
class CalibrationProfile:
    """Screen -> tile mapping plus the fixed ROIs for one capture resolution.

    One profile per (device, resolution). A resolution change invalidates
    every pixel constant in here, which is why the resolution is part of the
    identity rather than a field that can quietly disagree with the frames
    being fed in -- see calib/homography.py's own size assertion.
    """

    name: str
    frame_width: int
    frame_height: int
    homography: tuple[float, ...]
    """Row-major 3x3, screen pixels -> board tiles. Flattened to 9 floats so
    the profile stays plain JSON with no numpy dependency."""

    anchors_screen: dict[str, tuple[float, float]] = field(default_factory=dict)
    """The named landmark pixel coordinates the homography was solved from,
    kept so a profile can be re-solved or audited without re-clicking."""

    rois: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    """Named (x, y, w, h) rectangles in frame pixels: elixir bar, clock, the
    four hand slots, the next-card slot, the six tower HP bars."""

    reprojection_error_tiles: float = 0.0
    """Worst-case landmark reprojection error from the solve, in tiles. The
    stage-0 acceptance number; stored so it can be asserted on load instead
    of being a one-off console print at calibration time."""
