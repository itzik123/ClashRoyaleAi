"""The one and only place real seconds are converted to simulator ticks.

Every other module imports from here. There is no second copy of these
numbers anywhere in `perception/`, and tests/test_timebase.py asserts the
elixir constant against the live engine so a change on the C++ side surfaces
as a test failure rather than as silent drift.

--------------------------------------------------------------------------
HOW TICKS_PER_SECOND WAS ESTABLISHED
--------------------------------------------------------------------------
The engine never states its own tick duration. It has to be recovered from a
quantity that is expressed in both units, and there is exactly one such
quantity in the codebase: CardStats::attackCooldown, which is stored in ticks
while the real game publishes the same figure ("hit speed") in seconds.

Across every card sampled from CardRegistry.h, attackCooldown is the real hit
speed times ten, with no exceptions:

    Archers        9 ticks   <->  0.9 s
    Musketeer     10 ticks   <->  1.0 s
    Knight        12 ticks   <->  1.2 s
    Valkyrie      15 ticks   <->  1.5 s
    Hog Rider     16 ticks   <->  1.6 s
    King Tower    10 ticks   <->  1.0 s   (GameManager::addTower)

132 independent data rows agreeing on one ratio is a much stronger source
than any single constant, so this -- not the elixir rate -- is what defines
the tick.

--------------------------------------------------------------------------
THE ELIXIR DISCREPANCY, AND WHY IT IS NOT PAPERED OVER
--------------------------------------------------------------------------
GameManager::step() adds ELIXIR_REGEN_RATE = 0.035 per tick (verified
empirically: 5.000 -> 5.035 -> 5.070 -> 5.105 on consecutive ticks). At 10
ticks per second that is 0.35 elixir/s, i.e. 2.857 s per elixir -- against
the real game's 2.8 s. The simulator regenerates elixir about 2% too slowly.

This is small per tick and NOT small over a match. Deriving the opponent's
elixir at the simulator's rate would put the estimate roughly 1.3 elixir low
by the three-minute mark. That matters specifically because the cheapest,
highest-value alarm in the whole pipeline is "derived opponent elixir went
negative => a placement was definitely missed" -- and a systematic downward
bias turns that certain signal into a stream of false positives.

So the two rates are kept separate and named for what they are:

  * REAL_SECONDS_PER_ELIXIR_1X drives the opponent elixir model, because it
    models the real opponent.
  * SIM_ELIXIR_REGEN_PER_TICK describes the engine, and is used only when
    reasoning about what the engine will do.

Neither is "corrected" to match the other. That would be a gameplay change to
the simulator, which this module has no business making.
"""

from __future__ import annotations

# Ticks per real second. See the derivation above.
TICKS_PER_SECOND = 10

# GameManager.h: `const float ELIXIR_REGEN_RATE = 0.035f`, applied once per
# tick in step(). Asserted against the live engine in tests/test_timebase.py.
SIM_ELIXIR_REGEN_PER_TICK = 0.035

# Real-game single-elixir rate.
REAL_SECONDS_PER_ELIXIR_1X = 2.8

# PlayerState::initializeDeck sets both players to 5.0 at match start.
STARTING_ELIXIR = 5.0

# PlayerState caps elixir here (GameManager::step()'s std::min).
MAX_ELIXIR = 10.0

# Ratio by which the simulator's elixir economy runs slow, >1.0 meaning the
# simulator is slower than the real game. Exported rather than left implicit
# so anything comparing a simulated economy against an observed one can state
# the correction it is applying instead of hiding a magic 1.02.
SIM_ELIXIR_RATE_ERROR = (
    REAL_SECONDS_PER_ELIXIR_1X * SIM_ELIXIR_REGEN_PER_TICK * TICKS_PER_SECOND
)  # == 0.98: the simulator produces 98% of the real elixir per unit time.


def seconds_to_ticks(seconds: float) -> int:
    """Real seconds -> simulator ticks, rounded to nearest.

    Rounded, not truncated: truncation biases every conversion downward, and
    over the ~20 placements a match produces that accumulates into a
    consistent "everything happened slightly earlier than it did" error in
    the estimator.
    """
    return int(round(seconds * TICKS_PER_SECOND))


def ticks_to_seconds(ticks: int) -> float:
    """Simulator ticks -> real seconds."""
    return ticks / TICKS_PER_SECOND


def frame_index_to_ticks(frame_index: int, fps: float) -> int:
    """Video frame number -> simulator ticks.

    Only valid for constant-frame-rate sources. With VFR the frame index is
    not proportional to elapsed time at all and this silently produces
    garbage, which is why capture/video.py refuses to report an fps it could
    not confirm as constant.
    """
    if fps <= 0:
        raise ValueError(f"frame_index_to_ticks: non-positive fps {fps!r}")
    return seconds_to_ticks(frame_index / fps)


def elixir_regenerated(seconds: float, multiplier: float = 1.0) -> float:
    """Real elixir accrued over `seconds` at `multiplier` x rate.

    multiplier is 1/2/3 for single/double/triple elixir. The simulator has no
    such concept (see contracts.Phase); this function models the REAL game,
    and is used by track/opp_elixir.py, which also models the real game.
    """
    if seconds < 0:
        raise ValueError(f"elixir_regenerated: negative duration {seconds!r}")
    return seconds * multiplier / REAL_SECONDS_PER_ELIXIR_1X
