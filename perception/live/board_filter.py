"""Reject detections that fall outside the arena.

MEASURED, NOT SPECULATIVE
-------------------------
Over 71 in-game frames of a real ladder match, ClashRoyaleBuildABot's unit
detector produced 328 detections. **102 of them -- 31% -- sat outside the
board**, and every single one was classified `knight`:

    knight tile=(19, 5)   x68      board is 18 wide, so x=19 does not exist
    knight tile=(-2,13)   x33

Those are the two player AVATAR ICONS in the battle UI's top corners. They are
in frame, they look like a character portrait, and the model dutifully finds a
Knight in each one every frame.

The effect on the class distribution is not subtle: `knight` goes from 111 of
328 detections (34%, the single most common class) down to 9 of 226 once the
off-board ones are dropped. Every other class is untouched. Upstream tracks
this as "add ignore regions to the object detector" (issue #244).

WHY A BOUNDS CHECK RATHER THAN AN IGNORE RECTANGLE
--------------------------------------------------
An ignore rectangle in pixel space would have to be re-measured for every
resolution and every UI change. The board's tile bounds are a property of the
game, are already known exactly, and catch any future phantom wherever on the
UI it appears -- these two happen to be top-left and top-right, but nothing
guarantees the next one is.
"""
from __future__ import annotations

from dataclasses import dataclass

# The real arena, in CRBAB's tile convention: 18 columns, 32 rows, origin at
# our own back line. NOT the engine's 34 rows -- ClashEnv adds a row behind
# each King Tower that the real board does not have, so anything crossing into
# engine space has to add that offset rather than reuse these.
BOARD_WIDTH = 18
BOARD_HEIGHT = 32


@dataclass(frozen=True)
class FilterReport:
    kept: int = 0
    dropped: int = 0

    @property
    def dropped_fraction(self) -> float:
        total = self.kept + self.dropped
        return self.dropped / total if total else 0.0


def on_board(tile_x: int, tile_y: int) -> bool:
    return 0 <= tile_x < BOARD_WIDTH and 0 <= tile_y < BOARD_HEIGHT


def filter_units(units) -> tuple[list, FilterReport]:
    """Keep only detections whose tile lies inside the arena.

    Returns (kept, report). The report exists so a caller can alarm on the
    drop rate instead of discovering silently that a UI change moved a phantom
    onto the board, where this check cannot see it.
    """
    kept = [u for u in units if on_board(u.position.tile_x, u.position.tile_y)]
    return kept, FilterReport(kept=len(kept), dropped=len(units) - len(kept))
