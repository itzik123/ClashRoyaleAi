"""Reject detections that fall outside the arena.

CRBAB's unit detector finds a Knight in each of the two player avatar icons in
the battle UI's top corners, every frame, at tiles that do not exist (x=19,
x=-2). Upstream tracks this as issue #244.

A bounds check rather than a pixel ignore rectangle: tile bounds are a property
of the game, need no re-measuring per resolution or UI change, and catch a
future phantom wherever it appears.
"""
from __future__ import annotations

from dataclasses import dataclass

# The real arena in CRBAB's tile convention: 18 columns, 32 rows, origin at our
# back line. Not the engine's 34 rows (one extra behind each King).
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
    """Keep only detections whose tile lies inside the arena. Returns (kept,
    report), so a caller can alarm on the drop rate rather than miss a UI
    change that moved a phantom onto the board.
    """
    kept = [u for u in units if on_board(u.position.tile_x, u.position.tile_y)]
    return kept, FilterReport(kept=len(kept), dropped=len(units) - len(kept))
