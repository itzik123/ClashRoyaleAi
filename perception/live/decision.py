"""What a live policy returns, kept free of the capture stack.

`mvp_loop.py` imports the detector, the window capture and PIL at module scope.
A policy that only needs to say "play slot 2 at (3, 14)" should not pull all of
that in, so the type lives here and `mvp_loop` re-exports it.
"""
from __future__ import annotations

from dataclasses import dataclass

# A fixed defensive tile in our own half, in engine coordinates. A no-op still
# carries a tile, since the loop prints one.
DEFAULT_TILE = (9, 8)


@dataclass
class Decision:
    slot: int | None
    """Hand slot to play, or None to hold."""

    tile: tuple[int, int]
    """Engine tile, (x, y), the frame `engine_tile_centre` expects."""

    why: str
