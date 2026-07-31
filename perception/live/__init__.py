"""Live-sensor layer built on top of the vendored ClashRoyaleBuildABot.

Deliberately additive: nothing in `clashroyalebuildabot/` is edited, so it
stays updatable from upstream. What lives here is what CRBAB does not provide
and the engine's observation requires.

  king_hp       King Tower HP -- CRBAB reads only the four Princess towers,
                while the observation needs six (extra scalars 3-8).
  board_filter  rejects detections outside the arena; measured at 31% of all
                in-game detections on a real ladder match, all of them the
                two player avatar icons read as Knights.
"""
