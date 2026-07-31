"""Live-sensor layer built on top of the vendored ClashRoyaleBuildABot.

Deliberately additive: nothing in `clashroyalebuildabot/` is edited, so it
stays updatable from upstream. What lives here is what CRBAB does not provide
and the engine's observation requires.

  adapter       CRBAB `State` -> `contracts.GameState`. The join everything
                else feeds into; start here.
  unit_to_card  detector unit name -> engine card id; CRBAB names units, the
                engine names cards, and only 70 of 97 matched by name.
  unit_hp       per-unit HP and the badge's team reading. Fitted against 60
                hand labels: precision 0.98, recall 0.56 on "is this unit
                damaged", up from 0.79 / 0.34.
  king_hp       King Tower HP -- CRBAB reads only the four Princess towers,
                while the observation needs six (extra scalars 3-8).
  board_filter  rejects detections outside the arena; measured at 31% of all
                in-game detections on a real ladder match, all of them the
                two player avatar icons read as Knights.

Not yet built, and the adapter is not live until they are: a frame source
(`capture/window.py`), the match clock, and cumulative elixir spend for both
sides -- the one accumulator in the observation, where a missed placement is
permanent rather than self-correcting.
"""
