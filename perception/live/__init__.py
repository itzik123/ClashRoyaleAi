"""The live sensor, built on the vendored ClashRoyaleBuildABot.

Additive: no vendored file is modified, so what is kept stays diffable against
upstream. CRBAB is used purely as a sensor (`detectors/`, `namespaces/`,
`constants.py`, `models/`, `images/`); upstream's own agent was removed. This
package holds what CRBAB does not provide and the engine's observation
requires.

  adapter          CRBAB `State` -> `contracts.GameState`. Start here.
  unit_to_card     detector unit name -> engine card id (CRBAB names units,
                   the engine names cards).
  unit_hp          per-unit HP and the badge's team reading.
  king_hp          King Tower HP; CRBAB reads only the Princess towers.
  board_filter     rejects detections outside the arena.
  deck_hand        hand identity against templates of our own deck.
  hand_tracker     the hand, deduced from the card cycle.
  elixir_ledger    our cumulative elixir spend.
  match_state      the debounced "a battle is running" gate.
  pipeline         perception on its own thread, decisions at a fixed rate.
  action_gate      at most one tap per perceived board, never on a stale one.
  actuator         tile -> taps on the real screen.
  placement_confirm  did a tapped card actually reach the board.
  mvp_loop         the end-to-end loop.
"""
