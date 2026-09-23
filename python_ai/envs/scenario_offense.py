"""Offensive scenario injection: practice at the punish, not the defence.

    from python_ai.envs.scenario_offense import apply_offensive_scenario

Off by default (`OFFENSIVE_SCENARIO_PROB = 0.0`). Injection changes the state
distribution, not the payoff, so it only helps once playing the win condition
actually pays; then it helps credit assignment for rare punish windows. Turn it
on only after `prove_environment.py` shows the win condition paying. It never
touches the reward.

  punish_window  They just committed (bar near 1, units crossing one lane); we
                 hold 8 elixir and the win condition. The play: the other lane.
  counter_push   Our defence survived with units near the river and ~6 elixir
                 banked. The play: push behind them.

Contracts:
* `set_hand_for_team` returns False and changes nothing on an invalid hand;
  checked here, so a rejected setup is not counted as a scenario.
* `inject` bypasses hand, elixir and placement rules, hence absolute board
  coordinates, and only queues the spawn: one tick must be stepped (with
  `step_self_play`, so the heuristic gets no move) before it appears. Elixir is
  written after that tick.
"""
import os


import clash_royale_env as E
from python_ai import engine_constants as EC

CE = E.ClashRoyaleEnv

# Off by default; see the module docstring.
OFFENSIVE_SCENARIO_PROB = float(os.environ.get("CLASH_OFFENSIVE_SCENARIO_PROB", 0.0))

# A .pyd built before these bindings raises AttributeError mid-episode; detect
# it once here.
HAS_STATE_SETTERS = all(hasattr(CE, m)
                        for m in ("set_elixir_for_team", "set_hand_for_team"))

# Absolute board coordinates (inject bypasses frame conversion); team 0 attacks
# toward high y. Derived from ArenaLayout.
BRIDGE_XS = (EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X)
RIVER_Y = EC.BRIDGE_Y
OWN_SIDE_Y = 13.0        # our half, short of our Princess Towers
ENEMY_SIDE_Y = 20.0      # their half, just past the river


def _wincon_and_filler(deck):
    """(win condition id, three other deck cards): the hand a punish scenario
    needs.

    (None, None) for a deck without a win condition; the scenario is then
    skipped.
    """
    from python_ai.opponents.teacher import card_roles
    roles = card_roles(deck)
    wincon = next((c for c, r in roles.items() if r == "wincon"), None)
    if wincon is None:
        return None, None
    others = [c for c in deck if c != wincon][:3]
    if len(others) < 3:
        return None, None
    return wincon, others


def _settle(env):
    """One tick so queued `inject` spawns reach the board, via step_self_play so
    the C++ heuristic gets no free move.
    """
    env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 1)


def punish_window(env, rng, deck):
    """They just spent; we hold 8 and the win condition. Returns a name or None."""
    wincon, others = _wincon_and_filler(deck)
    if wincon is None:
        return None
    if not env.set_hand_for_team(0, [wincon] + others):
        # An unusable hand means this is not a punish scenario; do not count
        # it.
        return None

    # Their commitment in one lane, already crossing: two bodies, so one cheap
    # answer does not erase it.
    lane = int(rng.integers(2))
    x = BRIDGE_XS[lane]
    for cid, dy in ((wincon, 0.0), (others[0], 2.0)):
        env.inject(int(cid), float(x), float(RIVER_Y - dy), 1)
    _settle(env)
    env.set_elixir_for_team(0, 8.0)
    env.set_elixir_for_team(1, 1.0)
    return f"punish_window_lane{lane}"


def counter_push(env, rng, deck):
    """Our defence survived with units alive and elixir banked."""
    wincon, others = _wincon_and_filler(deck)
    if wincon is None:
        return None
    if not env.set_hand_for_team(0, [wincon] + others):
        return None

    lane = int(rng.integers(2))
    x = BRIDGE_XS[lane]
    # Survivors on our side of the river, able to escort a push.
    for cid, dy in ((others[0], 0.0), (others[1], 2.0)):
        env.inject(int(cid), float(x), float(OWN_SIDE_Y - dy), 0)
    _settle(env)
    env.set_elixir_for_team(0, 6.0)
    env.set_elixir_for_team(1, 3.0)
    return f"counter_push_lane{lane}"


SCENARIOS = (punish_window, counter_push)


def apply_offensive_scenario(env, rng, deck, prob=None):
    """With probability `prob`, rewrite the freshly reset state into a punish.

    Returns the scenario's name, or None if nothing was applied, including when
    the engine refused the setup.
    """
    p = OFFENSIVE_SCENARIO_PROB if prob is None else float(prob)
    if p <= 0.0 or float(rng.random()) >= p:
        return None
    if not HAS_STATE_SETTERS:
        raise RuntimeError(
            "offensive scenario injection needs set_elixir_for_team/"
            "set_hand_for_team, which this clash_royale_env.pyd does not "
            "export. The bindings exist in src/bindings.cpp -- the post-build "
            "copy into python_ai/ failed (MSB3073, a Python process held the "
            "DLL). Copy build_python/Release/clash_royale_env.cp311-win_amd64"
            ".pyd over python_ai/clash_royale_env.pyd with no Python running.")
    return SCENARIOS[int(rng.integers(len(SCENARIOS)))](env, rng, deck)
