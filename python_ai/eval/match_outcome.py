"""How a finished match is scored, matching the engine.

`include/core/TimeoutRules.h` decides a timed-out match in order:

  1. Fewer surviving towers loses.
  2. On equal counts, the side whose weakest surviving tower has lower HP loses.
  3. Only an exact tie on both is a draw.

Evaluation scripts use this rather than comparing tower counts, which calls
every equal-count finish a draw.

The engine's own verdict (`resolve_timeout_outcome`) is used when bound. The
fallback mirror reads the six tower HPs from the observation's extra scalars
(`tail = EXTRA_SCALARS_START`):

    tail+0        elapsed-time fraction
    tail+1, +2    both sides' cumulative elixir spend
    tail+3..+5    the observing team's king, left, right
    tail+6..+8    the opponent's king, left, right

Each is hp / MAX_BUILDING_HP and reads exactly 0.0 once destroyed, so filtering
zeros gives "surviving towers only". Normalised floats preserve order and
equality: one HP point (2.5e-4) is far above float32 resolution.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as _E  # noqa: E402

# Prefer the engine's verdict (ClashEnv::resolveTimeoutOutcome). The fallback
# serves a .pyd built before the binding; once every environment has a fresh
# build, drop it and tests/test_match_outcome_is_the_only_scorer.py.
_USE_BINDING = hasattr(_E.ClashRoyaleEnv, "resolve_timeout_outcome")


def score_from_towers(env, team=0):
    """1.0 win / 0.5 draw / 0.0 loss for `team`, by TimeoutRules.

    Safe on any finished match: a fallen King already makes the tower counts
    differ.
    """
    if _USE_BINDING:
        # loserTeam: -1 draw, 0 team 0 lost, 1 team 1 lost.
        loser = env.resolve_timeout_outcome()
        if loser == -1:
            return 0.5
        return 0.0 if loser == team else 1.0

    mine = env.get_towers_alive(team)
    theirs = env.get_towers_alive(1 - team)

    # 1. Fewer surviving towers loses.
    if mine != theirs:
        return 1.0 if mine > theirs else 0.0

    # 2. Equal counts: the lower weakest tower loses. Both halves come from
    #    `team`'s own observation. An empty slice (nothing standing) cannot
    #    occur here with equal counts unless both are empty.
    obs = np.asarray(env.get_observation_for_team(team), dtype=np.float32)
    tail = _E.ClashRoyaleEnv.EXTRA_SCALARS_START
    my_alive = obs[tail + 3:tail + 6]
    their_alive = obs[tail + 6:tail + 9]
    my_weakest = my_alive[my_alive > 0.0]
    their_weakest = their_alive[their_alive > 0.0]

    # Both empty: neither side has a tower standing, an exact tie.
    if my_weakest.size and their_weakest.size:
        a, b = float(my_weakest.min()), float(their_weakest.min())
        if a != b:
            return 1.0 if a > b else 0.0

    # 3. A genuine draw.
    return 0.5


def terminal_value(env, team=0):
    """The same verdict as +1 / 0 / -1, for use as a search leaf value."""
    return score_from_towers(env, team) * 2.0 - 1.0
