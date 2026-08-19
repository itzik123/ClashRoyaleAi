"""How a finished match is scored, in one place, matching the engine exactly.

WHY THIS EXISTS
---------------
`include/core/TimeoutRules.h` decides a timed-out match in three ordered steps:

  1. Fewer surviving towers loses.
  2. On EQUAL counts, the side whose weakest surviving tower has lower HP loses.
  3. Only an exact tie on both is a genuine draw.

Its own header records why: before it existed every timed-out match scored 0.0,
which taught the agent that running the clock out was free.

Four separate evaluation scripts (`net_ab.py`, `net_h2h.py`, and `net_h2h_search
.py` twice) each hand-rolled step 1 alone and called everything else a draw:

    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)

A match ending 3-3 on towers but 1200 HP against 90 HP on the weakest is a clear
win by the engine's own rules, and all four reported it as a draw -- biasing the
very head-to-head win rates those scripts exist to produce. Consolidated here so
there is one implementation to keep correct instead of four.

WHERE THE TIE-BREAK DATA COMES FROM
-----------------------------------
No new binding is needed. `ClashEnv::extractObservationForTeam` already appends
the six tower HPs as the last of NUM_EXTRA_SCALARS, laid out as:

    tail+0        elapsed-time fraction
    tail+1, +2    both sides' cumulative elixir spend
    tail+3..+5    the OBSERVING team's king, left, right
    tail+6..+8    the opponent's king, left, right

where `tail = observation_size() - NUM_EXTRA_SCALARS`. `probe_perfect_defense
.own_tower_hp_fraction` is the existing in-repo reader for the same slice.

Each value is `hp / MAX_BUILDING_HP`, and a destroyed tower reads exactly 0.0
because it is no longer a live entity -- so filtering zeros reproduces
TimeoutRules' "surviving towers only" restriction precisely.

Comparing normalised floats rather than TimeoutRules' raw ints is order- and
equality-preserving here: the divisor is the same constant for both sides, and
one HP point is 1/4008 = 2.5e-4 apart, four orders of magnitude above float32's
resolution (~6e-8) at these magnitudes. Two different integer HPs cannot collide.

This mirrors engine LOGIC rather than an engine CONSTANT, but CLAUDE.md's rule
is the same either way -- derive it where the bindings allow, and where they do
not, pin it to its source by name. `perception/UPSTREAM_REQUESTS.md` carries a
proposal to bind `TimeoutRules::resolve` directly, which would let this module
collapse to a pass-through.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as _E  # noqa: E402


def weakest_tower_hp(env, team=0):
    """Lowest HP among `team`'s SURVIVING towers, normalised, or None if none.

    None mirrors TimeoutRules' `std::numeric_limits<int>::max()` sentinel for a
    side with nothing left standing: unreachable from step 2, because a side
    with zero towers cannot have tied the count against a side with any.
    """
    obs = np.asarray(env.get_observation_for_team(team), dtype=np.float32)
    tail = env.observation_size() - _E.ClashRoyaleEnv.NUM_EXTRA_SCALARS
    own = obs[tail + 3:tail + 6]
    alive = own[own > 0.0]
    return float(alive.min()) if alive.size else None


def score_from_towers(env, team=0):
    """1.0 win / 0.5 draw / 0.0 loss for `team`, by TimeoutRules' rules.

    Safe to call on any finished match, not just a timed-out one: when a King
    has actually fallen the tower counts already differ, so step 1 decides it
    and the tie-break is never consulted.
    """
    mine = env.get_towers_alive(team)
    theirs = env.get_towers_alive(1 - team)

    # 1. Fewer surviving towers loses.
    if mine != theirs:
        return 1.0 if mine > theirs else 0.0

    # 2. Equal counts -> the lower weakest tower loses. Both slices come from
    #    ONE observation (own = tail+3..5, opponent = tail+6..8), so this is a
    #    single call, and `team`'s own perspective supplies both halves.
    obs = np.asarray(env.get_observation_for_team(team), dtype=np.float32)
    tail = env.observation_size() - _E.ClashRoyaleEnv.NUM_EXTRA_SCALARS
    my_alive = obs[tail + 3:tail + 6]
    their_alive = obs[tail + 6:tail + 9]
    my_weakest = my_alive[my_alive > 0.0]
    their_weakest = their_alive[their_alive > 0.0]

    # Both empty means neither side has a tower standing -- an exact tie, and
    # the count check above already proved the two sides agree.
    if my_weakest.size and their_weakest.size:
        a, b = float(my_weakest.min()), float(their_weakest.min())
        if a != b:
            return 1.0 if a > b else 0.0

    # 3. Genuine draw -- the only way to get one.
    return 0.5


def terminal_value(env, team=0):
    """The same verdict as +1 / 0 / -1, for use as a search leaf value.

    Kept next to score_from_towers rather than derived ad hoc at the call site,
    so the two scales can never drift apart in what they consider a win.
    """
    return score_from_towers(env, team) * 2.0 - 1.0
