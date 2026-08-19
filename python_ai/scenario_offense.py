"""Offensive scenario injection: practice at the PUNISH, not at the defence.

    from scenario_offense import apply_offensive_scenario

WHAT THIS IS, AND WHY IT IS DEFAULT-OFF
---------------------------------------
This is "Proposal A" from the 2026-08-19 curriculum pivot, kept because it is
genuinely useful and demoted because it cannot do the job it was proposed for.

Scenario injection changes the STATE DISTRIBUTION, not the PAYOFF. CLAUDE.md
records four interventions built to make the agent play its win condition -- a
reward multiplier, an advisor target, random forcing at eps=0.15, and
gate-timed SMART forcing -- and all four returned null or negative, because
every one of them moves a POLICY while the environment kept pricing the card at
negative value. Injecting a state where the punish "should" be good does not
help if, in that state, playing it still loses; it just pays the negative price
more often, which is exactly the observed dose-response.

So the ORDER matters. The payoff is fixed first (teacher.py: a symmetric 1.0x
economy against a competent opponent), and only then is injection worth adding,
for the thing it is actually good at: CREDIT ASSIGNMENT. A punish window is rare
and its causal link to the outcome is buried in a long GAE trace -- the same
argument that justified the existing defensive `SCENARIO_INJECTION_PROB = 0.30`
in `train_selfplay.py`.

`OFFENSIVE_SCENARIO_PROB` therefore defaults to **0.0**. Turn it on only after
`prove_environment.py` shows the win condition paying in the new environment.

THE TWO SCENARIOS
-----------------
`punish_window`  The opponent has just committed: their bar is near 1 and their
                 units are crossing in ONE lane. We hold 8 elixir with the win
                 condition in hand. The correct play is to send it down the
                 OTHER lane while answering cheaply -- which is the entire
                 argument for a 2.6 deck and the one situation the agent has
                 essentially never been in, because it sits under 3 elixir on
                 65.3% of decisions.

`counter_push`   Our defence has just survived with units alive near the river
                 and ~6 elixir banked. The correct play is to push behind them.
                 This is the other half of the punish that a purely defensive
                 policy never discovers: surviving units are a free tank.

WHAT IT DOES NOT DO
-------------------
It does not touch the reward, and it must not. A scenario that also paid a bonus
would be a reward change wearing a curriculum's clothes, and the smoke run could
not attribute anything.

CONTRACTS THAT BITE
-------------------
* `set_hand_for_team` RETURNS FALSE and changes nothing on a hand that is not a
  valid permutation of that team's remaining pool. It is checked here, because a
  silently rejected setup is worse than no setup at all: the episode still counts
  as injected and the diagnostic would report practice that never happened.
* `inject` bypasses hand, elixir and placement legality entirely -- correct for
  building a hypothetical position, and the reason the spawn coordinates below
  are ABSOLUTE board coordinates rather than either team's own frame.
* Elixir is clamped to [0, 10] by the engine, so a caller cannot construct a bar
  the engine itself could never reach.
"""
import os

import numpy as np

import clash_royale_env as E

CE = E.ClashRoyaleEnv

# Off by default -- see the module docstring. The payoff is fixed first.
OFFENSIVE_SCENARIO_PROB = float(os.environ.get("CLASH_OFFENSIVE_SCENARIO_PROB", 0.0))

# These two bindings were added by commit 26de409 (2026-08-17) and the
# post-build copy into python_ai/ FAILED silently -- exactly the MSB3073
# file-lock failure CLAUDE.md documents, which "looks exactly like the compile
# is broken and almost never is". A .pyd that predates them raises
# AttributeError deep inside a scenario constructor, mid-episode, which is a
# terrible place to discover a stale build. Detect it once, here, instead.
HAS_STATE_SETTERS = all(hasattr(CE, m)
                        for m in ("set_elixir_for_team", "set_hand_for_team"))

# ABSOLUTE board coordinates (inject bypasses every frame conversion).
# River is [15.5, 17.5) and the bridges sit at x = 4 and x = 14; team 0 attacks
# toward HIGH y, team 1 toward LOW y.
BRIDGE_XS = (4.0, 14.0)
RIVER_Y = 16.5
OWN_SIDE_Y = 13.0        # our half, short of our Princess towers
ENEMY_SIDE_Y = 20.0      # their half, just past the river


def _wincon_and_filler(deck):
    """(wincon id, three other deck cards) -- the hand a punish scenario needs.

    The win condition is the deck's building-targeter, derived from the engine
    by `teacher.card_roles` rather than hardcoded, so this survives a deck
    change. Returns (None, None) for a deck with no win condition, which is a
    legitimate deck -- the scenario is then skipped rather than crashing.
    """
    from teacher import card_roles
    roles = card_roles(deck)
    wincon = next((c for c, r in roles.items() if r == "wincon"), None)
    if wincon is None:
        return None, None
    others = [c for c in deck if c != wincon][:3]
    if len(others) < 3:
        return None, None
    return wincon, others


def punish_window(env, rng, deck):
    """They just spent; we hold 8 and the win condition. Returns a name or None."""
    wincon, others = _wincon_and_filler(deck)
    if wincon is None:
        return None
    if not env.set_hand_for_team(0, [wincon] + others):
        # Loud rather than silent: an unusable hand means this episode is NOT a
        # punish scenario and must not be counted as one.
        return None
    env.set_elixir_for_team(0, 8.0)
    env.set_elixir_for_team(1, 1.0)

    # Their commitment, in one lane, already crossing. Two bodies so a single
    # cheap answer does not trivially erase it -- otherwise the "punish" is free
    # and teaches nothing about the trade.
    lane = int(rng.integers(2))
    x = BRIDGE_XS[lane]
    for cid, dy in ((wincon, 0.0), (others[0], 2.0)):
        env.inject(int(cid), float(x), float(RIVER_Y - dy), 1)
    return f"punish_window_lane{lane}"


def counter_push(env, rng, deck):
    """Our defence survived with units alive and elixir banked."""
    wincon, others = _wincon_and_filler(deck)
    if wincon is None:
        return None
    if not env.set_hand_for_team(0, [wincon] + others):
        return None
    env.set_elixir_for_team(0, 6.0)
    env.set_elixir_for_team(1, 3.0)

    lane = int(rng.integers(2))
    x = BRIDGE_XS[lane]
    # Survivors on OUR side of the river, healthy enough to escort a push.
    for cid, dy in ((others[0], 0.0), (others[1], 2.0)):
        env.inject(int(cid), float(x), float(OWN_SIDE_Y - dy), 0)
    return f"counter_push_lane{lane}"


SCENARIOS = (punish_window, counter_push)


def apply_offensive_scenario(env, rng, deck, prob=None):
    """With probability `prob`, rewrite the freshly-reset state into a punish.

    Returns the scenario's name, or None if nothing was applied -- including
    when the engine REFUSED the setup, so a caller counting injections counts
    only the ones that really happened.
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
