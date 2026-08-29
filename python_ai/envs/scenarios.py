"""Scenario injection: reshaping the START-STATE distribution, not the reward.

Industry precedent -- robotics resets from curated states, AlphaGo trained on
curated positions, "Backplay". The problem it targets here: a win condition
dropped on the bridge is a "defend in the next couple of seconds or lose the
tower" moment, but in a full 3600-tick game the causal link between that drop
and the tower loss ~40 ticks later is buried under a long, noisy GAE trace and
is a rare event -- so the reflex never accumulates gradient.

DESIGN CHOICES THAT KEEP THIS FROM BECOMING A DIFFERENT GAME. The opponent is
NOT frozen -- team 1 keeps playing its normal PFSP policy on top of the injected
threat; it is the real engine, board and towers; and the existing reward already
scores "defend efficiently and keep something alive to counter-push", so no
bespoke scenario reward is needed. The one artificial edge -- an optional short
truncation window (`max_steps`) -- is handled with a proper value BOOTSTRAP (see
`rl/gae.py`), never a terminal, so the critic does not learn a biased "the world
ends here" value.

A scenario dict is: name, spawns [(card_id, x, y)], max_steps (None runs to the
natural end of the game), defensive, and optionally require_own_card /
warmup_ticks.
"""
import numpy as np

from python_ai import engine_constants as EC

# --- Scenario injection (start-state distribution design) ------------------
# Industry precedent: reshaping the START-STATE distribution is how rare-but-
# critical situations get learned when normal play visits them too seldom for
# the credit-assignment horizon to connect cause and effect (robotics resets
# from curated states; AlphaGo trained on curated positions; "Backplay"). The
# problem this targets here: a win-condition (Hog/Giant/...) dropped on the
# bridge is a "defend in the next couple of seconds or lose the tower" moment,
# but in a full 3600-tick game the causal link between that drop and the tower
# loss ~40 ticks later is buried under a long, noisy GAE trace and is a rare
# event -- so the reflex never gets enough gradient. We fix that by STARTING a
# fraction of episodes already in that state so the net sees it constantly.
#
# Design choices that keep it from becoming a different game (the isolation
# failure mode): the opponent is NOT frozen -- team 1 keeps playing its normal
# PFSP policy on top of the injected threat; it's the real engine/board/towers;
# and the existing reward (win/loss + compute_shaping's HP/elixir-trade terms)
# already scores "defend efficiently + keep something alive to counter-push",
# so no bespoke scenario reward is needed. The one artificial edge -- an
# optional short truncation window (max_steps) that focuses each episode on the
# critical moment -- is handled with a proper value BOOTSTRAP (see the training
# loop's is_terminal/needs_boot split), never a terminal, so the critic doesn't
# learn a biased "the world ends here" value.
SCENARIO_INJECTION_PROB = 0.30

# Building-targeter win-conditions -- every id here confirmed against
# CardRegistry.h directly (not from memory) as Archetype::MeleeBuildingTargeter/
# RangedBuildingTargeter/a building with a persistent tower-damage role, i.e.
# guaranteed to beeline for a tower ignoring troops in its path, matching the
# scenario's own premise ("defend or lose the tower in the next few seconds").
# Miner (52) deliberately excluded despite being a real-game win condition --
# this engine registers him as plain Archetype::MeleeSquad (no building-
# targeter/dig-anywhere behavior implemented), so injecting him wouldn't
# actually exercise the "must answer a beelining threat" reflex this scenario
# is for. Goblin Barrel (109) / Graveyard (110) also excluded for now -- both
# are spell(...)-registered (PeriodicSpawnEffect), and inject_enemy's
# card->spawnEntity(...) path is only confirmed exercised (via Hog/Royal
# Giant) for a troop/building CardDefinition; using it for a spell-shaped one
# is unverified, not worth risking on a data-fill task.
_WIN_CONDITION_IDS = [
    15,  # Hog Rider
    18,  # Royal Giant
    45,  # Balloon
    2,   # Giant
    19,  # Golem
    81,  # Battle Ram
    82,  # Royal Hogs
    83,  # Wall Breakers
    84,  # Electro Giant
    87,  # Ram Rider
    88,  # Goblin Giant
    89,  # Skeleton Barrel
    91,  # Lava Hound
]

# Ranged units commonly played to escort/protect a win-condition push (the
# "supported" scenario's second spawn) -- confirmed RangedSquad/ranged-role
# troops, a mix of cheap chip support and real mid-fight damage.
_SUPPORT_IDS = [
    6,   # Musketeer
    1,   # Archers
    11,  # Wizard
    44,  # Baby Dragon
    63,  # Magic Archer
    36,  # Executioner
    20,  # Dart Goblin
]

# Real board coords for inject_enemy (team 1, low-y-bound), which bypasses
# isValidPlacement so an on-the-bridge spawn inside the river band is allowed.
#
# These are BOARD coordinates. Do not justify them from
# ClashEnv::extractObservationForTeam's `riverRow = 17` / x-band 3-4 & 13-14 --
# that is the OBSERVATION channel-8 marker, a wider visual hint painted for the
# network, and it is a different frame. perception/geometry.py warns against
# exactly this conflation. The board's own geometry (ArenaLayout.h, bound as
# clash_royale_env.ARENA_*) is river [15.5, 17.5) with bridges at x = 2.5 and
# 14.5 -- each spanning two cells, 2-3 and 14-15.
#
# `_RIVER_Y` is nonetheless correct and must not be "corrected": y = 17.0 is
# inside the band, injectEnemy applies no clamp, and Board::getNextWaypoint
# classifies 17.0 as neither bank and routes to the bridge exit -- which is
# precisely the on-the-bridge spawn this wants.
_RIVER_Y = 17.0

# DERIVED, never restated -- and it was restated, and it went stale.
# This read `[3.5, 13.5]` from when the arena put bridges at 4.0 and 14.0. The
# 2026-08-21 re-centring moved them to 2.5 / 14.5, and since cell i covers
# [i-0.5, i+0.5], x = 13.5 is the edge of cell 13, which is WATER. Every
# right-lane bridge push was being injected off the bridge, and nothing caught
# it because the scenario still "worked" -- the unit swam to the nearest
# waypoint and the episode looked normal.
#
# Eighth instance of the stale-arena-copy defect CLAUDE.md tracks. Unlike
# web/viewer.html, this module CAN reach the source of truth, so it must.
_BRIDGE_LANES = [EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X]

def _scenario_bridge_push(rng):
    """The exact case: one enemy win-condition on a random bridge, nothing
    else engineered. Short window -- the defense itself resolves in ~2-4 steps,
    the rest lets a counter-push start and get shaped-rewarded."""
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push",
        "defensive": True,
        "spawns": [(int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y)],
        "max_steps": 15,
    }

def _scenario_bridge_push_supported(rng):
    """Win-condition + a ranged support just behind it (same lane) -- a tankier,
    two-part threat that a single cheap defender can't fully answer. Longer
    window for the bigger commitment."""
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push_supported",
        "defensive": True,
        "spawns": [
            (int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y),
            (int(rng.choice(_SUPPORT_IDS)), lane, _RIVER_Y + 3.0),
        ],
        "max_steps": 25,
    }

# --- Fireball target practice ----------------------------------------------
# Added 2026-08-09 after measuring WHY the agent almost never casts Fireball.
# Over 120 trials per constructed situation, with Fireball affordable in 100%
# of them, the policy put only 0.03-0.11 probability on it against a 0.20
# uniform baseline -- and, critically, when it DID cast, the mean distance from
# the target centroid was 2.8 tiles in its own half and 7.7-11.5 tiles at an
# enemy tower, against a blast radius of 2.5.
#
# So the low usage is not timidity and not an exploration failure: it is an
# ACCURATE valuation of the agent's own aim. spell_value_shaping charges a full
# -1.00 * w for a cast that kills nothing, and a policy that cannot aim is
# correctly declining to pay it. Raising the cast incentive without fixing the
# aim would make it cast more and miss more -- which is the mechanism behind
# the recorded 97% -> 23% win-rate collapse when Fireball use was forced.
#
# Scenario injection is the right lever because it changes the START-STATE
# DISTRIBUTION, not the reward: it buys dense practice at the aiming problem
# without biasing the optimum. Exactly the argument that justified the
# bridge-push scenarios -- a high-value Fireball moment is rare and its credit
# is buried in a long GAE trace.
#
# Low-HP bodies only. Fireball does 689, so these die to one well-placed cast
# and a whiff is genuinely punished; Barbarians (691 HP) are deliberately NOT
# here, since surviving by 2 HP would teach that a perfect cast still failed.
_FIREBALL_SWARM_IDS = [
    41,  # Minions        (3 bodies)
    1,   # Archers        (2 bodies)
    64,  # Firecracker    (304 hp)
    6,   # Musketeer
]

def _scenario_fireball_swarm(rng):
    """A cheap swarm already inside our half -- the defensive value cast.

    Placed past the river and short of the Princess Towers, so it is a live
    threat the agent must answer THIS second rather than a distant one it can
    ignore. Several separate cards so the cluster is many bodies, which is what
    makes one Fireball a large positive elixir trade.
    """
    lane = rng.choice(_BRIDGE_LANES)
    n = int(rng.integers(3, 6))
    cx, cy = lane, 12.0
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-1.1, 1.1)),
                       float(cy + rng.uniform(-1.1, 1.1))))
    # Defensive: the swarm is inside OUR half, so "did not take a big hit"
    # is a real question with a real answer.
    return {"name": "fireball_swarm", "spawns": spawns, "max_steps": 12,
            "defensive": True}

def _scenario_fireball_tower_value(rng):
    """Enemy troops hugging their OWN princess tower -- the two-for-one cast.

    One Fireball centred here hits the troops and the tower together, which is
    the case spell_value_shaping was written to pay for and the one the agent
    currently misses by 7.7-11.5 tiles. Spawned just in front of the tower so
    both fall inside a single 2.5 radius.
    """
    tower_x = 4.0 if rng.random() < 0.5 else 14.0
    n = int(rng.integers(2, 5))
    cx, cy = tower_x, 25.6
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-0.9, 0.9)),
                       float(cy + rng.uniform(-0.9, 0.9))))
    # NOT defensive: the troops are at THEIR tower, nothing threatens us, so
    # "did not take a big hit" is true whatever the agent does -- including
    # doing nothing. Scoring it on that axis inflates ScenDef toward 1.0 and
    # says nothing about whether the cast was made or aimed.
    return {"name": "fireball_tower_value", "spawns": spawns, "max_steps": 15,
            "defensive": False}

# --- Giant: unmasking the win condition -------------------------------------
#
# The Giant is not undervalued, it is UNAFFORDABLE. Measured on the ep-17k
# checkpoint over 1,839 decision steps: the Giant is in hand on 80.6% of them
# but legal on only 4.7% (5.9% of in-hand), and when it IS legal the policy
# picks it at P = 0.157 against a uniform 0.200. It was sampled on 0.54% of
# steps, and 1 of 900 logged placements across the run was a Giant.
#
# That is a masked-slot problem, and it is the exact mechanism already on
# record for this deck: at 0.35 elixir per decision a 5-cost card is legal only
# after ~14 consecutive non-spending steps, so its slot is masked nearly every
# time it is checked and never accumulates gradient. No opponent, reward or
# entropy change reaches an action that is never sampled -- which is why this
# is a START-STATE change and not any of those.
#
# (builder_fn, weight). Extend freely -- offensive/punish/endgame scenarios
# drop in here with the same machinery. Set a scenario's "max_steps" to None
# to run it to the natural end of the game instead of a focused window.
#
# The two Fireball scenarios take half the injection budget, which is a
# REALLOCATION rather than an addition: ScenDef has been running at 0.99, so
# the bridge-push scenarios are saturated and no longer teaching the reflex
# they were added for. Their share drops from 100% to 50% of injected episodes
# (SCENARIO_INJECTION_PROB itself is unchanged at 0.30).
# giant_commit was REMOVED on 2026-08-19 and this is GAMEPLAY-AFFECTING.
# It required card id 2 (Giant) in our own hand, and DEFAULT_DECK became 2.6
# Hog Cycle on 2026-08-16 -- a deck that cannot contain it. So every time it was
# drawn it spent 24 futile `reset()` calls hunting for the card, fell through,
# and then ran a quiet board with banked elixir and no spawns: the one thing its
# own docstring said was NOT the point ("the whole point is an unmasked Giant
# slot"). At weight 1.0 of 6.0 it was diluting 17% of the injection budget into
# a no-op. Removing it redistributes that share back to the four live
# scenarios, so injected-episode composition changes and ScenDef/ScenOff are
# not comparable across this edit.
#
# Total is now 5.0: Fireball 60%, bridge 40%.
SCENARIOS = [
    (_scenario_bridge_push, 1.0),
    (_scenario_bridge_push_supported, 1.0),
    (_scenario_fireball_swarm, 1.5),
    (_scenario_fireball_tower_value, 1.5),
]

def sample_scenario(rng):
    builders, weights = zip(*SCENARIOS)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    return builders[rng.choice(len(builders), p=weights)](rng)
