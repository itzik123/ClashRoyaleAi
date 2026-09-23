"""The utility-search teacher: a strong deterministic bot, at a SYMMETRIC economy.

WHY THIS EXISTS
---------------
Every win rate that matters in this project was earned against the C++
`HeuristicOpponent` at a permanent 1.0x-1.5x elixir multiplier, and phase 1's
curriculum topped out at 1.5x. CLAUDE.md's "1.5x Curriculum Overfitting
Hypothesis" records the consequence, measured with a SMART-forced A/B (advisor
timing gate + advisor bridge cell), n=120 paired:

    opponent   baseline win   forced win   delta      p
    1.00x         1.000          1.000     +0.0000    VOID (ceiling)
    1.25x         0.950          0.825     -0.1250    0.0059
    1.50x         0.617          0.317     -0.3000    3.2e-06

The cost of playing the win condition falls monotonically as the opponent's
economy falls. Mechanically: a punish window lasts about
`answer_cost / (m * r)`, so at m=1.5 it is two thirds its natural length, while
our 4 elixir is spent regardless and what they do with their surplus scales with
m. Expected value moves as 1/m, expected counter-cost as m. Defence moves the
OPPOSITE way, because a multiplier increases exactly the threat volume defence
is priced against. The multiplier does not shift the optimum, it INVERTS the
ranking of strategy classes.

That is why four separate interventions -- a reward multiplier, an advisor
target, random forcing, and gate-timed smart forcing -- all returned null or
negative. Every one of them moves a POLICY. None of them changes the PAYOFF.

THE TRAP IN THE OBVIOUS FIX
---------------------------
Deleting the multiplier is necessary and NOT sufficient. At 1.0x the C++
heuristic is beaten ~100% -- recorded for the ep-64k Giant-deck net and again
for the ep-25202 2.6 net. Removing the handicap without replacing the opponent
converts a MISPRICED environment into a ZERO-GRADIENT one, which is exactly why
the 1.00x row above is VOID.

So the multiplier is replaced by COMPETENCE. Both sides always run at 1.0x and
difficulty is dialed on this bot's lookahead, candidate width and epsilon
(`TEACHER_STAGES`).

WHY HAND-WRITTEN AND NOT THE NEURAL SEARCH WE ALREADY OWN
---------------------------------------------------------
`shipping.py` (cured weights + search horizon 12) scores 0.9225 against
heuristic@1.5x and would be a far stronger teacher -- but it cannot be used at
Episode 0. Search scores candidates with the NET'S OWN CRITIC, so at random init
the expert is not an expert: measured paired, n=60, random init @1.0x scores
0.450 policy / 0.417 search while overriding 21.5% of decisions, against the
trained net's 0.483 / 0.667 at 10.2%. It deviates twice as often and gains
nothing.

A hand-written utility function has no cold start. That is the decisive argument
for this module existing at all.

ARCHITECTURE: RULES PROPOSE, SIMULATION RANKS
--------------------------------------------
The external blueprint that motivated this asked for "no if-else trees, a pure
utility function over cards x legal tiles". That is half wrong here, and the
measurements say so. The hand-written rules in `tactics.py` are the strongest
placement signal this project has ever produced:

    card                          advisor   trained net   random legal cell
    Cannon   (tower HP preserved)   564.1        12.1           353.5
    Fireball (elixir killed)         2.405        0.000           0.276
    Giant    (enemy tower damage)   535.6         3.3            86.7

So the rules are the CANDIDATE GENERATOR and forward simulation is the RANKER.
Enumerating 4 cards x 612 tiles and rolling each one out costs ~100x more for no
gain, and CLAUDE.md already records that the advisor's essentially two-cell
Giant rule is "a very strong prior this board rewards".

Measured cost on this box (i5-13420H, 2026-08-19):

    snapshot        0.0352 ms      candidate @3s   0.2500 ms  -> K=16: 4.00 ms
    10-tick step    0.0598 ms      candidate @6s   0.4403 ms  -> K=16: 7.04 ms

THE SCORE, AND THE ONE THING THE BLUEPRINT GOT BACKWARDS
--------------------------------------------------------
Every candidate is rolled forward on `env.snapshot()` and scored by DIFFERENCE
against the NO-OP rollout. That baseline is what makes each score a MARGINAL
value: it subtracts off whatever was going to happen anyway, so a candidate is
credited only for what it CHANGES, and doing nothing scores exactly 0.0.

The blueprint expected the baseline to absorb its `- w5 * OpportunityCost` term.
It does not, and saying so plainly matters: the baseline removes COUNTERFACTUAL
value, but the elixir actually spent is still real and still has to be charged.
`W_COST` does that, in the same currency as `W_TRADE` -- one unit per elixir,
because elixir spent and elixir destroyed are the same thing.

THE MYOPIA PROBLEM, AND WHY THERE IS A POSITIONAL TERM
-------------------------------------------------------
A Hog placed at the bridge deals ZERO tower damage inside a 3-6 s horizon: it
has ~12 tiles to cross at Fast speed, which is ~13 s. Scored on realized damage
alone the teacher would refuse to ever play its win condition and would turtle --
reproducing the exact pathology this whole pivot exists to remove.

`positional_advantage` is the terminal evaluation that fixes it: our units'
HP weighted by how far up the board they are, minus the enemy's the same way. It
is the hand-written stand-in for the value bootstrap that `search_ab_test` gets
from the critic. Without it the teacher is a turtle by construction.

WHY IT READS THE OBSERVATION AND NOT THE ENGINE
------------------------------------------------
Same contract `tactics.py` holds, for the same reason:
`perception/tests/test_encoder_matches_engine.py` pins the live encoder bit-equal
to `getObservationForTeam(0)`, so anything computed from an observation behaves
identically in simulation and on a real screen. `env` is used ONLY for
`snapshot()`, legality and hand/elixir -- never to read entity positions.

SIDE-AGNOSTIC BY CONSTRUCTION, AND THE FRAME TRAP
---------------------------------------------------
`getObservationForTeam(1)` is already mirrored, and `stepSelfPlay` mirrors team
1's action back with `y_abs = (BOARD_HEIGHT - 1) - y1`. So a bot written purely
against "own observation -> own action frame" plays either side unchanged, which
is what makes `prove_environment.py`'s Teacher-vs-Teacher test possible.

BUT `is_valid_placement(card, x, y, team)` takes ABSOLUTE y for BOTH teams.
Verified empirically 2026-08-19: own-frame y=15.0 maps to absolute 18.0 and is
legal for team 1, while the unmirrored 15.0 is not. Mixing the two frames makes
every team-1 placement illegal -- and the symptom is a bot that quietly never
plays anything, which reads as "weak teacher", not as "broken code". That is the
same silent-failure shape as the 2026-07-31 team-1 observation bug.
`to_absolute_y` is the single conversion point and
`test_teacher_candidates_are_all_legal_for_either_team` pins both sides.
"""
import functools
import itertools

import numpy as np

import clash_royale_env as E
from python_ai import engine_constants as EC
from python_ai.advisors import card_probes, tactics

CE = E.ClashRoyaleEnv

# `execute_steps` steps candidate rollouts with the OBSERVATION-FREE binding
# (perception/UPSTREAM_REQUESTS.md item 21). Probed once, here, and fatally --
# NOT with a getattr fallback.
#
# A fallback would keep running on a stale `.pyd` while silently handing back
# the entire speedup, and "the .pyd was stale" is a trap this project has
# already been bitten by once: it predated the commit adding
# set_elixir_for_team / set_hand_for_team and silently blocked two measurements.
# The failure mode of a fallback is a performance regression nobody can see; the
# failure mode of this is one line naming the fix.
if not hasattr(CE, "step_self_play_fast"):
    raise ImportError(
        "clash_royale_env is missing step_self_play_fast, so it predates "
        "UPSTREAM_REQUESTS item 21. The .pyd is stale -- rebuild it:\n"
        '  "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community'
        '\\MSBuild\\Current\\Bin\\MSBuild.exe" '
        "build_python\\clash_royale_env.vcxproj "
        "/p:Configuration=Release /p:Platform=x64 /m\n"
        "(from the PowerShell tool, never Bash -- see CLAUDE.md), then "
        "tools/audit/verify_pyd.py to confirm the copy landed.")

BOARD_H = CE.BOARD_HEIGHT
BOARD_W = CE.BOARD_WIDTH
HAND_SIZE = CE.HAND_SIZE
N_CH = CE.NUM_CHANNELS
PLANE = BOARD_H * BOARD_W
MAX_TROOP_HP = CE.MAX_TROOP_HP

# The mirror axis stepSelfPlay itself uses: realY1 = (BOARD_HEIGHT - 1) - y1.
MIRROR_Y = float(BOARD_H - 1)

# Ally troop channels (0-2). tactics.py owns the enemy side (4-6); this is the
# ally mirror of it, needed for the positional term and for support placement.
CH_ALLY_TROOP = (0, 1, 2)
CH_ALLY_COUNT = CE.CH_COUNT           # ally is the base index, enemy is base+1

# Overflow threshold, matching train.W_ELIXIR_OVERFLOW's own 9.0. Above it the
# elixir bar is throwing away income, so spending is close to free and the cost
# charge should not stop it.
ELIXIR_OVERFLOW_AT = 9.0

# Where `effective_play_margin`'s taper STARTS. Deliberately NOT
# ELIXIR_OVERFLOW_AT, and deliberately a separate constant from it.
#
# MEASURED 2026-08-28, replay_ep2018.json (stage 1, ep 2018). Tying the taper to
# the 9.0 overflow line meant the bar sat at the FULL play_margin for every
# elixir value from 0 to 9, so the teacher was maximally reluctant across almost
# its whole operating range and only relented in the last elixir before
# overflow. Observed: it lost a Princess Tower at tick 159 having spent 2 elixir
# (one Ice Golem, tick 111) while its bar ran 5.0 -> 8.6, and its next play
# landed at tick 561 -- the exact tick elixir first reached 9.60.
#
# That is the same failure `effective_play_margin`'s own docstring was written
# to prevent ("against a PASSIVE opponent nothing clears a fixed 3.0, so the bot
# froze"); the taper fixed the shape but started too late to fix the range.
# 6.0 is the midpoint of the bar: below it holding really is cheap and the
# teacher should be picky, above it income is increasingly at risk.
#
# GAMEPLAY-AFFECTING for phase 1. The stage 0/1 teacher defends materially more,
# so every curriculum win-rate gate is calibrated against a different opponent
# and win rates earned before this date are not comparable across it.
MARGIN_TAPER_START = 6.0


# --------------------------------------------------------------------------
# card roles -- derived from the engine, never a hardcoded id list
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _card_table_cached(deck_key):
    return _card_table_uncached(list(deck_key))


def card_peak_hp(deck):
    """{card_id: single-body HP} for the deck's TROOPS, from the engine.

    PEAK CELL, not the sum: channels 0-7 OVERWRITE rather than accumulate
    (open problem #4), so a 3-body Skeletons spreads over 3 cells and its sum
    reads 243 while one skeleton is 81. The tank question is "how much does ONE
    body absorb", which is the max.

    Spells and buildings are absent by construction -- neither can escort.
    """
    return dict(_card_table_cached(tuple(deck))["hp"])


def tank_id(deck):
    """The card that goes IN FRONT of the win condition, or None.

    DERIVED, never a literal. `DEFAULT_DECK` has already changed twice, and a
    hardcoded 40 would quietly mean "Ice Golem" forever -- the same failure mode
    `card_roles` exists to avoid. The tank is simply the highest-HP troop that
    is not itself the win condition: Ice Golem 1315 > Musketeer 721 >
    Ice Spirit 230 > Skeletons 81 for 2.6, and the Hog's own 1697 is excluded
    because it is the card being escorted.
    """
    table = _card_table_cached(tuple(deck))
    roles, hp = table["roles"], table["hp"]
    troops = [c for c in hp if roles.get(c) != "wincon"]
    if not troops:
        return None
    return max(troops, key=lambda c: (hp[c], -c))


def card_roles(deck):
    """{card_id: "wincon"|"spell"|"building"|"ranged"|"melee"} for one deck.

    Memoized by deck, because the derivation below builds one `ClashRoyaleEnv`
    per non-spell non-building card and phase 1 runs `num_envs = 8` teachers
    that would otherwise each redo it. Returns a copy so a caller cannot mutate
    the cached table.

    DERIVED BY INJECTION, the same technique `gym_wrapper._find_win_condition`
    already uses: `get_card_info` exposes cost/is_spell/is_building but no
    archetype, so the class is recovered by injecting the card on an empty board
    and reading which of ClashEnv's type channels lights up (0 melee, 1 ranged,
    2 building-targeter, 3 building).

    A hardcoded `{15: "wincon"}` would silently mean "Hog Rider" forever and be
    wrong the next time `DEFAULT_DECK` changes -- which it already has, twice.

    The win condition itself comes from `resolve_win_condition`, which
    `gym_wrapper._find_win_condition` now shares. The two used to be separate
    copies, and both ranked by cost -- which is how the agent's reward term came
    to credit an Ice Golem as a win condition (audit 07, F2).
    """
    return dict(_card_table_cached(tuple(deck))["roles"])


# How long a tower-threat probe rolls forward. A siege building has to survive
# its deploy time, acquire and then out-range the tower; 1200 ticks (2:00) is
# comfortably past that and still well inside a match.
TOWER_THREAT_PROBE_TICKS = 1200
#: Deck for the probes below. Contents are irrelevant -- the card under test is
#: injected, never played from hand -- but a ClashRoyaleEnv needs eight ids.
_PROBE_DECK = None


def _probe_env():
    """An empty board, both sides holding some legal deck."""
    global _PROBE_DECK
    if _PROBE_DECK is None:
        _PROBE_DECK = list(E.get_all_card_ids())[:HAND_SIZE * 2]
    env = CE(_PROBE_DECK, _PROBE_DECK, 3600)
    env.reset()
    return env


@functools.lru_cache(maxsize=1)
def _probe_env_cached():
    """One shared empty board for pure legality queries (never stepped)."""
    return _probe_env()


def _enemy_tower_hp(env):
    return sum(max(0.0, env.get_tower_hp(1, s)) for s in (0, 1, 2))


@functools.lru_cache(maxsize=1)
def own_half_max_row():
    """The furthest-forward row a BUILDING may legally occupy, from the engine.

    This is the siege row and there is no second copy of it: `ArenaLayout` owns
    the geometry and `get_own_half_max_y` is how Python reads it.
    """
    return int(_probe_env().get_own_half_max_y())


@functools.lru_cache(maxsize=256)
def siege_reach(card_id):
    """Enemy tower HP a BUILDING takes from the furthest-forward legal row.

    MEASURED, never a name list. Sweeping all 170 legal cells for a Mortar on
    2026-09-06: exactly 34 of them damage the enemy tower (rows 13-15) and the
    other 136 do precisely ZERO. Cannon, Tesla, Inferno Tower, Bomb Tower and
    Tombstone score 0 from every cell; Mortar reads 1596 and X-Bow 3824.

    So this is not "is the building good" -- it is the sharp, binary question of
    whether the card has any route to a tower at all, which is what decides
    whether a deck HAS a win condition.

    `step_self_play`, never `step`: plain `step` runs the C++ HeuristicOpponent,
    which would defend against the probe. And the score is enemy tower HP
    actually lost, not `get_tower_damage_dealt`, which reads 1620 on a board
    with nothing whatever on it.
    """
    if not E.get_card_info(card_id)["is_building"]:
        return 0.0
    best = 0.0
    row = float(own_half_max_row())
    for x in (float(int(EC.BOARD_CENTER_X)), float(int(EC.LEFT_LANE_X)),
              float(int(EC.RIGHT_LANE_X))):
        env = _probe_env()
        if not env.is_valid_placement(card_id, x, row, 0):
            continue
        before = _enemy_tower_hp(env)
        env.inject(card_id, x, row, 0, -1.0, 0)
        for _ in range(TOWER_THREAT_PROBE_TICKS // 10):
            env.step_self_play(HAND_SIZE, 0.0, 0.0, HAND_SIZE, 0.0, 0.0, 10,
                               False, False, False, False)
        best = max(best, before - _enemy_tower_hp(env))
    return best


@functools.lru_cache(maxsize=256)
def spell_spawns_bodies(card_id):
    """Does this SPELL put friendly bodies on the board?

    THE DISCRIMINATOR HAS TO BE THE SPAWN, not the damage. Every direct spell
    hurts a tower it is cast on -- Fireball, Rocket and Poison all do -- so
    "does it damage the enemy tower" would promote Rocket to win condition in
    log bait and Fireball in mortar cycle. What makes Goblin Barrel a win
    condition is that it delivers three bodies onto the tower, and bodies are
    the thing a spell cannot otherwise produce.

    Measured on an empty board: Goblin Barrel 3 bodies, Graveyard 1, Fireball /
    The Log / Rocket / Zap / Arrows / Poison / Tornado 0.
    """
    if not E.get_card_info(card_id)["is_spell"]:
        return 0
    env = _probe_env()
    env.inject(card_id, float(int(EC.LEFT_LANE_X)), float(EC.princess_y(1)),
               0, -1.0, 0)
    peak = 0
    for _ in range(6):                      # past the ~10-tick cast delay
        env.step_self_play(HAND_SIZE, 0.0, 0.0, HAND_SIZE, 0.0, 0.0, 10,
                           False, False, False, False)
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        peak = max(peak, sum(int((obs[ch * PLANE:(ch + 1) * PLANE] > 1e-6).sum())
                             for ch in CH_ALLY_TROOP))
    return peak


def spell_geometry(card_id):
    """(radius, damage) the teacher aims `card_id` with: the card's OWN, measured.

    Every spell used to be aimed with FIREBALL's disc -- `tactics.spell_catch_map`
    called with its defaults (radius 2.5, a 689 cap per unit) whatever was in
    hand, in `_top_spell_cells`, both spell combos and the rung-0 rules gate.
    `advisor_target` was moved to per-card geometry on 2026-09-15; this was the
    copy that audit missed.

    Measured 2026-09-16 on 320 mid-match boards against eight pool decks, each
    cell scored by the ENGINE (the elixir value the cast actually killed):

        Rocket    1.85 -> 2.54  (+38%; better on 75 boards, worse on 4)
        Zap       0.62 -> 0.69  (+11%; 5 / 0)
        Poison    1.42 -> 1.52  (+7%; 18 / 8)
        Arrows    1.50 -> 1.56  (+4%; 7 / 2)
        Lightning 3.50 -> 3.50  (8 / 8)
        Fireball  identical cell on 320 / 320 -- the 2.6 mirror is unchanged

    Rocket is the one that mattered: capped at Fireball's 689, a Rocket on a
    tank scored no better than a Rocket on a Skeleton pile, so a 6-elixir spell
    went to swarms. A card the probe declines (The Log, a roller) keeps the
    Fireball disc it was aimed with before -- its value is a corridor, which no
    disc describes.
    """
    effect = card_probes.spell_effect(int(card_id))
    if effect is None:
        return tactics.FIREBALL_RADIUS, tactics.FIREBALL_DAMAGE
    return effect


def spell_catch_for(obs, card_id):
    """`tactics.spell_catch_map` for THIS spell's own radius and damage."""
    radius, damage = spell_geometry(card_id)
    return tactics.spell_catch_map(obs, radius, 0, damage)


@functools.lru_cache(maxsize=256)
def building_spawns_bodies(card_id):
    """Peak count of friendly TROOP bodies a building puts on an empty board.

    The discriminator `siege_reach` alone lacks. "Damages the enemy tower from
    the own half" is true of an X-Bow, which FIRES at it, and equally of a
    Barbarian Hut, whose Barbarians WALK to it. Measured 2026-09-23 (400 ticks):

        X-Bow 0   Mortar 0            -- siege: the building itself fires
        Barbarian Hut 5   Tombstone 6   Goblin Cage 1   Goblin Hut 1
        Goblin Drill 4                -- spawners
    """
    if not E.get_card_info(card_id)["is_building"]:
        return 0
    env = _probe_env()
    row = float(own_half_max_row())
    x = float(int(EC.BOARD_CENTER_X))
    if not env.is_valid_placement(card_id, x, row, 0):
        x = float(int(EC.LEFT_LANE_X))
    env.inject(card_id, x, row, 0, -1.0, 0)
    peak = 0
    for _ in range(40):
        env.step_self_play(HAND_SIZE, 0.0, 0.0, HAND_SIZE, 0.0, 0.0, 10,
                           False, False, False, False)
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        peak = max(peak, sum(int((obs[ch * PLANE:(ch + 1) * PLANE] > 1e-6).sum())
                             for ch in CH_ALLY_TROOP))
    return peak


def siege_building(card_id):
    """A building that attacks the enemy tower ITSELF, from the own half.

    ONE definition, read by both the win-condition resolver and
    `card_probes.building_defends`. It was `siege_reach > 0` in both, which
    also admitted every SPAWNER: the resolver named Tombstone the win condition
    of a Splashyard deck over its Graveyard (Tombstone's 1134 in a 1200-tick
    siege window against the Graveyard's 730 in a 300-tick troop window), and a
    Barbarian Hut outranked the Giant beside it at 6182. Measured over 28 decks
    on 2026-09-23: those two change and the other 26 -- all 16 of the pool --
    resolve exactly as before. TODO 00.6.

    A deploy-anywhere building (Goblin Drill) is not a siege building either: it
    is played BESIDE the enemy tower like a Miner, and from the own siege row it
    measured 0 tower damage in 300 ticks against 2654 from beside the tower.
    """
    info = E.get_card_info(card_id)
    if not info["is_building"] or info.get("deploy_anywhere", False):
        return False
    return siege_reach(card_id) > 0.0 and building_spawns_bodies(card_id) == 0


#: How long the win-condition probe rolls a lone attacker forward. SHORT on
#: purpose: over two minutes almost anything that walks at an undefended tower
#: eventually takes it, so a long window saturates and stops separating a Hog
#: from an Ice Golem. Thirty seconds measures what the card is FOR.
WINCON_PROBE_TICKS = 300

#: A win condition must take at least this much enemy tower HP per elixir in
#: WINCON_PROBE_TICKS, alone, from where it is played to attack.
#:
#: MEASURED 2026-09-15 (`wincon_damage_per_elixir`, 300 ticks):
#:
#:     Mighty Miner 1636   Royal Hogs 773   Goblin Drill 664   Ram Rider 652
#:     X-Bow 637   Hog Rider 634   Battle Ram 634   Miner 582   Royal Giant 525
#:     Giant 507   Balloon 507   Goblin Giant 422   Mortar 399   Wall Breakers 350
#:     Golem 312   Electro Giant 256   Goblin Barrel 240   GRAVEYARD 146
#:     Lava Hound 129   Ice Golem 84   Skeleton Barrel 81   Fireball 0   Cannon 0
#:
#: So the floor can NOT separate "real win condition" from "cheap tank": a
#: Skeleton Barrel (81) is a genuine win condition and scores level with an Ice
#: Golem (84), and a Graveyard (146) sits barely above a Lava Hound (129). A floor at 150
#: would have silently returned None for graveyard_control -- the very
#: regression the 2026-09-06 fix removed. The window also saturates at one
#: Princess (2534 HP), which is why so many cards read 2534/cost.
#:
#: The floor therefore only excludes cards with NO route to a tower. RANKING is
#: what fixes the measured defects (Miner over Ice Golem, Balloon over Lava
#: Hound). A deck whose best candidate is weak still gets one -- the teacher
#: stays offensive -- and `validate_deck` WARNS about it at startup.
WINCON_MIN_DAMAGE_PER_ELIXIR = 50.0
#: Below this the resolved win condition is reported as WEAK by validate_deck.
WINCON_WEAK_DAMAGE_PER_ELIXIR = 200.0


def _roll_idle(env, ticks):
    for _ in range(ticks // 10):
        env.step_self_play(HAND_SIZE, 0.0, 0.0, HAND_SIZE, 0.0, 0.0, 10,
                           False, False, False, False)


def _attack_cell(card_id, env):
    """Where this card is PLAYED to attack, or None when it has no such cell.

    A walking building-targeter starts on our own side of the left bridge; a
    deploy-anywhere troop (Miner, Goblin Drill) goes next to the enemy tower,
    which is the entire point of the card; a body-spawning spell lands on it.
    """
    info = E.get_card_info(card_id)
    lane_x = float(int(EC.LEFT_LANE_X))
    tower_y = float(EC.princess_y(1))
    if info["is_spell"]:
        # The cell must be one the card can actually be CAST on. `inject`
        # bypasses legality, and a rolling spell (Barbarian Barrel) is confined
        # to our own half and the river -- injected onto the tower anyway it
        # scored 617 and was promoted to win condition of graveyard_control
        # over the Graveyard itself.
        return (lane_x, tower_y) if env.is_valid_placement(card_id, lane_x, tower_y, 0) else None
    if info.get("deploy_anywhere", False):
        for dy in (3.0, 4.0, 2.0, 5.0):
            for dx in (0.0, 1.0, -1.0, 2.0):
                x, y = lane_x + dx, tower_y - dy
                if env.is_valid_placement(card_id, x, y, 0):
                    return (x, y)
        return None
    return (lane_x, float(own_half_max_row()))


@functools.lru_cache(maxsize=512)
def wincon_damage_per_elixir(card_id):
    """Enemy tower HP per elixir this card takes ALONE, from its attack cell.

    MEASURED, not ranked by cost. Ranking by cost was the defect this replaces:
    "the most expensive building-targeter" promoted a 2-elixir Ice Golem to win
    condition on any deck whose real win condition is not a building-targeter
    (a Miner deck), and inverted LavaLoon (Lava Hound over Balloon). Buildings
    reuse `siege_reach`, which already sweeps the siege row over 1200 ticks.

    Scored as enemy tower HP ACTUALLY LOST, never `get_tower_damage_dealt`.
    """
    info = E.get_card_info(card_id)
    cost = max(float(info["cost"]), 1.0)
    if info["is_building"] and not info.get("deploy_anywhere", False):
        return siege_reach(card_id) / cost
    if info["is_spell"] and spell_spawns_bodies(card_id) == 0:
        return 0.0
    env = _probe_env()
    cell = _attack_cell(card_id, env)
    if cell is None:
        return 0.0
    before = _enemy_tower_hp(env)
    env.inject(card_id, cell[0], cell[1], 0, -1.0, -1)
    _roll_idle(env, WINCON_PROBE_TICKS)
    return (before - _enemy_tower_hp(env)) / cost


def _wincon_eligible(card_id, is_building_targeter):
    """Could this card be a deck's route to a tower at all?

    A building-targeter walks past defenders; a deploy-anywhere troop or
    building skips them; a siege building out-ranges the tower; a spawning spell
    delivers bodies onto it. A Knight or a P.E.K.K.A. also damages an EMPTY
    tower, so measured damage alone would promote them -- eligibility is by
    CLASS, and measurement only ranks within it. A SPAWNER building (Tombstone,
    Barbarian Hut, Goblin Cage) reaches a tower only by walking bodies there, the
    same way a Knight does, and is not a class that wins games -- see
    `siege_building`.
    """
    info = E.get_card_info(card_id)
    if info["is_building"]:
        if info.get("deploy_anywhere", False):
            return wincon_damage_per_elixir(card_id) > 0.0
        return siege_building(card_id)
    if info["is_spell"]:
        return spell_spawns_bodies(card_id) > 0
    return is_building_targeter or bool(info.get("deploy_anywhere", False))


def resolve_win_condition(deck, building_targeters):
    """THE deck's win condition, or None. One definition for the whole repo.

    Used by the teacher's role table AND by `gym_wrapper._find_win_condition`,
    which feeds the agent's `W_WIN_CONDITION_DAMAGE` reward term. Two copies of
    this question had already diverged once each way.
    """
    eligible = [c for c in deck
                if _wincon_eligible(c, c in building_targeters)]
    scored = []
    for c in eligible:
        per = wincon_damage_per_elixir(c)
        if per >= WINCON_MIN_DAMAGE_PER_ELIXIR:
            scored.append((per * max(float(E.get_card_info(c)["cost"]), 1.0), per, c))
    if not scored:
        return None
    # Ranked by ABSOLUTE tower damage, then by damage per elixir. Per-elixir
    # first was measured wrong: the probe saturates at one Princess (2534), so
    # every card that takes a tower reads 2534/cost and the CHEAPEST wins --
    # which named the Miner (1746 absolute) over the Balloon (2534) in LavaLoon.
    # The per-elixir tiebreak still puts Hog (634) over Giant (507) at 2534.
    return max(scored, key=lambda t: (t[0], t[1], -t[2]))[2]


def _card_table_uncached(deck):
    """One injection pass, two derived tables.

    Roles and HP come from the SAME probe because it is the same probe: the
    injection that reads which type channel lights up also carries the unit's
    normalised HP in that channel. Deriving them separately would double the
    ~8 throwaway `ClashRoyaleEnv` constructions this costs per deck.
    """
    roles = {}
    hp = {}
    targeters = []
    for cid in deck:
        info = E.get_card_info(cid)
        if info["is_spell"]:
            roles[cid] = "spell"
            continue
        if info["is_building"]:
            roles[cid] = "building"
            continue
        env = CE(list(deck), list(deck), 100)
        env.reset()
        env.inject(cid, 9.0, 8.0, 0)
        env.step(HAND_SIZE, 0.0, 0.0, 1)
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        lit = [ch for ch in range(3)
               if float(obs[ch * PLANE:(ch + 1) * PLANE].max()) > 1e-6]
        hp[cid] = float(max(
            (obs[ch * PLANE:(ch + 1) * PLANE].max() for ch in range(3)),
            default=0.0)) * MAX_TROOP_HP
        if 2 in lit:
            targeters.append(cid)
            roles[cid] = "melee"          # provisional; promoted below
        elif 1 in lit:
            roles[cid] = "ranged"
        else:
            roles[cid] = "melee"
    # ONE resolver, measured -- see resolve_win_condition. The two branches this
    # replaces ranked by COST: "most expensive building-targeter", with a
    # siege/spawning-spell fallback only when there was none. Measured
    # 2026-09-15 (audit 05) that promoted a 2-elixir Ice Golem on a Miner deck,
    # returned None for Miner control outright (a Miner targets ground), and
    # inverted LavaLoon.
    wincon = resolve_win_condition(deck, set(targeters))
    if wincon is not None:
        roles[wincon] = "wincon"
    return {"roles": roles, "hp": hp}


# --------------------------------------------------------------------------
# cycle tracking -- exact for our own hand, free, no engine change
# --------------------------------------------------------------------------
class CycleTracker:
    """How many plays until a given card returns to hand.

    The hand is a deterministic rotation: a played card goes to the back of the
    4-card queue behind the hand. The queue is not observable, but the ORDER
    cards left the hand is, and that order IS the queue order. So our own cycle
    is exact and costs nothing beyond watching `get_hand_for_team`.

    The initial queue (the 4 cards not in the opening hand) has no observable
    order, so it starts in deck order and self-corrects within one full cycle.
    That is a bounded, one-time inaccuracy, and it is why `cycle_value` is a
    SCORED TERM rather than a hard gate: a wrong estimate then costs a little
    ranking quality instead of freezing the bot into a wrong line.

    LIMITATION, stated rather than hidden: a deck with duplicate card ids would
    confuse the identity-based bookkeeping below. `sample_random_deck` draws
    without replacement and `DEFAULT_DECK` has no duplicates, so this does not
    arise today.
    """

    def __init__(self, deck):
        self.deck = list(deck)
        self.reset()

    def reset(self):
        self._hand = None
        self._queue = []

    def observe(self, hand):
        hand = list(hand)
        if self._hand is None:
            self._hand = hand
            self._queue = [c for c in self.deck if c not in hand]
            return
        for c in self._hand:
            if c not in hand:                      # left the hand => was played
                if c in self._queue:
                    self._queue.remove(c)
                self._queue.append(c)
        for c in hand:
            if c not in self._hand and c in self._queue:
                self._queue.remove(c)              # arrived from the queue
        self._hand = hand

    def distance_to(self, card_id):
        """0 if it is in hand, else how many plays until it arrives."""
        if self._hand is not None and card_id in self._hand:
            return 0
        if card_id in self._queue:
            return self._queue.index(card_id) + 1
        return len(self.deck)


# --------------------------------------------------------------------------
# weight profiles
# --------------------------------------------------------------------------
# Units are ELIXIR-EQUIVALENTS throughout, which is what makes the weights
# comparable at all: W_TRADE is 1.0 because one elixir destroyed is worth one
# elixir, and everything else is priced against that.
#
#   w_twr / w_def  per point of tower HP (100 HP ~ 1 elixir at 0.01)
#   w_trade        per elixir of value killed / lost
#   w_crown        per net surviving tower
#   w_pos          per unit of positional advantage (see positional_advantage)
#   w_cost         per elixir spent -- the opportunity cost the no-op baseline
#                  does NOT absorb
#   w_cycle        per unit of cycle value
#
# These are STARTING values selected on the reasoning above, not on an outcome.
# `prove_teacher.py --sweep` re-selects them against win rate versus the C++
# heuristic, and the chosen profile is then CONFIRMED on a fresh independent
# run -- the protocol this project adopted after a +0.105 collapsed to +0.016
# under 4x the power.
PROFILES = {
    "aggressive": dict(w_twr=0.015, w_def=0.007, w_trade=0.8, w_crown=5.0,
                       w_pos=25.0, w_cost=1.0, w_cycle=0.40),
    "balanced":   dict(w_twr=0.010, w_def=0.010, w_trade=1.0, w_crown=5.0,
                       w_pos=20.0, w_cost=1.0, w_cycle=0.30),
    "defensive":  dict(w_twr=0.006, w_def=0.016, w_trade=1.2, w_crown=5.0,
                       w_pos=15.0, w_cost=1.0, w_cycle=0.20),
}

# The competence ladder that replaces CURRICULUM_STAGES' elixir multipliers.
#
# LOOKAHEAD IS THE PRIMARY PROGRESSION AXIS. Stage 0 is deliberately
# short-sighted: it still places structurally well, because the candidate cells
# come from tactics.py either way, but with 1 s of foresight it cannot see a
# trade going bad and an unpolished agent can beat it. The top rung simulates a
# full 10 s forward, which is long enough to watch a committed push arrive, be
# answered, and be counter-pushed -- the whole exchange the earlier gates could
# not see.
#
# DEPTH IS CHEAP AND WIDTH IS NOT, measured on this box: one engine step costs
# 0.015 ms while one scored candidate costs a rollout, so a 3 s candidate is
# 0.25 ms and a 6 s candidate 0.44 ms. 10 s therefore lands near 0.7 ms per
# candidate, ~8 ms per decision at K~11 -- against a ~22 s wall-clock episode
# per env at num_envs=8, i.e. a few percent. shipping.py reached the same
# conclusion for the neural search from the other direction.
#
# THE 12-SECOND CAVEAT STILL APPLIES AND IS WHY THIS STOPS AT 10 s.
# `rollout_stats` rolls forward with both sides no-oping, and CLAUDE.md records
# that past ~12 s of that a rollout "stops resembling the game" -- the neural
# search measured horizon 20 WORSE than horizon 12 for exactly this reason
# (0.875 vs 0.963). 100 ticks sits inside the validated regime; going further
# would buy a longer simulation of a fiction.
#
# epsilon is the probability of substituting a uniformly random LEGAL action
# (no-op included) for the argmax, and falls to zero at the top so the final
# rung is fully deterministic given its profile.
# `max_combos` is the THIRD competence axis, added 2026-08-20 with multi-card
# planning. It is still competence and never economy: it is how many two-card
# SEQUENCES the bot is allowed to simulate per decision. Zero on the two
# shortest rungs is not a policy choice -- their horizons (0 and 10 ticks)
# cannot reach the follow-up at +10 ticks, so a combo there would be scored on
# a rollout that never plays half of it (COMBO_MIN_HORIZON_TICKS).
# `reactive` is the FOURTH competence axis (2026-08-21): does the rollout
# opponent answer, or stand still? Off on the two shortest rungs -- they are
# structurally inert (stage 0 takes `_rules_only` and never rolls out; stage
# 10 ticks is one chunk, which is the counter's own delay) AND it states the
# cold-start intent: phase 1's teacher models the RL AGENT, which at episode 0
# cannot defend, so assuming a competent answer there would price every attack
# as punished by an opponent who would not punish it.
# ONE KNOB PER RUNG (2026-09-03). The six-rung version of this table moved
# THREE and FOUR axes at once and called the bundle a "stage": 2 -> 3 was
# horizon 30 -> 50 AND epsilon 0.10 -> 0.05 AND max_combos 2 -> 3, and 3 -> 4
# added k_cells on top. So "advance one stage" was never a small step, and the
# 2026-08-28 run paid for it -- 23,040 consecutive episodes (ep 9,640 ->
# 32,680) at stage 3, win rate oscillating 0.15-0.60 against a 0.80 gate, never
# once advancing. That is the entire measured cost of this table's granularity.
#
# Eleven rungs now, each moving ONE axis (rung 3 moves k_cells and combos
# together only because `max_combos` is inert below COMBO_MIN_HORIZON_TICKS and
# would otherwise be a knob that silently does nothing). The horizon axis is
# 0/1/2/2/3/4/5/6/7/8.5/10 s -- the gradual progression the old table skipped.
#
# THE OLD STAGES ARE ALL STILL HERE, at rungs 0, 1, 4, 6, 8, 10, and they are
# reachable by horizon so a checkpoint can be migrated exactly rather than
# renumbered by guess -- see `remap_legacy_stage`. Nothing about the teacher's
# top end changed: rung 10 is bit-identical to the old stage 5.
TEACHER_STAGES = [
    {"horizon_ticks":   0, "epsilon": 0.30, "k_cells": 1, "max_combos": 0,
     "reactive": False},                            # rules only  (old stage 0)
    {"horizon_ticks":  10, "epsilon": 0.20, "k_cells": 1, "max_combos": 0,
     "reactive": False},                            # 1 s         (old stage 1)
    {"horizon_ticks":  20, "epsilon": 0.15, "k_cells": 1, "max_combos": 0,
     "reactive": False},                            # 2 s
    {"horizon_ticks":  20, "epsilon": 0.12, "k_cells": 2, "max_combos": 2,
     "reactive": False},                            # 2 s + width and combos
    {"horizon_ticks":  30, "epsilon": 0.10, "k_cells": 2, "max_combos": 2,
     "reactive": False},                            # 3 s         (old stage 2)
    {"horizon_ticks":  40, "epsilon": 0.08, "k_cells": 2, "max_combos": 3,
     "reactive": False},                            # 4 s
    {"horizon_ticks":  50, "epsilon": 0.05, "k_cells": 2, "max_combos": 3,
     "reactive": False},                            # 5 s         (old stage 3)
    {"horizon_ticks":  60, "epsilon": 0.04, "k_cells": 3, "max_combos": 3,
     "reactive": False},                            # 6 s + width
    {"horizon_ticks":  70, "epsilon": 0.02, "k_cells": 3, "max_combos": 4,
     "reactive": False},                            # 7 s         (old stage 4)
    {"horizon_ticks":  85, "epsilon": 0.01, "k_cells": 3, "max_combos": 4,
     "reactive": False},                            # 8.5 s
    {"horizon_ticks": 100, "epsilon": 0.00, "k_cells": 3, "max_combos": 4,
     "reactive": True},                             # 10 s        (old stage 5)
]

#: The six-rung table this replaced, by `horizon_ticks`. A saved
#: `curriculum_stage` is an INDEX into whatever table was live when it was
#: written, so resuming a pre-2026-09-03 checkpoint against the eleven-rung
#: table would silently reinterpret it -- a run saved at the old stage 3 (5 s)
#: would come back at the new rung 3 (2 s), a two-rung demotion reported
#: nowhere. Horizon is the axis that is stable across both tables, so the remap
#: is by lookup and not by arithmetic.
_LEGACY_STAGE_HORIZONS = [0, 10, 30, 50, 70, 100]


def remap_legacy_stage(stage):
    """Old six-rung index -> the eleven-rung index with the same horizon.

    Returns `stage` unchanged if it cannot be a legacy index. Called only from
    `CurriculumManager.load_state_dict`, which records whether a checkpoint
    predates the table so this cannot fire twice on the same number.
    """
    stage = int(stage)
    if not 0 <= stage < len(_LEGACY_STAGE_HORIZONS):
        return min(stage, len(TEACHER_STAGES) - 1)
    want = _LEGACY_STAGE_HORIZONS[stage]
    for i, cfg in enumerate(TEACHER_STAGES):
        if cfg["horizon_ticks"] == want:
            return i
    return min(stage, len(TEACHER_STAGES) - 1)


# Own-back-half cells used by `wincon_mode="cycle"` -- see UtilityTeacher's
# docstring. Several offered because `candidates()` filters through the engine's
# legality predicate and a single cell could be refused (tower footprint, back-row
# dead zone), which would silently turn "cycle" into "ban" and confound the
# experiment those two modes exist to separate.
WINCON_DUD_CELLS = [(2.0, 2.0), (15.0, 2.0), (2.0, 4.0), (15.0, 4.0), (9.0, 5.0)]


# --------------------------------------------------------------------------
# multi-card combos (2026-08-20)
# --------------------------------------------------------------------------
# One decision = one `skip_frames` block = 10 ticks = 1.0 s. `gym_wrapper.step`
# hands the teacher exactly ONE `(slot, x, y)` per decision, and -- measured,
# and pinned by `test_a_zero_tick_step_places_nothing` -- a 0-tick
# `step_self_play` places nothing at all, because placement is processed inside
# the tick loop. So a combo is not two cards on one tick; it is a SEQUENCE of
# placements across consecutive decisions, which is also the shape the +448.5
# tower-HP "supported push" was measured at.
COMBO_FOLLOWUP_DELAY_TICKS = 10

# How long the rollout opponent takes to answer an attacking placement.
#
# ONE DECISION, matching `skip_frames`, because that is the fastest a real
# opponent could possibly react -- it cannot answer a card on the tick it
# lands. Answering instantly would make the rollout opponent superhuman, and
# the sweep shows what that costs: a responder that decided EVERY chunk scored
# +0.2217 against +0.2450 for one that decided every third, at 2.1x the price.
# More reaction is not better past the point where it stops resembling a
# player.
COUNTER_DELAY_TICKS = 10

# A rollout may only CHARGE the opponent's answer when it is long enough to
# also SEE the attack's payoff.
#
# The asymmetry: the counter's cost lands at +10 ticks, but a Hog needs ~130 to
# cross ~12 tiles at Fast speed. A short rollout therefore charges the answer in
# full and credits none of the push -- a systematic anti-attack bias that gets
# WORSE the shorter the horizon. Measured against a PASSIVE opponent (which is
# what an episode-0 agent is), 20 seeded openings, share of decisions landing a
# card, and how many openings froze to under 5 plays in 120 decisions:
#
#     horizon    OFF      ON     froze
#        30     12.2%    9.8%     3/20
#        50     11.6%   10.1%     3/20
#        70     12.5%   11.3%     1/20
#       100     11.8%   12.3%     0/20
#
# The +0.1500 win rate was measured at 100, where the bias is gone. Enabling it
# at 30 would ship the zero-gradient failure the 2026-08-19 curriculum pivot
# exists to remove -- a teacher that freezes against a weak opponent.
#
# Same shape as COMBO_MIN_HORIZON_TICKS: never simulate half an interaction and
# score it as though it were whole.
COUNTER_MIN_HORIZON_TICKS = 100

#: The counter models an opponent who ANSWERS. It is switched off while the real
#: opponent has spent no elixir for this many consecutive decisions (~8 s at one
#: decision per second), and back on the moment they play.
#:
#: MEASURED 2026-09-15 (audit 05, BUG 2): unconditional, it froze the top-rung
#: teacher against a passive opponent in 30 of 116 matches -- a banked 10-elixir
#: opponent can always afford the imagined answer, so every attack scored
#: negative and the teacher held with a full bar. Rung 10 is where the curriculum
#: ends, and "the agent banks elixir and holds" is exactly the state it froze in.
#: Eight decisions is long enough that ordinary tempo (a player waiting a few
#: seconds for elixir) keeps the counter on, and short enough that a genuinely
#: passive opponent is punished within one push.
COUNTER_PASSIVE_DECISIONS = 8

# THE GAP IS A SEARCHED AXIS, not a constant, and that is the change that made
# combos reachable at all.
#
# Measured 2026-08-20, teacher vs teacher at stage 5 over ~2,400 decisions: the
# bar's mean is 1.79 and its p90 is 3.30, and a pair was affordable in exactly
# TWO states, both of them the 5.00 opening. At a one-second gap both cards
# have to be affordable at once (Skeletons + Hog needs 4.65), which essentially
# never happens. The cost is really paid ACROSS the gap, so three seconds of
# regeneration is worth 1.05 elixir and five seconds 1.75 -- which moves the
# same pair to 3.95 and 3.25, i.e. from never to sometimes.
#
# And the longer gap is also the BETTER PLAY, which is what makes this a fix
# rather than a loophole. The escort has to eat its own 1.0 s of deploy time
# before it can move at all, so after one second it is barely half a tile ahead
# of the win condition; the +448.5 tower-HP result is for a tank one to two
# tiles in front, which is three to five seconds of walking.
#
# Which gap is right in a given state is exactly the sort of question this
# module answers by rolling it forward rather than by arguing, so all three are
# offered and the simulator ranks them -- the same contract `_cells_for` holds
# for placement, extended to timing.
#
# AND THE RANKING IS TACTICAL, NOT AN ARTIFACT, which was worth checking rather
# than assuming. In real play the 5 s gap is chosen 16 times out of 19, and the
# obvious suspicion is a terminal-evaluation bias: `positional_advantage` is
# read at the END of the horizon, so a card placed at t=50 is fresher there than
# one placed at t=10 and might score higher for no tactical reason. Measured
# with the bar pinned at 8.0 so that all three gaps are affordable and ONLY the
# gap varies (n=21 states, same pair, same cells):
#
#     gap 10 (1 s)   mean score +5.281   median +4.524
#     gap 30 (3 s)   mean score +2.529   median +2.039
#     gap 50 (5 s)   mean score +1.761   median +1.555
#
# The preference is monotone toward the TIGHT escort, which is what the +448.5
# "tank one decision ahead" result says it should be. The 5 s gap dominates real
# play purely because at a bar whose p90 is 3.30 it is the only one that can be
# paid for. That is also why the gaps are offered shortest-first.
COMBO_FOLLOWUP_DELAYS = (10, 30, 50)

# A combo may only be PROPOSED when the rollout is long enough to actually
# simulate its second card. Below this the rollout would charge both costs and
# credit one card's value -- strictly worse than not proposing it at all, and
# invisible, because the symptom is "the teacher never escorts". Applied
# per-candidate against its own gap, so a short rung simply sees fewer gaps.
COMBO_MIN_HORIZON_TICKS = 20

# Elixir that regenerates during the one decision between the two cards.
# Derived from tactics' ELIXIR_REGEN_RATE (itself named against ClashEnv.h)
# rather than re-stating 0.35 -- measured 0.34999847 per 10 ticks.
# Elixir regenerated across one decision. Kept as the named unit the affordability
# arithmetic below is expressed in -- `_pairs` scales it by each candidate's own
# gap rather than assuming one decision.
ELIXIR_PER_DECISION = tactics.ELIXIR_REGEN_RATE * COMBO_FOLLOWUP_DELAY_TICKS

# THE RESERVE THAT PROTECTS A COMMITTED PLAN, in elixir-equivalents like every
# other weight here.
#
# Once the bot has paid for the first half of a combo, the second half is only
# a plan if the money for it is still there when the gap closes. This charges
# any OTHER spend that would leave the follow-up unaffordable. It is bounded by
# construction to the one to five decisions a plan is actually live, which is
# what makes it work where the first attempt did not.
#
# A FLAT SAVINGS CHARGE WAS TRIED FIRST AND IS MEASURED DEAD, recorded here so
# it is not re-proposed: charging every marginal spend while a push was within
# six decisions of affordable moved the bar from 1.79 to 1.73 across reserves
# of 0.0 / 1.5 / 3.0 / 5.0 -- i.e. not at all, and if anything the wrong way. A
# per-decision charge cannot manufacture multi-second saving when the bot has
# many attractive cheap plays and the defence exemption keeps firing. Same
# principle CLAUDE.md already records for back-row structure penalties: a
# penalty cannot move a distribution with no mass to move.
#
# 1.5 is set to outbid the thing it must actually suppress -- a 1-cost cycle
# card whose entire marginal value is positional, which at w_pos=20 scores about
# 1.2, against a measured median of 1.37 for the spends in question. It is
# deliberately NOT set from the +448.5 tower HP a push is worth (4.5
# elixir-equivalents at w_twr): that is the value of EXECUTING the push, not of
# protecting it for one more second. Pass `combo_reserve=0.0` to disable.
COMBO_RESERVE = 1.5

#: Every family `_legal_combos` can emit, in the order it offers them. Named
#: here so one can be ABLATED by configuration rather than by editing source --
#: the win-rate A/B came back with two of three runs pointing negative, and the
#: only honest way to find out which family is responsible is to remove one and
#: re-measure with the arms otherwise byte-identical.
COMBO_FAMILIES = ("supported_push", "counter_push", "defensive_stack",
                  "cheap_defence", "spell_then_push", "push_then_spell")


class PlacementStep:
    """One card going down at one cell, `delay_ticks` after the plan starts."""

    __slots__ = ("slot", "card_id", "x", "y", "delay_ticks")

    def __init__(self, slot, card_id, x, y, delay_ticks=0):
        self.slot = int(slot)
        self.card_id = int(card_id)
        self.x = float(x)
        self.y = float(y)
        self.delay_ticks = int(delay_ticks)

    def __repr__(self):
        return (f"Step(slot={self.slot}, card={self.card_id}, "
                f"x={self.x:.1f}, y={self.y:.1f}, +{self.delay_ticks}t)")


class Candidate:
    """A SEQUENCE of placements the teacher is willing to simulate.

    Was one `(slot, cell)` until 2026-08-20. `.slot/.x/.y/.card_id` survive and
    now mean THE FIRST STEP, because that is the placement `act()` returns this
    decision and every caller downstream unpacks exactly three values.

    The empty sequence is the no-op, which is always present and always scores
    exactly 0 -- it IS the baseline every other score is marginal against.
    """

    __slots__ = ("steps", "role", "kind")

    def __init__(self, steps, role, kind):
        self.steps = tuple(steps)
        self.role = role
        self.kind = kind

    @classmethod
    def single(cls, slot, card_id, x, y, role, kind="single"):
        return cls([PlacementStep(slot, card_id, x, y, 0)], role, kind)

    @classmethod
    def combo(cls, first, second, kind, role="combo"):
        return cls([first, second], role, kind)

    # -- the first step, which is what gets played THIS decision -----------
    @property
    def slot(self):
        return self.steps[0].slot if self.steps else HAND_SIZE

    @property
    def card_id(self):
        return self.steps[0].card_id if self.steps else -1

    @property
    def x(self):
        return self.steps[0].x if self.steps else 0.0

    @property
    def y(self):
        return self.steps[0].y if self.steps else 0.0

    # -- the sequence ------------------------------------------------------
    @property
    def placements(self):
        """[(slot, x, y), ...] -- the action representation, in order."""
        return [(st.slot, st.x, st.y) for st in self.steps]

    @property
    def cards(self):
        return [st.card_id for st in self.steps]

    @property
    def is_combo(self):
        return len(self.steps) > 1

    @property
    def max_delay(self):
        return max((st.delay_ticks for st in self.steps), default=0)

    @property
    def total_cost(self):
        return float(sum(E.get_card_info(c)["cost"] for c in self.cards))

    def __repr__(self):
        if not self.steps:
            return "Candidate(no-op)"
        body = " -> ".join(f"{st.card_id}@({st.x:.0f},{st.y:.0f})"
                           for st in self.steps)
        return f"Candidate[{self.kind}]({body})"


NOOP = Candidate([], "noop", "noop")


def ally_hp_map(obs):
    """Ally troop HP per cell in absolute HP -- the mirror of
    `tactics.enemy_hp_map`, which only covers channels 4-6."""
    sp = tactics.spatial(obs)
    hp = sp[list(CH_ALLY_TROOP)].sum(axis=0) * MAX_TROOP_HP
    count = np.maximum(1.0, sp[CH_ALLY_COUNT] * tactics.MAX_CELL_UNITS)
    return hp * count


_ROW_WEIGHT = np.arange(BOARD_H, dtype=np.float32) / float(BOARD_H - 1)


def positional_advantage(obs):
    """Board control, in HP-fractions weighted by progress up the board.

    THE TERM THAT STOPS THE TEACHER TURTLING. Realized damage inside a 3-6 s
    rollout cannot see a win condition: a Hog placed at the bridge has ~12 tiles
    to cross at Fast speed, i.e. ~13 s, so on damage alone every wincon play
    scores its cost and nothing else, and the argmax is "never attack". This is
    the hand-written stand-in for the value bootstrap `search_ab_test` gets from
    the critic.

    In the observer's own frame HIGH y is toward the enemy towers, so ally mass
    is weighted by y and enemy mass by its mirror. Normalised by MAX_TROOP_HP so
    the scale is "how many full-HP troops, how far forward".
    """
    ally = ally_hp_map(obs) / MAX_TROOP_HP
    enemy = tactics.enemy_hp_map(obs) / MAX_TROOP_HP
    return float((ally.sum(axis=1) * _ROW_WEIGHT).sum()
                 - (enemy.sum(axis=1) * _ROW_WEIGHT[::-1]).sum())


class UtilityTeacher:
    """A deterministic sparring partner strong enough to make attacking pay.

    Plays either side. `act(env, obs_own)` returns `(slot, x, y)` in the
    actor's OWN frame, ready to hand straight to `step_self_play` -- team 0's
    coordinates in slots 0-2, team 1's in slots 3-5.

    `wincon_mode` exists for `prove_environment.py` and nothing else:

      "attack"  the real rule -- the win condition goes to a bridge.
      "cycle"   it is still drawn, still played, still costs its 4 elixir and
                still rotates the hand, but it is placed in our own back half
                where it is a dud. CLAUDE.md measures that difference directly:
                bridge 535.6 enemy tower damage against ~3 for a back-row cell.
      "ban"     never played at all.

    "cycle" is the arm the falsifier actually uses. "ban" looks like the obvious
    control and is CONFOUNDED: a card that is never played never leaves the hand,
    so banning it also permanently clogs a hand slot and costs elixir efficiency
    for a reason that has nothing to do with the economy under test. "cycle"
    holds spend, cycle and hand occupancy identical across arms and varies only
    WHERE the card lands, which is the one thing the hypothesis is about.
    """

    def __init__(self, deck, team, profile=None, horizon_ticks=30, k_cells=2,
                 epsilon=0.0, seed=None, wincon_mode="attack", max_combos=3,
                 combo_reserve=None, reactive_rollout=True):
        self.deck = list(deck)
        self.team = int(team)
        #: Does the rollout opponent ANSWER, or stand still? See
        #: `counter_schedule` for the measurement and for why the counter is
        #: open-loop. `False` is action-identical to the pre-2026-08-21 teacher.
        self.reactive_rollout = bool(reactive_rollout)
        self.wincon_mode = wincon_mode
        self.roles = card_roles(self.deck)
        self.wincon_id = next((c for c, r in self.roles.items() if r == "wincon"),
                              None)
        self.tank_id = tank_id(self.deck)
        self.horizon_ticks = int(horizon_ticks)
        self.k_cells = int(k_cells)
        self.max_combos = int(max_combos)
        self.combo_reserve = float(COMBO_RESERVE if combo_reserve is None
                                   else combo_reserve)
        self.combo_families = COMBO_FAMILIES
        self.epsilon = float(epsilon)
        # The second half of a chosen combo, carried to the NEXT decision. See
        # `commit` for why it is offered there rather than executed there.
        self.pending = None
        # Ticks until the pending step is DUE. A gap that was scored at three
        # seconds has to be played at three seconds, or the plan that ran in
        # simulation is not the plan that reaches the board.
        self.pending_ticks = 0
        # Diagnostic only: which candidate KIND the last `act` settled on.
        # Lives here rather than in the harness because a combo's first step is
        # indistinguishable from the same card played alone once it has been
        # reduced to a returned (slot, x, y).
        self.last_kind = "noop"
        # A profile is either a NAME in PROFILES or an explicit weight set.
        # The mapping form exists for sweeps: `PROFILES` has three entries and
        # the interesting weights are usually between or below them. Copied
        # rather than referenced, because a sweep builds both sides of a match
        # from one dict and a teacher that mutated it would corrupt the arm.
        self._fixed_profile = (dict(profile) if isinstance(profile, dict)
                               else profile)
        self.rng = np.random.default_rng(seed)
        self.cycle = CycleTracker(self.deck)
        self.profile = self._resolve_profile(profile)
        self.lane_bias = 0
        # A play must beat holding by more than this -- the bot's ONE economy
        # control, and until 2026-08-21 it was set to a value that could not
        # perform it.
        #
        # It was 0.05, against a MEASURED median of 1.37 for the marginal cheap
        # plays it exists to stop (the plays actually chosen while the win
        # condition sat in hand and nothing threatened). 27x too low, so it
        # never bound, and the bot spent to ~1.8 elixir continuously -- which
        # is why a 5-6 elixir escorted push was affordable on 2 decisions in
        # ~2,400 and the combo generator had nothing to buy.
        #
        # 3.0 selected by sweep and CONFIRMED on a fresh independent run, both
        # paired on shared openings within one process (the engine's shuffle is
        # unseeded, so cross-run levels are not comparable -- see
        # UPSTREAM_REQUESTS item 7). Head to head against the old profile,
        # sides swapped:
        #
        #   margin  vs old   elixir p90   overflow   combos, % of plays
        #   0.05    0.500       3.55        0.1%          2.2%
        #   2.0     0.812       5.60        1.3%          6.9%
        #   3.0     0.969       7.95        4.3%         14.4%
        #   4.0     0.969       9.46       12.9%         18.2%
        #
        # 4.0 is not better -- identical strength, three times the wasted
        # income. 3.0 is the knee.
        #
        # NOTE WHAT THIS IS NOT. Lowering `w_pos` was the obvious lever and is
        # measured WRONG: across 20 -> 8 it raises elixir (1.83 -> 2.67) but
        # combo share goes 1.1% -> 0.0/0.0/0.2/0.0/0.2%. `w_pos` prunes plays by
        # HP-per-elixir, and an escorted push (388 HP/elixir) sits BELOW a naked
        # Hog (424), so it kills the combo before the cheap cards it was meant
        # to replace. The naked-unit reward and the combo reward are the same
        # term and cannot be separated by that weight.
        #
        # GAMEPLAY-AFFECTING for phase 1: every win rate earned against
        # `teacher@stage N` before this date describes a bot that dumped.
        self.play_margin = 3.0

    # -- lifecycle ---------------------------------------------------------
    def reset(self, rng=None):
        """New match. Redraws the profile and lane bias unless a profile was
        pinned at construction -- a FULLY deterministic opponent is memorizable,
        which is the single-counter-line version of the echo chamber this whole
        pivot exists to avoid."""
        if rng is not None:
            self.rng = rng
        self.cycle.reset()
        self.pending = None
        self.pending_ticks = 0
        #: Opponent reactivity, read from the engine each decision -- see
        #: COUNTER_PASSIVE_DECISIONS.
        self._opp_spent_seen = None
        self._opp_idle_decisions = 0
        if self._fixed_profile:
            self.profile = self._resolve_profile(self._fixed_profile)
        else:
            names = list(PROFILES)
            self.profile = PROFILES[names[int(self.rng.integers(len(names)))]]
        self.lane_bias = int(self.rng.integers(2))

    @staticmethod
    def _resolve_profile(profile):
        """A name, an explicit weight set, or None -> "balanced".

        Returns a COPY in every case. `reset()` re-resolves each match, and
        without the copy a teacher that touched `self.profile` would edit the
        module-level `PROFILES` table for the whole process -- which in a sweep
        would silently change every later arm.
        """
        if isinstance(profile, dict):
            return dict(profile)
        return dict(PROFILES[profile] if profile else PROFILES["balanced"])

    def set_deck(self, deck):
        """Point the teacher at a different deck.

        NEEDED BY PHASE 1'S `random_opponent`, which re-rolls the opponent deck
        every few hundred episodes. Without it the role table and the cycle
        tracker keep describing the deck the teacher was CONSTRUCTED with, so a
        random deck's win condition would be treated as a plain melee troop and
        the cycle term would count cards that are not in the deck -- a silent
        degradation that looks like "the teacher is weak against random decks".

        `card_roles` is memoized per deck, so re-rolling a deck the teacher has
        already seen costs nothing.
        """
        deck = list(deck)
        if deck == self.deck:
            return
        self.deck = deck
        self.roles = card_roles(self.deck)
        self.wincon_id = next((c for c, r in self.roles.items() if r == "wincon"),
                              None)
        self.tank_id = tank_id(self.deck)
        self.cycle = CycleTracker(self.deck)
        self.cycle.reset()
        self.pending = None
        self.pending_ticks = 0

    def set_stage(self, stage):
        """Apply one rung of TEACHER_STAGES. Difficulty is competence only --
        this never touches elixir."""
        cfg = TEACHER_STAGES[int(np.clip(stage, 0, len(TEACHER_STAGES) - 1))]
        self.horizon_ticks = cfg["horizon_ticks"]
        self.epsilon = cfg["epsilon"]
        self.k_cells = cfg["k_cells"]
        self.max_combos = cfg["max_combos"]
        self.reactive_rollout = cfg["reactive"]

    # -- frames ------------------------------------------------------------
    def to_absolute_y(self, y_own):
        """Own-frame y -> the ABSOLUTE y `is_valid_placement` expects.

        THE conversion point. `step_self_play` mirrors team 1's y itself, but
        `is_valid_placement` does not mirror anything, so a team-1 bot that
        skips this checks legality against team 0's half and every candidate is
        rejected -- silently, as a bot that never plays."""
        return float(y_own) if self.team == 0 else MIRROR_Y - float(y_own)

    # -- candidate generation ---------------------------------------------
    def candidates(self, env, obs_own, elixir=None):
        """Every (slot, cell) worth simulating this step, plus the no-op.

        Cells come from `tactics.py`'s engine-validated rules. Everything is
        filtered through the engine's OWN legality predicate rather than a
        second copy of the board geometry -- the failure mode that once made
        58.7% of card choices silent no-ops.
        """
        hand = list(env.get_hand_for_team(self.team))
        if elixir is None:
            elixir = env.get_elixir_for_team(self.team)
        elixir = float(elixir)

        out = [NOOP]
        for slot, cid in enumerate(hand):
            if slot >= HAND_SIZE:
                break
            info = E.get_card_info(cid)
            if info["cost"] > elixir + 1e-6:
                continue
            role = self.roles.get(cid, "melee")
            if role == "wincon" and self.wincon_mode == "ban":
                continue
            for (x, y) in self._cells_for(role, cid, obs_own):
                xi, yi = float(int(x)), float(int(y))
                if env.is_valid_placement(cid, xi, self.to_absolute_y(yi), self.team):
                    out.append(Candidate.single(slot, cid, xi, yi, role))

        # The second half of a plan made last decision, offered as an ordinary
        # candidate. It is RE-SCORED here rather than executed blindly -- see
        # `commit`.
        follow = self._followup_candidate(env, hand, elixir)
        if follow is not None:
            out.append(follow)

        out.extend(self._legal_combos(env, obs_own, hand, elixir))
        return out

    # -- the plan ----------------------------------------------------------
    def commit(self, cand):
        """Record a chosen combo's second step for the next decision.

        WHY THE FOLLOW-UP IS OFFERED AND NOT EXECUTED. A blind commitment would
        place the win condition into whatever the board became one second later,
        which is the "send it alone into a counter-push" mistake with an extra
        step. Re-scoring is not a weaker plan, it is a BETTER one: by the next
        decision the tank is physically on the board, so an ordinary solo
        rollout of the win condition already SEES the escort in front of it. The
        combo's job was to make the tank's own placement look worth making --
        the synergy does not have to be carried forward as a bonus, because the
        simulator can observe it directly. Adding one would double-count it.

        What the plan DOES carry is the exact cell, in the exact lane, which the
        single-card rules would not otherwise propose together.
        """
        if not cand.is_combo:
            # NOT a clear. A plan that is still waiting out its gap must
            # survive the ordinary single plays made while it waits -- that is
            # the whole point of a gap longer than one decision.
            return
        self.pending = cand.steps[1]
        self.pending_ticks = self.pending.delay_ticks

    def tick_plan(self):
        """One decision passes. Returns True while a plan is still WAITING.

        Called BEFORE `candidates`, because a plan committed with a 30-tick gap
        is due three decisions later and the clock has to have advanced before
        the follow-up can be offered. Getting this order wrong delays every plan
        by one decision, which is invisible except as a slightly wrong escort
        distance -- the sort of off-by-one this project has paid for before.
        """
        if self.pending is None:
            return False
        self.pending_ticks -= COMBO_FOLLOWUP_DELAY_TICKS
        return self.pending_ticks > 0

    def _followup_candidate(self, env, hand, elixir):
        """The pending step as a candidate, or None if the world moved on."""
        st = self.pending
        if st is None or self.pending_ticks > 0:
            return None
        if st.slot >= HAND_SIZE or hand[st.slot] != st.card_id:
            return None                       # the card left that slot
        if E.get_card_info(st.card_id)["cost"] > elixir + 1e-6:
            return None                       # we spent it elsewhere
        if not env.is_valid_placement(st.card_id, st.x,
                                      self.to_absolute_y(st.y), self.team):
            return None
        return Candidate.single(st.slot, st.card_id, st.x, st.y,
                                self.roles.get(st.card_id, "melee"),
                                kind="followup")

    # -- combos ------------------------------------------------------------
    def _legal_combos(self, env, obs, hand, elixir):
        """Curated two-card sequences, filtered through the engine's legality.

        WIDTH IS THE EXPENSIVE AXIS. One engine step is 0.015 ms but each extra
        candidate is a whole rollout, so this enumerates a handful of named
        tactics rather than the cross product of pairs x cells. Families are
        taken ROUND-ROBIN under `max_combos`, so a small budget still sees one
        of each rather than two variants of the first.
        """
        if self.max_combos <= 0 or self.horizon_ticks < COMBO_MIN_HORIZON_TICKS:
            return []
        slots = {}
        for slot, cid in enumerate(hand[:HAND_SIZE]):
            slots.setdefault(int(cid), slot)

        families = [getattr(self, f"_combo_{name}")(obs, slots, elixir)
                    for name in self.combo_families]
        out = []
        for row in itertools.zip_longest(*families):
            for c in row:
                if c is None or len(out) >= self.max_combos:
                    continue
                if self._combo_is_legal(env, c):
                    out.append(c)
            if len(out) >= self.max_combos:
                break
        return out

    def _combo_is_legal(self, env, cand):
        """EVERY step, or the whole sequence is dropped.

        A combo whose second step is illegal is worse than no combo: it is
        charged for two cards and plays one, so escorting looks bad for a reason
        that has nothing to do with escorting.
        """
        if cand.steps[0].slot == cand.steps[1].slot:
            # Playing a slot refills it from the queue, so the second step would
            # place whatever arrived, not the card this plan was scored on.
            return False
        return all(env.is_valid_placement(st.card_id, st.x,
                                          self.to_absolute_y(st.y), self.team)
                   for st in cand.steps)

    def _gaps(self):
        """The follow-up gaps this rung can SEE, shortest first.

        A gap is offerable only when the rollout runs past it -- otherwise the
        pair is charged for two cards and simulated with one. Shortest first so
        the round-robin's first pass takes the tightest escort available, which
        is the one the simulator prefers whenever it is affordable.
        """
        return [d for d in COMBO_FOLLOWUP_DELAYS
                if d + COMBO_FOLLOWUP_DELAY_TICKS <= self.horizon_ticks]

    def _pairs(self, slots, first, second, cell1, cell2, kind, elixir):
        """Every affordable gap for one (first -> second) tactic.

        AFFORDABILITY IS PER-GAP and it is the point. The first card is paid
        now; the second is paid `d` ticks later out of what has regenerated by
        then, capped at the engine's own ceiling. That is why a three-second
        gap can buy a pair a one-second gap cannot.
        """
        c1 = float(E.get_card_info(first)["cost"])
        c2 = float(E.get_card_info(second)["cost"])
        out = []
        for d in self._gaps():
            later = min(tactics.MAX_ELIXIR,
                        float(elixir) - c1 + tactics.ELIXIR_REGEN_RATE * d)
            if float(elixir) + 1e-6 < c1 or later + 1e-6 < c2:
                continue
            out.append(Candidate.combo(
                PlacementStep(slots[first], first,
                              float(int(cell1[0])), float(int(cell1[1])), 0),
                PlacementStep(slots[second], second,
                              float(int(cell2[0])), float(int(cell2[1])), d),
                kind=kind))
        return out

    def _combo_supported_push(self, obs, slots, elixir):
        """THE combo the 2026-08-19 deploy-time change made correct.

        Tank first, win condition one to five seconds behind it, same lane --
        the gap is searched, see COMBO_FOLLOWUP_DELAYS. Measured on
        the same engine: a lone commitment is worth -556.3 tower HP marginally
        and a supported one +448.5 [+137.3, +760.1]; escorting inside a punish
        window is worth +650 [+429, +878].

        THE ORDER IS THE WHOLE POINT and it is not symmetric. The tank has to
        eat its own 1.0 s of deploy time BEFORE the win condition arrives, so
        that the tower has something to lock onto when the Hog crosses. Sending
        the win condition first is the naked push the engine now punishes -- so
        that ordering is not offered at all.

        Two variants, ranked by the simulator rather than by argument: the win
        condition on the tank's own cell, and one tile behind it. Which is
        better depends on the relative speeds after deploy, which is exactly the
        kind of question a rollout answers and a comment does not.
        """
        wc, tank = self.wincon_id, self.tank_id
        if (wc is None or tank is None or wc == tank
                or self.wincon_mode != "attack"
                or wc not in slots or tank not in slots):
            return []
        bx, by, _ = tactics.best_hog_cell(obs)
        behind = max(0.0, by - 1.0)
        return (self._pairs(slots, tank, wc, (bx, by), (bx, behind),
                            "supported_push", elixir)
                + self._pairs(slots, tank, wc, (bx, by), (bx, by),
                              "supported_push", elixir))

    def _combo_counter_push(self, obs, slots, elixir):
        """The CHEAP escort, and the reason it exists is arithmetic.

        Ice Golem + Hog costs 6 and this bot's bar reaches 6 on 0.3% of
        decisions. A 1-cost body in front of the win condition costs 5, which it
        does reach. The escort is worse at tanking and the placement is
        otherwise identical, so this is strictly a price/quality pair and the
        simulator is the right thing to choose between them -- which is why both
        are offered rather than one being picked here.

        Skipped when the cheapest body IS the tank, which would just duplicate
        `_combo_supported_push` and spend a rollout on it.
        """
        wc = self.wincon_id
        if (wc is None or self.wincon_mode != "attack" or wc not in slots):
            return []
        bodies = [c for c in slots
                  if self.roles.get(c) in ("melee", "ranged") and c != wc]
        if not bodies:
            return []
        escort = min(bodies, key=lambda c: E.get_card_info(c)["cost"])
        if escort == self.tank_id:
            return []
        bx, by, _ = tactics.best_hog_cell(obs)
        return self._pairs(slots, escort, wc, (bx, by), (bx, max(0.0, by - 1.0)),
                           "counter_push", elixir)

    def _combo_defensive_stack(self, obs, slots, elixir):
        """Building to hold the push, a body to kill what it holds.

        The 2.6 defensive pair. The building lands FIRST because it is the piece
        whose value comes from being there early -- it has to survive its deploy
        time and start pulling before the support arrives. The second variant is
        the centre pull (in front of our own King, between the Princess towers),
        which is where a Cannon drags a lane-committed win condition off its
        path; `_cells_for` already offers that cell to the Cannon alone.
        """
        if tactics.threat_level(obs) <= 0.0:
            return []
        buildings = [c for c in slots if self.roles.get(c) == "building"]
        bodies = [c for c in slots if self.roles.get(c) in ("melee", "ranged")]
        if not buildings or not bodies:
            return []
        b = min(buildings, key=lambda c: E.get_card_info(c)["cost"])
        # The cheapest body: playing anything advances the cycle, and a cheap
        # one is the efficient way to pull the win condition closer.
        body = min(bodies, key=lambda c: E.get_card_info(c)["cost"])
        tx, ty = self._deepest_threat(obs)
        if tx is None:
            return []
        body_cell = (tx, min(ty, float(tactics.BRIDGE_ROW)))
        bx, by, _ = tactics.best_building_cell(obs)
        return (self._pairs(slots, b, body, (bx, by), body_cell,
                            "defensive_stack", elixir)
                + self._pairs(slots, b, body, (tactics.OWN_KING[0], 11.0),
                              body_cell, "defensive_stack", elixir))

    def _combo_cheap_defence(self, obs, slots, elixir):
        """Two cheap bodies onto the same threat, one gap apart.

        THE FAMILY THAT EXISTS BECAUSE OF PRICE. Measured 2026-08-20: combos are
        chosen on 0.28% of decisions and the bar's p90 is 3.30, so every family
        that needs 5-6 elixir is priced out of nearly every state. Ice Spirit +
        Skeletons is TWO elixir and is a real 2.6 defensive pair -- chip and
        stall, then bodies -- so it fires in the states the expensive families
        cannot reach. Unlike `_combo_defensive_stack` it needs no building.

        The cheapest body lands first: its job is to arrive before the threat
        does, and it is the one whose 1.0 s of deploy time is most affordable to
        pay early. The second lands a tile back so a splash answer cannot catch
        both at once.

        The win condition is excluded -- it is not a defensive body, and sending
        it into an incoming push is the trade `hog_should_commit` already
        refuses.
        """
        if tactics.threat_level(obs) <= 0.0:
            return []
        bodies = [c for c in slots
                  if self.roles.get(c) in ("melee", "ranged")
                  and c != self.wincon_id]
        if len(bodies) < 2:
            return []
        bodies.sort(key=lambda c: (E.get_card_info(c)["cost"], c))
        first, second = bodies[0], bodies[1]
        tx, ty = self._deepest_threat(obs)
        if tx is None:
            return []
        ty = min(ty, float(tactics.BRIDGE_ROW))
        return self._pairs(slots, first, second, (tx, ty),
                           (tx, max(0.0, ty - 1.0)), "cheap_defence", elixir)

    def _combo_spell_then_push(self, obs, slots, elixir):
        """Clear the lane, then walk into it.

        The spell goes first because its value is realized instantly and the
        push's value depends on what is left standing. Gated on the catch map
        actually catching something -- a spell cast at nothing is 2-4 elixir for
        zero, and forcing Fireball once dropped win rate 97% -> 23%.
        """
        spells = [c for c in slots if self.roles.get(c) == "spell"]
        if not spells:
            return []
        spell = min(spells, key=lambda c: E.get_card_info(c)["cost"])
        catch = spell_catch_for(obs, spell)
        if float(catch.max()) <= 0.0:
            return []
        i = int(np.argmax(catch))
        scell = (float(i % BOARD_W), float(i // BOARD_W))
        pusher = None
        if (self.wincon_mode == "attack" and self.wincon_id in slots):
            pusher = self.wincon_id
        elif self.tank_id in slots:
            pusher = self.tank_id
        if pusher is None:
            return []
        bx, by, _ = tactics.best_hog_cell(obs)
        return self._pairs(slots, spell, pusher, scell, (bx, by),
                           "spell_then_push", elixir)

    def _combo_push_then_spell(self, obs, slots, elixir):
        """The win condition, then the spell that answers its answer.

        This is the "Hog + predictive Log" shape, and it is stated here with its
        own limitation because the limitation is structural rather than a bug.
        `rollout_stats` rolls forward with BOTH SIDES NO-OPING, so the defender
        never plays the Skeletons the Log is meant to pre-empt, and a genuinely
        PREDICTIVE cast therefore scores zero value in simulation and can never
        win the argmax. What is scoreable, and what this proposes, is a spell
        aimed at defenders ALREADY on the board on the lane being attacked.

        Left in rather than dropped: the family costs one rollout, it fires only
        when the catch map is already non-empty, and it is the only shape that
        can support a committed push. If the rollout ever gains an opponent
        model, this is the candidate that starts paying.
        """
        wc = self.wincon_id
        if (wc is None or self.wincon_mode != "attack" or wc not in slots):
            return []
        spells = [c for c in slots if self.roles.get(c) == "spell"]
        if not spells:
            return []
        spell = min(spells, key=lambda c: E.get_card_info(c)["cost"])
        catch = spell_catch_for(obs, spell)
        if float(catch.max()) <= 0.0:
            return []
        i = int(np.argmax(catch))
        scell = (float(i % BOARD_W), float(i // BOARD_W))
        bx, by, _ = tactics.best_hog_cell(obs)
        return self._pairs(slots, wc, spell, (bx, by), scell,
                           "push_then_spell", elixir)

    def _cells_for(self, role, card_id, obs):
        """The rule layer: 1-3 tactically sensible cells for one card."""
        k = max(1, self.k_cells)
        if role == "wincon":
            if self.wincon_mode == "cycle":
                return list(WINCON_DUD_CELLS)
            info = E.get_card_info(card_id)
            # deploy_anywhere FIRST: a Goblin Drill is a building AND
            # deploy-anywhere, and from the siege row it measured 0 tower damage
            # in 300 ticks against 2654 beside the tower -- see siege_building.
            if info["is_building"] and not info.get("deploy_anywhere", False):
                return self._siege_cells(obs)[:k]
            if info["is_spell"]:
                return self._tower_cells(obs)[:k]
            if info.get("deploy_anywhere", False):
                # A Miner's value is that it skips the bridge. Sending it there
                # (best_hog_cell) played it as a slow ground troop.
                return self._beside_tower_cells(card_id, obs)[:k]
            x, y, _ = tactics.best_hog_cell(obs)
            cells = [(x, y)]
            if k > 1:
                other = (tactics.BRIDGE_XS[1] if int(x) == tactics.BRIDGE_XS[0]
                         else tactics.BRIDGE_XS[0])
                cells.append((float(other), float(tactics.BRIDGE_ROW)))
            return cells[:k]

        if role == "spell":
            return self._top_spell_cells(obs, k, card_id)

        if role == "building":
            x, y, _ = tactics.best_building_cell(obs)
            cells = [(x, y)]
            if k > 1:
                # The 2.6 centre-pull: in front of the King, between the two
                # Princess towers, which is where a Cannon drags a lane-committed
                # win condition off its path.
                cells.append((float(tactics.OWN_KING[0]), 11.0))
            return cells[:k]

        # Plain troops: meet the deepest threat as far forward as legal, and
        # (second) support our own most advanced unit. An anti-air card meets a
        # flying building-targeter FIRST when one is coming: it is the threat
        # nothing else in the deck can answer (TODO 00.5).
        cells = []
        tx, ty = self._deepest_threat(obs)
        ax, ay = self._air_siege_threat(obs)
        if ax is not None and card_probes.damages_air(card_id):
            tx, ty = ax, ay
        if tx is not None:
            cells.append((tx, min(ty, float(tactics.BRIDGE_ROW))))
        sx, sy = self._support_cell(obs)
        if sx is not None:
            cells.append((sx, sy))
        if not cells:
            # Nothing to answer and nothing to support. Offer the bridge on this
            # match's biased lane so a cheap card can still open a push -- the
            # no-op baseline is what decides whether that is actually worth it.
            cells.append((float(tactics.BRIDGE_XS[self.lane_bias]),
                          float(tactics.BRIDGE_ROW)))
        return cells[:k]

    def _siege_cells(self, obs):
        """Where a Mortar or an X-Bow has to stand to threaten anything.

        The furthest-forward legal row, which `own_half_max_row` reads off the
        engine. This is not a preference: sweeping all 170 legal cells, 34 damage
        the enemy tower and 136 do exactly ZERO, so a siege building one row too
        far back is worth precisely as much as not playing it.

        Centre first -- an X-Bow at x 7-10 on that row measured 3824 against 2534
        out at the edge, because the centre column reaches BOTH Princess Towers.
        The threatened lane comes second so the ranker can prefer the lane that
        is actually under pressure.
        """
        row = float(own_half_max_row())
        bx, _by, _ = tactics.best_hog_cell(obs)
        cells = [(float(int(EC.BOARD_CENTER_X)), row), (float(int(bx)), row)]
        other = (tactics.BRIDGE_XS[1] if int(bx) == tactics.BRIDGE_XS[0]
                 else tactics.BRIDGE_XS[0])
        cells.append((float(other), row))
        return cells

    def _beside_tower_cells(self, card_id, obs):
        """Legal cells just in front of each enemy Princess, weaker tower first.

        For a deploy-anywhere troop. The tower's own cell is illegal for a BODY
        (its footprint), so this steps toward the river until the engine
        accepts the placement -- the engine decides, not a copied radius.
        """
        out = []
        for x, y in self._tower_cells(obs):
            for dy in (3.0, 4.0, 2.0, 5.0):
                if self.env_valid(card_id, x, y - dy):
                    out.append((x, y - dy))
                    break
        return out or self._tower_cells(obs)

    def env_valid(self, card_id, x, y):
        """`is_valid_placement` for an OWN-FRAME cell, on a shared probe board.

        Own frame is team 0's frame (`_to_board` mirrors afterwards), so this
        asks as team 0. Legality is mirror-symmetric -- audit 06 compared all
        132 cards x 612 cells for both teams and found zero asymmetries -- and
        board-state independent for the enemy half.
        """
        return _probe_env_cached().is_valid_placement(card_id, float(x), float(y), 0)

    def _tower_cells(self, obs):
        """Where a spawning spell has to land: ON an enemy Princess Tower.

        Goblin Barrel measured 1320 there against 600 thrown into our own half.
        The old path sent it through `_top_spell_cells`, which aims at ENEMY
        TROOP CLUSTERS -- so on a quiet board it proposed the argmax of an
        all-zero map, cell (0, 0), and the barrel was never worth playing.

        The weaker tower first: a spell win condition wins by finishing one
        tower, not by spreading chip across two.
        """
        y = float(EC.princess_y(1))
        lanes = [(float(int(EC.LEFT_LANE_X)), y, 1), (float(int(EC.RIGHT_LANE_X)), y, 2)]
        lanes.sort(key=lambda c: self._enemy_tower_fraction(obs, c[2]))
        return [(x, yy) for x, yy, _slot in lanes]

    def _enemy_tower_fraction(self, obs, slot):
        """Enemy Princess Tower HP, normalised, straight off the observation.

        Extra scalars 6-8 are enemy king / left / right -- the same tail
        `gym_wrapper` re-scales for the lethal-spell term. Read forward from
        EXTRA_SCALARS_START, never backward from the end: the layout grows by
        appending and a backward offset is a scheduled defect.
        """
        return float(obs[EC.EXTRA_SCALARS_START + 6 + slot])

    def _top_spell_cells(self, obs, k, card_id):
        """Top-k cells of THIS spell's catch map with non-maximum suppression, so
        the second candidate is a different DECISION and not the same blast
        shifted one tile. Map and suppression disc are the card's own -- see
        `spell_geometry`."""
        radius, _damage = spell_geometry(card_id)
        m = spell_catch_for(obs, card_id).copy()
        cells = []
        for _ in range(k):
            i = int(np.argmax(m))
            v = float(m.reshape(-1)[i])
            x, y = float(i % BOARD_W), float(i // BOARD_W)
            cells.append((x, y))
            if v <= 0.0:
                break
            for dy, dx in tactics._disc_offsets(radius):
                ay, ax = int(y) + dy, int(x) + dx
                if 0 <= ay < BOARD_H and 0 <= ax < BOARD_W:
                    m[ay, ax] = 0.0
        return cells

    def _air_siege_threat(self, obs):
        """(x, y) of the deepest flying BUILDING-targeter, or (None, None).

        The one threat a ground-only card cannot touch -- see
        `tactics.air_siege_map`. Scans the whole board, like `_deepest_threat`.
        """
        m = tactics.air_siege_map(obs)
        if m.sum() <= 0.0:
            return None, None
        ys, xs = np.nonzero(m)
        deepest = int(np.argmin(ys))
        return float(xs[deepest]), float(ys[deepest])

    def _deepest_threat(self, obs):
        """(x, y) of the enemy mass furthest into our half, or (None, None).

        Scans the WHOLE board, not just our own half -- `train_selfplay`'s
        scripted bots measured that reaction latency, not sampling weight, was
        what made them lose: noticing a push only once it has crossed is already
        too late at post-2026-08-07 speeds.
        """
        hp = tactics.enemy_hp_map(obs)
        if hp.sum() <= 0.0:
            return None, None
        ys, xs = np.nonzero(hp)
        deepest = int(np.argmin(ys))       # low y = closest to OUR towers
        return float(xs[deepest]), float(ys[deepest])

    def _support_cell(self, obs):
        """(x, y) just behind our most advanced unit, or (None, None)."""
        hp = ally_hp_map(obs)
        if hp.sum() <= 0.0:
            return None, None
        ys, xs = np.nonzero(hp)
        lead = int(np.argmax(ys))          # high y = furthest toward the enemy
        y = float(np.clip(ys[lead] - 2.0, 0.0, float(tactics.BRIDGE_ROW)))
        return float(xs[lead]), y

    # -- scoring -----------------------------------------------------------
    def effective_play_margin(self, elixir):
        """`play_margin`, tapered to zero as the bar approaches overflow.

        A FIXED bar is wrong and the measurement that shows it is the one where
        the opponent does nothing. Against the C++ heuristic a high bar looks
        excellent (margin 3.0 beat the shipped profile 0.969 head to head)
        because an active opponent constantly creates scoreable situations.
        Against a PASSIVE opponent nothing clears a fixed 3.0, so the bot froze:
        14 plays across 6 matches, elixir pinned at 9.56, tower damage 9143 ->
        4063, and a dropped match it should win trivially.

        An episode-0 agent IS passive. A fixed high bar would therefore hand
        phase 1 exactly the zero-gradient environment the 2026-08-19 curriculum
        pivot exists to remove -- while looking strong on every benchmark that
        uses an active opponent.

        The taper has the same SHAPE as `score`'s overflow relief but starts at
        MARGIN_TAPER_START, not at ELIXIR_OVERFLOW_AT. Anchoring it on the
        overflow line left the bar at its full height for 0-9 elixir -- i.e.
        across almost the whole range -- which reproduced the freeze this
        docstring describes rather than removing it. See MARGIN_TAPER_START for
        the measurement.

        `score`'s own `overflow_relief` is deliberately NOT changed with it:
        that one is about the COST charge being refunded when income is
        genuinely being thrown away, which really is a 9.0 question.
        """
        relief = max(0.0, float(elixir) - MARGIN_TAPER_START) / (
            10.0 - MARGIN_TAPER_START)
        return float(self.play_margin) * (1.0 - min(1.0, relief))

    def margin_for(self, cand, elixir):
        """The bar THIS candidate has to clear.

        Zero for a plan's due second half. `play_margin` exists to stop the bot
        DUMPING -- spending on a marginal play when holding was free -- and for
        a follow-up holding is not free: the first card is already on the board
        and already paid for, so declining does not bank the elixir, it wastes
        the commitment.

        Narrow by construction: the follow-up still has to be the ARGMAX over
        every other candidate, so all this skips is the floor whose premise is
        false. Parallel to `plan_reserve_penalty`'s exemption for the same step.
        """
        if cand.kind == "followup":
            return 0.0
        return self.effective_play_margin(elixir)

    def rollout_ticks(self):
        """How long every rollout in one decision runs.

        ONE number for the whole decision, candidates and baseline alike. If a
        combo were rolled longer than the no-op it is scored against, the
        difference would carry the extra time as well as the extra cards, and
        every combo would look good for the wrong reason.
        """
        return max(COMBO_FOLLOWUP_DELAY_TICKS, self.horizon_ticks)

    def execute_steps(self, s, cand, ticks):
        """Play `cand`'s scheduled placements while advancing `s` by `ticks`.

        Chunked at 10 ticks, matching `skip_frames` -- the granularity the
        teacher is actually driven at, so a plan that simulates well is a plan
        it can really execute. Chunk boundaries are pulled in to land exactly on
        any scheduled offset, so a follow-up cannot be rounded into the wrong
        second.
        """
        sched = {}
        for st in cand.steps:
            sched.setdefault(int(st.delay_ticks), st)
        counters = self.counter_schedule(cand)
        t = 0
        while t < ticks and not s.is_game_over():
            st = sched.get(t)
            slot = st.slot if st is not None else -1
            x = st.x if st is not None else 0.0
            y = st.y if st is not None else 0.0
            nxt = min(ticks, t + COMBO_FOLLOWUP_DELAY_TICKS)
            for d in sched:
                if t < d < nxt:
                    nxt = d
            for d in counters:
                if t < d < nxt:
                    nxt = d
            oslot, ox, oy = self.counter_action(s, counters.get(t))
            # step_self_play_FAST: same advance, without building the two
            # 13,606-float observations this loop has always thrown away.
            # Measured at teacher stage 5: ~9.7 chunks per rollout x ~6.5
            # rollouts per decision = 1,000,152 vectors built per 48 episodes
            # against 51,566 read. The ONE observation a rollout does read is
            # `rollout_stats`'s own get_observation_for_team, below.
            if self.team == 0:
                s.step_self_play_fast(slot, x, y, oslot, ox, oy, nxt - t)
            else:
                s.step_self_play_fast(oslot, ox, oy, slot, x, y, nxt - t)
            t = nxt
        return s

    # -- the reacting opponent ---------------------------------------------
    def counter_schedule(self, cand):
        """{tick: (x, y)} -- when and where the rollout opponent answers.

        OPEN-LOOP BY MEASUREMENT, not by laziness. The answer is derived once,
        here, from OUR OWN candidate; nothing is read from the board inside the
        rollout. Four responders were compared as paired win rate over 150
        seeded openings (sides swapped, control exactly 0.500): this one scores
        +0.1583 [+0.1033, +0.2117] at 0.91x the no-op's cost, and every
        arm-vs-arm comparison against the three closed-loop variants is a NULL
        (p 0.0857 to 0.832). Indistinguishable effect, so the cheapest wins --
        and this one is cheaper than doing nothing, because the counter ends
        matches sooner.

        ONLY AN ATTACKING PLACEMENT DRAWS AN ANSWER. A card played in our own
        half is not a threat the opponent has to spend on, and charging one for
        it would penalise defence -- the failure mode that made an
        always-answering responder score BELOW a less frequent one.
        """
        if not self.reactive_rollout:
            return {}
        if self.horizon_ticks < COUNTER_MIN_HORIZON_TICKS:
            return {}
        if getattr(self, "_opp_idle_decisions", 0) >= COUNTER_PASSIVE_DECISIONS:
            # The real opponent is not answering anything; do not score our
            # attacks against an answer they are demonstrably not giving.
            return {}
        out = {}
        for st in cand.steps:
            if st.y < tactics.BRIDGE_ROW - 1:
                continue
            out[int(st.delay_ticks) + COUNTER_DELAY_TICKS] = (st.x, st.y)
        return out

    def _observe_opponent_spend(self, env):
        """Count consecutive decisions in which the opponent spent no elixir.

        Reads the engine's cumulative spend rather than inferring plays from
        the hand, because a play that the engine refused spends nothing and is
        not an answer.
        """
        try:
            spent = float(env.get_elixir_spent(1 - self.team))
        except Exception:                          # a stub env without the stat
            return
        prev = getattr(self, "_opp_spent_seen", None)
        if prev is not None and spent <= prev + 1e-6:
            self._opp_idle_decisions = getattr(self, "_opp_idle_decisions", 0) + 1
        else:
            self._opp_idle_decisions = 0
        self._opp_spent_seen = spent

    def counter_action(self, s, cell):
        """The opponent's (slot, x, y) for one chunk, in THEIR own frame."""
        if cell is None:
            return -1, 0.0, 0.0
        opp = 1 - self.team
        # Our attacking cell mirrored into their frame: our y=15 arrives at
        # their MIRROR_Y - 15. Clamped to their own half, since a defender
        # answers on its own side.
        rx = float(int(cell[0]))
        ry = float(int(max(0.0, min(float(tactics.BRIDGE_ROW),
                                    MIRROR_Y - cell[1]))))
        elixir = float(s.get_elixir_for_team(opp))
        hand = list(s.get_hand_for_team(opp))
        best = None
        for slot, cid in enumerate(hand[:HAND_SIZE]):
            if self.roles.get(cid, "melee") not in ("melee", "ranged", "building"):
                continue
            cost = float(E.get_card_info(cid)["cost"])
            if cost > elixir + 1e-6:
                continue
            if best is None or cost < best[0]:
                best = (cost, slot, cid)
        if best is None:
            return -1, 0.0, 0.0
        _, slot, cid = best
        y_abs = float(ry) if opp == 0 else MIRROR_Y - float(ry)
        if not s.is_valid_placement(cid, rx, y_abs, opp):
            return -1, 0.0, 0.0
        return slot, rx, ry

    def rollout_stats(self, env, cand):
        """Roll one candidate forward on a SNAPSHOT and read the engine.

        Both sides no-op after the play, the same assumption `search_ab_test`
        makes. CLAUDE.md records that past ~12 s of that a rollout "stops
        resembling the game", so the 3-6 s horizons here sit well inside the
        validated regime.

        A COMBO is the same loop with the second card played into one of the
        no-op chunks instead of a no-op -- no new engine capability, which is
        what made this the cheap half of TODO.md item 1.
        """
        s = env.snapshot()
        me, opp = self.team, 1 - self.team
        spent_before = float(env.get_elixir_spent(me))
        self.execute_steps(s, cand, self.rollout_ticks())
        killed = sum(s.get_elixir_value_killed_by(c, me) for c in self.deck)
        lost = sum(s.get_elixir_value_killed_by(c, opp) for c in self.deck)
        return {
            "elixir_spent": float(s.get_elixir_spent(me)) - spent_before,
            "tower_dealt": float(s.get_tower_damage_dealt(me)),
            "tower_taken": float(s.get_tower_damage_dealt(opp)),
            "killed": float(killed),
            "lost": float(lost),
            "crowns": float(s.get_towers_alive(me) - s.get_towers_alive(opp)),
            "pos": positional_advantage(
                np.asarray(s.get_observation_for_team(me), np.float32)),
        }

    def score(self, env, cand, baseline, obs_own):
        """Marginal utility of one candidate against the no-op baseline.

        Returns EXACTLY 0.0 for the no-op, by definition: it IS the baseline.
        Everything else is credited only for what it changes.
        """
        if not cand.steps:
            return 0.0
        st = self.rollout_stats(env, cand)
        p = self.profile
        cost = self.sequence_cost(cand)
        elixir = tactics.own_elixir(obs_own)

        u = (p["w_twr"] * (st["tower_dealt"] - baseline["tower_dealt"])
             - p["w_def"] * (st["tower_taken"] - baseline["tower_taken"])
             + p["w_trade"] * ((st["killed"] - baseline["killed"])
                               - (st["lost"] - baseline["lost"]))
             + p["w_crown"] * (st["crowns"] - baseline["crowns"])
             + p["w_pos"] * (st["pos"] - baseline["pos"])
             + p["w_cycle"] * self.sequence_cycle_value(cand.cards))
        # Opportunity cost. Above the overflow line the bar is discarding income,
        # so holding is not actually free and the charge is relieved -- the same
        # 9.0 threshold train.W_ELIXIR_OVERFLOW uses.
        overflow_relief = max(0.0, elixir - ELIXIR_OVERFLOW_AT) / (
            10.0 - ELIXIR_OVERFLOW_AT)
        u -= p["w_cost"] * cost * (1.0 - min(1.0, overflow_relief))
        u -= self.plan_reserve_penalty(
            cand, obs_own, list(env.get_hand_for_team(self.team)), elixir)
        return float(u)

    def plan_reserve_penalty(self, cand, obs, hand, elixir):
        """Elixir-equivalent charge for spending a committed plan's money.

        A CHARGE, NOT A GATE. The bot still ranks by simulation and a play worth
        more than the reserve still wins; what this removes is the marginal
        cycle card that is worth just enough to beat holding and, in doing so,
        makes unaffordable the follow-up whose first half has already been paid
        for.

        SCOPED TO A LIVE PLAN, which is what makes it work where the first
        attempt did not. A flat "save toward some future push" charge was
        measured dead -- the bar moved 1.79 -> 1.73 across reserves of 0.0 to
        5.0. This one only has to hold elixir for the one to five decisions a
        plan is actually waiting out its gap.

        THE THREAT EXEMPTION IS THE LOAD-BEARING PART. CLAUDE.md records that a
        FLAT solvency reserve "blocks exactly the spends that build a push", and
        a reserve that holds elixir through an incoming push makes the same
        mistake pointed at defence -- it does not save elixir, it loses the
        tower. The threshold is `tactics.HOG_MAX_THREAT`, reused rather than
        restated: it is already this project's calibrated "our half is clear
        enough to commit the win condition" line, measured against a median
        threat of 721 on a contested board. An earlier version used `> 0.0`,
        which was measured to be an OFF SWITCH -- in a real match something is
        on our half almost always.
        """
        st = self.pending
        if self.combo_reserve <= 0.0 or st is None or not cand.steps:
            return 0.0
        if cand.kind == "followup" or st.card_id in cand.cards:
            return 0.0                        # the plan's own second half
        if tactics.threat_level(obs) > tactics.HOG_MAX_THREAT:
            return 0.0                        # a real push: defence is free
        need = float(E.get_card_info(st.card_id)["cost"])
        regen = tactics.ELIXIR_REGEN_RATE * max(0, self.pending_ticks)
        if float(elixir) - self.sequence_cost(cand) + regen + 1e-6 >= need:
            return 0.0                        # affordable even after this spend
        return float(self.combo_reserve)

    def sequence_cost(self, cand):
        """Total elixir a candidate commits -- BOTH cards of a combo.

        The failure this prevents is silent: a pair charged for one card is a
        bot that believes escorting is free, and it would then escort
        everything. `W_COST` is the opportunity cost the no-op baseline does not
        absorb, and a second card is a second real payment.
        """
        return float(sum(E.get_card_info(c)["cost"] for c in cand.cards))

    def sequence_cycle_value(self, card_ids):
        """Cycle value of a whole sequence, with the queue advancing between
        plays.

        Each play moves the win condition one place closer, so the second card
        is credited against a SHORTER distance than the first. Summing
        `cycle_value` twice would over-pay a combo for a cycle it only advances
        once per card.
        """
        if self.wincon_id is None:
            return 0.0
        d = self.cycle.distance_to(self.wincon_id)
        total = 0.0
        for cid in card_ids:
            if cid == self.wincon_id:
                d = 0                      # it is being played; nothing to pull
                continue
            if d <= 0:
                continue
            cost = float(E.get_card_info(cid)["cost"])
            total += float(d) / max(1.0, cost)
            d -= 1
        return float(total)

    def cycle_value(self, card_id):
        """2.6 is DEFINED by cycling back to the win condition faster than the
        opponent cycles their answer, and the observation carries no cycle
        information at all (open problem #3). This is the only place that
        knowledge enters.

        A scored term, never a gate: a wrong estimate costs ranking quality,
        not the ability to act.
        """
        if self.wincon_id is None:
            return 0.0
        if card_id == self.wincon_id:
            return 0.0
        d = self.cycle.distance_to(self.wincon_id)
        if d <= 0:
            return 0.0
        cost = float(E.get_card_info(card_id)["cost"])
        # Playing anything advances the queue by one, so a cheap card is the
        # efficient way to pull the win condition closer.
        return float(d) / max(1.0, cost)

    # -- the decision ------------------------------------------------------
    def ability_flags(self, env, obs_own):
        """(activate slot 1, activate slot 2) for this decision.

        A HEURISTIC, and a deliberately plain one: activate a ready Champion
        ability when the opponent has a real force on the board. Until
        2026-09-16 the teacher never activated anything, so the mirror of a
        Champion deck played the Champion as a plain troop. Readiness alone is
        not enough -- it is true from the second a Champion lands with elixir
        to spare (measured for Golden Knight, Archer Queen and Monk), so it
        would spend the ability on arrival. The force bar is the advisor's own
        threat constant, not a new number.
        """
        from python_ai.rl.abilities import ability_engine_slots
        slots = ability_engine_slots(self.deck)
        if not slots:
            return (False, False)
        enemy_hp = float(tactics.enemy_hp_map(obs_own).sum())
        if enemy_hp <= tactics.DECK_COVERAGE_THREAT_HP:
            return (False, False)
        flags = {1: False, 2: False}
        for slot in slots:
            flags[slot] = bool(env.is_champion_ability_ready(self.team, slot))
        return (flags[1], flags[2])

    def act(self, env, obs_own):
        """(slot, x, y) in our own frame. slot == HAND_SIZE means hold."""
        self.cycle.observe(env.get_hand_for_team(self.team))
        self._observe_opponent_spend(env)
        self.last_kind = "noop"
        # The clock first, so a plan that comes due this decision is offered
        # this decision. See `tick_plan`.
        waiting = self.tick_plan()
        cands = self.candidates(env, obs_own)
        # SINGLE USE ONCE DUE. While a plan is still waiting out its gap it is
        # carried (and its money is reserved); on the decision it comes due it
        # is offered exactly once and then dropped, because a stale plan places
        # a card against a board that no longer exists.
        due = self.pending is not None and not waiting

        if self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon:
            # A uniformly random LEGAL action, no-op included -- waiting is a
            # real move and must stay in the noise distribution.
            c = cands[int(self.rng.integers(len(cands)))]
            if due:
                self.pending, self.pending_ticks = None, 0
            self.commit(c)
            self.last_kind = c.kind
            return c.slot, c.x, c.y

        playable = [c for c in cands if c.steps]
        if not playable:
            if due:
                self.pending, self.pending_ticks = None, 0
            return HAND_SIZE, 0.0, 0.0

        if self.horizon_ticks <= 0:
            out = self._rules_only(obs_own, playable)
            if due:
                self.pending, self.pending_ticks = None, 0
            return out

        baseline = self.rollout_stats(env, NOOP)
        elixir_now = tactics.own_elixir(obs_own)
        best, best_score = None, None
        for c in playable:
            sc = self.score(env, c, baseline, obs_own)
            if sc <= self.margin_for(c, elixir_now):
                continue           # does not beat holding by enough to matter
            if best is None or sc > best_score:
                best, best_score = c, sc
        if due:
            self.pending, self.pending_ticks = None, 0
        if best is None:
            return HAND_SIZE, 0.0, 0.0
        self.commit(best)
        self.last_kind = best.kind
        return best.slot, best.x, best.y

    def _rules_only(self, obs_own, playable):
        """Rung 0 ONLY (`horizon_ticks == 0`): no rollout at all, so the easiest
        rung costs nothing. This said "Stages 0-1" -- true of the 6-rung table,
        not of the 11-rung one, where rung 1 already rolls out 10 ticks. The gate matters more than the ranking here -- a bot that plays
        on every affordable step is the elixir-dumping opponent this project
        already replaced once."""
        threat = tactics.threat_level(obs_own)
        # AIR (TODO 00.5): the part of the threat only an anti-air card can
        # answer -- a flying building-targeter already on our half. With one
        # present, an anti-air card outranks everything, and a ground-only card
        # is played only if there is ALSO a ground threat for it to meet: a
        # Skeleton dropped under a Balloon is an elixir gift. Measured before:
        # the rung-0 teacher answered a lone Balloon with Skeletons and Ice Golem
        # more often than with its Musketeer or Ice Spirit.
        air = tactics.air_siege_map(obs_own)
        air[int(np.ceil(tactics.RIVER_Y)):] = 0.0
        air_hp = float(air.sum())
        ground_threat = threat - air_hp
        best, best_pri = None, 0.0
        for c in playable:
            pri = 0.0
            if c.role in ("melee", "ranged") and threat > 0.0:
                pri = 3.0
                if air_hp > 0.0:
                    pri = (3.5 if card_probes.damages_air(c.card_id)
                           else 3.0 if ground_threat > 0.0 else 0.0)
            elif c.role == "building" and threat > 0.0:
                pri = 2.5
                if air_hp > 0.0:
                    pri = (3.25 if card_probes.damages_air(c.card_id)
                           else 2.5 if ground_threat > 0.0 else 0.0)
            elif c.role == "spell":
                # Only cast when the catch uses the spell's FULL damage -- for a
                # Fireball, something worth its 4 elixir; forcing Fireball once
                # dropped win rate 97% -> 23%. The spell's own map and its own
                # damage: Fireball's 689 held a Zap back from a 243-HP Skeleton
                # clump and threw a Rocket at a lone Musketeer.
                _radius, full = spell_geometry(c.card_id)
                caught = spell_catch_for(obs_own, c.card_id).max()
                pri = 2.0 if caught >= full else 0.0
            elif c.role == "wincon" and tactics.hog_should_commit(obs_own):
                pri = 1.5
            if pri > best_pri:
                best, best_pri = c, pri
        if best is None:
            return HAND_SIZE, 0.0, 0.0
        self.last_kind = best.kind
        return best.slot, best.x, best.y
