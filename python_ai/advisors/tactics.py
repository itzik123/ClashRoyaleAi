"""Deterministic tactical advisor: spell aiming and defensive building placement.

WHY THIS EXISTS
---------------
Measured on `model_weights_dist_e3.pth` (40 greedy episodes, 1.5x opponent
elixir), three of eight deck cards are dead and their placement head is a
CONSTANT function of the board:

    card        H(place|card)   modal cell   modal share   plays/40 eps
    Cannon          0.098         (11, 0)       91.0%           24
    Fireball        0.141         (11, 0)       58.4%            2
    Giant           0.147         (11, 0)       54.1%            2
    Mini PEKKA      0.086         (14,15)       19.0%          230

Mini PEKKA has LOWER entropy than all three and is the most-played card, so
entropy is not the discriminator -- MODAL-CELL STABILITY ACROSS STATES is. A
healthy head is sharp but moves its mode with the board; these three return the
same cell in 54-91% of states regardless of what is happening.

(11,0) is our own back row behind the King. A Cannon there has 5.5 tiles of
range and 95.8% of the ones actually placed never had a single enemy inside it.

The root cause is a gradient COVERAGE hole, not the reward (see CLAUDE.md and
rl/ppo.py's `mb_placed`): both the actor loss and the placement entropy bonus
flow only through `placement_given_card` for the card that was CHOSEN, so a
card the policy has stopped playing receives zero placement gradient forever.
That is a deadlock -- frozen head makes the card worthless, worthlessness keeps
it unplayed, unplayed keeps the head frozen -- and it is self-sustaining, so no
amount of additional training escapes it.

This module supplies the missing signal: a cheap, deterministic, *correct*
answer for where those cards want to go, usable three ways --
  * as a candidate action for decision-time search (`search_ab_test`),
  * as a supervised placement target that gives the frozen head a gradient,
  * as a live-path override.

WHY IT READS THE OBSERVATION AND NOT THE ENGINE
-----------------------------------------------
`perception/tests/test_encoder_matches_engine.py` pins the live encoder
bit-equal to `getObservationForTeam(0)`, so anything computed from the
observation behaves identically in simulation and on the real screen. Reading
`env.getEntities()` instead would be more precise and would not transfer.

Every engine constant here is READ from the bindings wherever a binding exists.
Four are not exposed at all -- MAX_CELL_UNITS, MAX_UNIT_SPEED, the tower
positions, and OWN_HALF_RIVER_BUFFER -- so those are typed once WITH the header
named next to them, which is the fallback CLAUDE.md prescribes for values that
cannot be derived. `tests/test_tactics.py` pins each of them against the header
it came from, so a change on the C++ side fails a test here rather than
silently going stale. This project has paid for the second-copy mistake twice.
"""
from functools import lru_cache

import numpy as np

import clash_royale_env as E

from python_ai import engine_constants as EC
from python_ai.rewards import weights as W

CE = E.ClashRoyaleEnv

# --- engine geometry / normalizers, pulled live -----------------------------
BOARD_H = CE.BOARD_HEIGHT
BOARD_W = CE.BOARD_WIDTH
N_CH = CE.NUM_CHANNELS
SPATIAL = N_CH * BOARD_H * BOARD_W
MAX_TROOP_HP = CE.MAX_TROOP_HP
MAX_MATCH_ELIXIR = CE.MAX_MATCH_ELIXIR   # normaliser for the two spend scalars
MAX_CELL_UNITS = 5.0          # ClashEnv.h MAX_CELL_UNITS -- not bound; see below
MAX_UNIT_SPEED = 1.5          # ClashEnv.h MAX_UNIT_SPEED -- not bound; see below
# MAX_CELL_UNITS and MAX_UNIT_SPEED are the two normalizers ClashEnv.h uses that
# pybind does NOT expose (unlike MAX_TROOP_HP / MAX_BUILDING_HP / CH_*). They are
# hardcoded here with this comment naming the header, which is exactly the
# fallback CLAUDE.md prescribes for values that are not derivable. If either
# changes in ClashEnv.h, `test_python_ai.py::test_normalizers_match_header` fails.

# Enemy-side channel indices (side offset 1 = enemy, see ClashEnv.h).
CH_ENEMY_TROOP = (4, 5, 6)    # melee / ranged / building-targeter
CH_ENEMY_BUILDING = 7
CH_ENEMY_COUNT = CE.CH_COUNT + 1
CH_ENEMY_SPEED = CE.CH_SPEED + 1
CH_ENEMY_DPS = CE.CH_DPS + 1

# River START edge, DERIVED rather than typed. GameManager::getOwnHalfMaxY()
# returns getRiverStart() - OWN_HALF_RIVER_BUFFER, so adding the buffer back
# recovers the river edge exactly. The buffer itself is the one piece not
# bound (GameManager.h: `OWN_HALF_RIVER_BUFFER = 0.5f`) -- named here per
# CLAUDE.md's fallback rule, and pinned by
# tests/test_tactics.py::test_board_geometry_constants_match_their_headers.
#
# This was `RIVER_Y = 15.5` until 2026-08-20. The river has already MOVED once
# (2026-07-29, [16,18) -> [15.5,17.5)) and took stale Python copies with it, so
# the literal was the exact hazard this module's own docstring warns about.
_OWN_HALF_RIVER_BUFFER = 0.5   # GameManager.h
_probe = CE(list(range(8)), list(range(8)), 100)
RIVER_Y = _probe.get_own_half_max_y() + _OWN_HALF_RIVER_BUFFER
del _probe

# DERIVED, as of 2026-08-21. These used to be hardcoded here under the "no
# binding exposes board entities" escape hatch -- and they went stale the moment
# the arena was corrected, holding King x=9.0 and left Princess x=4.0 against an
# engine that had moved to 8.5 and 3.0. ArenaLayout.h is bound now, so read it.
OWN_PRINCESS = ((EC.LEFT_LANE_X, EC.princess_y(0)),
                (EC.RIGHT_LANE_X, EC.princess_y(0)))
OWN_KING = (EC.BOARD_CENTER_X, EC.king_y(0))

# Fireball. THE ID AND THE DAMAGE COME FROM `rewards.weights`, which owns the
# single definition of both -- this file used to repeat `FIREBALL_DAMAGE =
# 689.0` under a comment claiming it was "read from the registry", which it was
# not: `get_card_info` exposes cost, name, is_spell and placement_radius, but
# NOT damage. Nothing derived it, so nothing would have caught the two copies
# drifting -- leaving the shaping term and this advisor disagreeing about one
# physical fact, one calling a tower lethal while the other called the same
# cast worthless.
#
# The RADIUS stays here because `weights` has no use for it and therefore no
# definition of it; it is annotated with the header that owns it, which is the
# documented fallback where a value genuinely cannot be derived.
FIREBALL_ID = W.FIREBALL_CARD_ID
FIREBALL_RADIUS = 2.5         # CardRegistry.h:752 spell(7,...,2.5f,689,10,'O')
FIREBALL_DAMAGE = W.FIREBALL_DAMAGE
FIREBALL_DELAY_TICKS = 10
CANNON_ID = 25
CANNON_RANGE = 5.5            # CardRegistry.h:736 building(25,...,5.5f,202,10)


def spatial(obs):
    """(N_CH, 34, 18) view of the spatial half of a flat observation."""
    a = np.asarray(obs, dtype=np.float32)
    return a[:SPATIAL].reshape(N_CH, BOARD_H, BOARD_W)


def enemy_hp_map(obs):
    """Approximate enemy troop HP per cell, in absolute HP.

    Channels 4-6 hold normalized HP and OVERWRITE rather than accumulate (open
    problem #3 in CLAUDE.md), so a stacked cell under-reports. CH_COUNT is the
    only channel that accumulates, so multiplying by it recovers most of the
    loss -- that is precisely the gap CH_COUNT was added to close.
    """
    sp = spatial(obs)
    hp = sp[list(CH_ENEMY_TROOP)].sum(axis=0) * MAX_TROOP_HP
    count = np.maximum(1.0, sp[CH_ENEMY_COUNT] * MAX_CELL_UNITS)
    return hp * count


def enemy_speed_map(obs):
    """Enemy move speed per cell in tiles/tick (denormalized)."""
    return spatial(obs)[CH_ENEMY_SPEED] * MAX_UNIT_SPEED


def advance(hp, speed, ticks):
    """Shift enemy mass `ticks` ticks along its direction of travel.

    Enemies advance toward OUR towers, i.e. toward decreasing y in the team-0
    observation frame. Mass is split between the two integer rows it lands
    between, so a half-tile step moves half the mass -- a hard round would make
    the lead a step function of speed and quantize away every sub-tile lead.

    This is a first-order model, deliberately: it ignores pathing to the bridge
    and target re-acquisition. `test_python_ai.py` measures it against the
    simulator's own 10-tick rollout, which is the ground truth it approximates.
    """
    out = np.zeros_like(hp)
    dy = speed * ticks
    for y in range(BOARD_H):
        row = hp[y]
        if not row.any():
            continue
        for x in range(BOARD_W):
            if row[x] <= 0.0:
                continue
            ny = y - dy[y, x]
            lo = int(np.floor(ny))
            frac = ny - lo
            if 0 <= lo < BOARD_H:
                out[lo, x] += row[x] * (1.0 - frac)
            if 0 <= lo + 1 < BOARD_H:
                out[lo + 1, x] += row[x] * frac
    return out


@lru_cache(maxsize=None)
def _disc_offsets(radius):
    """Cell offsets within `radius`, as an immutable tuple.

    MEMOIZED: a pure function of one float, over the handful of distinct radii
    this module ever asks for (spell radii and unit ranges), so the cache is
    bounded in practice and every hit is exact. It cannot change a downstream
    number -- the result is integer offsets from a hypot test --
    `tests/test_tactics_disc_offsets.py` recomputes the original definition
    inline and compares exactly rather than taking that on trust.

    A TUPLE, not a list, because memoizing hands the SAME object to every
    caller: mutation would now corrupt every other call site instead of a local
    copy. Both callers only iterate.

    This replaces a module-level `_FIREBALL_DISC` that was computed at import
    and then never read once -- the cache was built, left unwired, and
    `spell_catch_map` recomputed the identical list inside its scatter loop
    anyway.
    """
    r = int(np.ceil(radius))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Units are summarized at their CELL CENTRE, because the encoder
            # truncates a float position into a cell; the aim point is the
            # integer cell coordinate the engine actually receives.
            if np.hypot(dx + 0.5, dy + 0.5) <= radius:
                offs.append((dy, dx))
    return tuple(offs)


def spell_catch_map(obs, radius=FIREBALL_RADIUS, lead_ticks=0,
                    damage=FIREBALL_DAMAGE):
    """(34,18) map of the enemy value a spell aimed at each cell would catch.

    Per-cell contribution is capped at `damage` per unit present: overkill is
    wasted, and without the cap the map is dominated by whatever has the most
    raw HP (a Giant) rather than by the clump the spell actually kills.

    TWO THINGS MEASURED HERE THAT CAME OUT AGAINST THE OBVIOUS GUESS, both on
    1,059 states paired against the engine's own `get_elixir_value_killed_by`
    (share of the achievable value at each state; see PLACEMENT_COLLAPSE.md):

                        lead=0    lead=10 (1.0s)
        damage-cap      75.5%       52.4%
        kill-weighted   75.0%       51.2%

    * `lead_ticks` DEFAULTS TO 0 -- do not "fix" this. Fireball has
      spellDelayTicks=10, so aiming where the target will be looks obviously
      right, and it is measurably wrong: it costs 23 points of achievable
      value. The blast radius is 2.5 tiles while 1s of movement is only
      0.6-1.6 tiles at post-2026-08-07 speeds, so a target aimed at directly is
      still inside the blast after it moves -- while a target that is STANDING
      STILL (engaged in combat, which is the common case in the states worth
      spelling) gets led straight off the edge of it. Leads of 0-6 ticks were
      indistinguishable; only the full 10 hurt.
    * Weighting by what the spell KILLS rather than what it damages changes
      nothing (75.0 vs 75.5). That variant was built, measured paired, and
      deleted; the elixir-value objective is already well approximated by the
      damage cap.
    """
    hp = enemy_hp_map(obs)
    count = np.maximum(1.0, spatial(obs)[CH_ENEMY_COUNT] * MAX_CELL_UNITS)
    if lead_ticks:
        hp = advance(hp, enemy_speed_map(obs), lead_ticks)
    effective = np.minimum(hp, damage * count)

    out = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
    ys, xs = np.nonzero(effective)
    # HOISTED out of the per-cell loop -- it is loop-invariant, and the scatter
    # loop in `_reach_cover` below already binds it this way. Left inside, it
    # rebuilt the identical list once per occupied enemy cell.
    offsets = _disc_offsets(radius)
    for y, x in zip(ys, xs):
        v = effective[y, x]
        for dy, dx in offsets:
            ay, ax = y - dy, x - dx
            if 0 <= ay < BOARD_H and 0 <= ax < BOARD_W:
                out[ay, ax] += v
    return out


def best_spell_cell(obs, legal=None, radius=FIREBALL_RADIUS,
                    lead_ticks=0, damage=FIREBALL_DAMAGE):
    """(x, y, caught_hp) -- where to aim, and how much it is worth.

    `legal` is an optional (34*18,) bool mask; pass the net's own
    `placement_mask` so the advisor can never propose a cell the engine will
    silently refuse (the failure mode that made 58.7% of card choices no-ops
    before the placement mask existed).
    """
    m = spell_catch_map(obs, radius, lead_ticks, damage)
    flat = m.reshape(-1).copy()
    if legal is not None:
        flat[~np.asarray(legal, dtype=bool).reshape(-1)] = -1.0
    i = int(np.argmax(flat))
    return float(i % BOARD_W), float(i // BOARD_W), float(flat[i])


GIANT_ID = 2
# Our side of the river (the river is [15.5, 17.5)), so a Giant placed here
# starts crossing immediately instead of walking the length of our own half.
BRIDGE_ROW = 15
# Bridge CELLS, truncated from the engine's bridge centres the same way
# cell_to_xy does it (never round() -- see CLAUDE.md's discretization note).
# A two-tile bridge centred on 2.5 spans cells 2 and 3; int() picks 2.
BRIDGE_XS = (int(EC.LEFT_BRIDGE_X), int(EC.RIGHT_BRIDGE_X))


def best_giant_cell(obs, legal=None):
    """(x, y, expected_damage_rank) -- where to commit the win condition.

    MEASURED, not assumed. Injecting a Giant at a candidate cell and running the
    engine 600 ticks (it crosses ~15 tiles at 0.06 tiles/tick), scoring enemy
    tower damage against the same state with no Giant, over 913 states:

        policy's own cell        3.3      <- it places at its own back row
        back row (y=1)           3.0
        lane, mid (y=11)       172.8
        random legal cell       94.3
        BRIDGE, weaker lane    535.6      +532.3 vs policy, sign p = 3.0e-87

    A back-row Giant is a legitimate Clash opening in general, but against this
    engine's opponent it contributes essentially nothing: the policy's own cell
    and an explicit back-row rule both score ~3 damage, i.e. the Giant dies or
    stalls before it arrives. At the bridge it is 162x better.

    Lane choice goes to whichever side the enemy is currently defending LESS,
    which is the only board-dependent part of the rule -- and it is what makes
    this an advisor rather than a constant.
    """
    hp = enemy_hp_map(obs)
    left, right = hp[:, :BOARD_W // 2].sum(), hp[:, BOARD_W // 2:].sum()
    # Go away from the enemy's mass; ties break left, arbitrarily but fixed.
    x = BRIDGE_XS[0] if right >= left else BRIDGE_XS[1]
    y = BRIDGE_ROW
    if legal is not None:
        flat = np.asarray(legal, bool).reshape(-1)
        if not flat[y * BOARD_W + x]:
            # Walk inward along the row to the nearest legal column rather than
            # silently emitting a cell the engine will refuse.
            for dx in range(1, BOARD_W):
                for cand in (x - dx, x + dx):
                    if 0 <= cand < BOARD_W and flat[y * BOARD_W + cand]:
                        return float(cand), float(y), 0.0
    return float(x), float(y), 0.0


# --------------------------------------------------------------------------
# Hog Rider -- the 2.6 win condition
# --------------------------------------------------------------------------
# CardRegistry.h: id 15, cost 4, building-targeter. Hardcoded like every other
# id in this file because tactics.py must not import gym_wrapper (that pulls in
# gymnasium and torch, and this module is imported by the advisor target on the
# rollout path). test_hog_id_matches_the_registry pins it.
HOG_ID = 15
HOG_COST = 4.0

# Elixir kept back AFTER paying for the Hog.
#
# CALIBRATED, after the first attempt set this to 3.0 on Clash theory alone --
# hold back enough for the Cannon -- and the gate then opened on 0 of 542
# states. Measured on this policy's own trajectory: elixir median 2.25, p90
# 4.35. A cheap-cycle deck spends continuously and essentially never banks 7,
# so a 3-elixir reserve is not a conservative rule here, it is an off switch,
# and an advisor that never speaks distils nothing.
#
# At 0 the rule fires whenever the Hog is merely AFFORDABLE, which is ~10% of
# states -- ample signal. The safety it gives up is covered by HOG_MAX_THREAT
# below: with our half clear there is nothing to reserve against yet, and 2.6
# cycles back to a defence in a few seconds.
HOG_DEFENSIVE_RESERVE = 0.0

# Enemy HP already on our half above which we defend instead of committing.
#
# ABSOLUTE HP, not a normalised fraction -- enemy_hp_map's docstring says so and
# the first version of this constant (0.35) did not, which made it ~3 orders of
# magnitude too strict: measured threat_level has median 721 on a contested
# board. For scale a Musketeer is ~600 HP, so this admits a clear board or one
# small unit already being handled, and refuses a real push.
HOG_MAX_THREAT = 800.0

# Estimated opponent elixir above which we do NOT commit. At 7+ they can answer
# the Hog AND counter-push, which is the situation the rule exists to avoid.
HOG_MAX_OPP_ELIXIR = 7.0

# ELIXIR_REGEN_RATE from ClashEnv.h -- not bound, so hardcoded with the header
# named, per this project's rule for constants that cannot be derived.
ELIXIR_REGEN_RATE = 0.035
STARTING_ELIXIR = 5.0
MAX_ELIXIR = 10.0
TRAINING_MAX_TICKS = 3600.0


# FORWARD offsets into the extra-scalar block. Verified against the engine on
# 2026-08-29 by moving each quantity and watching which index responds: S+0 is
# elapsed time over max_ticks, S+1 OUR cumulative elixir spend and S+2 the
# OPPONENT'S over MAX_MATCH_ELIXIR, then six tower HPs (own king, own princess
# x2, enemy king, enemy princess x2 -- 2534/4008 = 0.632 identifies them).
#
# THESE WERE NEGATIVE INDICES UNTIL 2026-08-29, AND HAD BEEN WRONG SINCE THE
# CYCLE BLOCKS LANDED ON 2026-08-27. obs[-9] is index 13967, which is inside
# the card-recency block that now sits BEHIND the extra scalars, so
# `elapsed_ticks` returned a recency float -- measured 0.0 at engine tick 300,
# where the forward offset correctly reads 300.0.
#
# CLAUDE.md records that defect and says five call sites were fixed. This file
# was a sixth and the sweep could not have found it: that sweep searched for
# `observation_size() - NUM_EXTRA_SCALARS`, and these two sites spell the same
# bug as a negative index. Same defect, different spelling.
#
# What it cost while it was live: `opp_elixir_estimate` computes income over
# `elapsed_ticks`, so with the clock pinned at zero it returned essentially the
# starting elixir forever, and `hog_advice`'s gate -- the only win-condition
# advisor target -- was gated on a constant rather than on the board.
_S = EC.EXTRA_SCALARS_START
IDX_ELAPSED = _S + 0
IDX_OWN_SPEND = _S + 1
IDX_OPP_SPEND = _S + 2


def elapsed_ticks(obs):
    """Match time in ticks, from the extra-scalar tail."""
    return float(np.asarray(obs, dtype=np.float32)[IDX_ELAPSED]) * TRAINING_MAX_TICKS


def opp_elixir_estimate(obs, multiplier=1.0):
    """Opponent's current elixir, inferred from income minus observed spend.

    The observation carries their CUMULATIVE SPEND, not their bar, so this
    reconstructs the bar: start + regen*time - spent, clamped to the range the
    engine can reach.

    KNOWN BIAS, stated because it decides which way the gate errs. `multiplier`
    is the opponent's elixir multiplier, which the curriculum raises to 1.5 and
    which the OBSERVATION DOES NOT CARRY. Left at 1.0 this UNDER-estimates their
    elixir at higher stages, so the gate opens more often than it should exactly
    where the opponent is strongest. That is the wrong direction, and it is why
    HOG_MAX_OPP_ELIXIR is set well below the 10 cap rather than near it.

    The net's own aux head estimates this to MAE ~0.96 and would be the better
    source, but it is not available here: tactics.py takes an observation, not a
    network, and the advisor target is computed on the rollout path where
    threading the net through would change that contract.
    """
    a = np.asarray(obs, dtype=np.float32)
    spent = float(a[IDX_OPP_SPEND]) * MAX_MATCH_ELIXIR
    gained = STARTING_ELIXIR + ELIXIR_REGEN_RATE * elapsed_ticks(obs) * multiplier
    return float(np.clip(gained - spent, 0.0, MAX_ELIXIR))


def best_hog_cell(obs, legal=None):
    """(x, y, 0.0) -- the bridge, on whichever lane the enemy defends LESS.

    Placement is the front row of our own half at a bridge column. The row
    matters more than the column: from y=15 the Hog is across the river
    immediately, which is the whole point of the card -- it minimises the
    opponent's reaction time. Placed deeper it walks the length of our own half
    first, which is what the untrained policy does and what scores ~3 damage.

    Identical in form to the rule measured for the Giant (bridge, weaker lane:
    535.6 enemy tower damage against 3.3 for the policy's own cell, n=913,
    p=3.0e-87). The Hog is faster and cheaper, so the argument is strictly
    stronger for it -- but that is an argument, and this rule is ENGINE-SCORED
    separately before it is allowed to train anything.
    """
    hp = enemy_hp_map(obs)
    left, right = hp[:, :BOARD_W // 2].sum(), hp[:, BOARD_W // 2:].sum()
    x = BRIDGE_XS[0] if right >= left else BRIDGE_XS[1]
    y = BRIDGE_ROW
    if legal is not None:
        flat = np.asarray(legal, bool).reshape(-1)
        if not flat[y * BOARD_W + x]:
            for dx in range(1, BOARD_W):
                for cand in (x - dx, x + dx):
                    if 0 <= cand < BOARD_W and flat[y * BOARD_W + cand]:
                        return float(cand), float(y), 0.0
    return float(x), float(y), 0.0


def hog_should_commit(obs, multiplier=1.0, cost=None):
    """Is NOW the moment to send the win condition? (timing, not placement)

    Three conditions, all necessary:

      1. OUR HALF IS CLEAR. Committing 4 elixir into an active push means
         defending the answer with what is left, and 2.6 has no card that
         defends and pushes at once.
      2. WE STAY SOLVENT. Elixir after the Hog must still cover a real answer,
         which is the Cannon at 3 -- not the 1-cost cycle cards, which do not
         stop anything on their own.
      3. THEY ARE NOT BANKED. Against an opponent sitting near max elixir the
         Hog is answered AND counter-pushed, which is the trade this rule is
         specifically meant to refuse.

    Returning False is not a no-op: `hog_advice` then emits NO TARGET, and the
    advisor-target machinery falls back to the entropy term for that row. That
    matters because tactics always returning a cell is what teaches a CONSTANT
    -- the exact pathology the 2026-08-14 cure was built to undo. The gate is
    the load-bearing half of the rule.
    """
    # No enemy ANYWHERE means the lane choice carries no information: left and
    # right both sum to zero, the tiebreak fires, and the rule degenerates to a
    # fixed cell. Distilling that teaches a constant, which is the collapse this
    # gate exists to prevent -- so decline, exactly as the "cell" kind already
    # did. This is not the same test as the threat check below: that one is
    # about a push on OUR half, this one is about having any read at all.
    if float(enemy_hp_map(obs).sum()) <= 0.0:
        return False
    if threat_level(obs) > HOG_MAX_THREAT:
        return False
    # `cost` is the WALKING WIN CONDITION's own cost (a Royal Giant is 6, not
    # the Hog's 4); None keeps the historical Hog value for existing callers.
    if own_elixir(obs) < (HOG_COST if cost is None else float(cost)) + HOG_DEFENSIVE_RESERVE:
        return False
    if opp_elixir_estimate(obs, multiplier) > HOG_MAX_OPP_ELIXIR:
        return False
    return True


def hog_advice(obs, legal=None, multiplier=1.0, cost=None):
    """(x, y) to commit the win condition now, or None to say nothing this step."""
    if not hog_should_commit(obs, multiplier, cost):
        return None
    x, y, _rank = best_hog_cell(obs, legal)
    return x, y


def threat_map(obs):
    """Enemy HP already on our half, the part we actually have to answer."""
    hp = enemy_hp_map(obs)
    out = hp.copy()
    out[int(np.ceil(RIVER_Y)):] = 0.0
    return out


def threat_level(obs):
    """Scalar: total enemy troop HP on our half."""
    return float(threat_map(obs).sum())


#: Enemy HP on our half above which the deck-coverage floor is allowed to push.
#: DERIVED from ElixirGate's own `threat_hp` default rather than restated: that
#: value is calibrated as "just under a single Musketeer (721) and above a lone
#: Minion (230), so any real commitment releases the reserve", which is exactly
#: the question the floor needs answered -- is a defence actually called for.
DECK_COVERAGE_THREAT_HP = 400.0


def threat_level_batch(obs):
    """`threat_level` over a (B, obs_dim) torch batch, in absolute enemy HP.

    Exists because the PPO update needs this per ROW and cannot afford a numpy
    round-trip per sample. It is deliberately written directly beneath the
    scalar version and reproduces its arithmetic step for step -- channels,
    the CH_COUNT correction, and the same `ceil(RIVER_Y)` cutoff -- because a
    second, drifting definition of "threat" is exactly the defect class this
    project has now found in eight arena constants.
    `tests/test_threat_gated_deck_coverage.py` pins the two against each other.
    """
    import torch
    sp = obs[:, :SPATIAL].reshape(-1, N_CH, BOARD_H, BOARD_W)
    hp = sp[:, list(CH_ENEMY_TROOP)].sum(dim=1) * MAX_TROOP_HP
    count = torch.clamp(sp[:, CH_ENEMY_COUNT] * MAX_CELL_UNITS, min=1.0)
    per_cell = hp * count
    per_cell = per_cell[:, :int(np.ceil(RIVER_Y)), :]
    return per_cell.sum(dim=(1, 2))


def threat_lane(obs):
    """-1 left, +1 right, 0 none -- which side the push is on.

    Uses all enemy mass past the halfway point of the enemy half, not just what
    has already crossed: a Cannon placed only once the push is on our side is
    already too late at 0.6-1.6 tiles/s.
    """
    hp = enemy_hp_map(obs)
    approaching = hp[: int(RIVER_Y) + 6]
    left = approaching[:, : BOARD_W // 2].sum()
    right = approaching[:, BOARD_W // 2:].sum()
    if left <= 0 and right <= 0:
        return 0
    return -1 if left > right else 1


def own_elixir(obs):
    """Our current elixir, read from the scalar tail (ClashEnv divides by 10)."""
    return float(np.asarray(obs, dtype=np.float32)[SPATIAL]) * 10.0


class SolvencyGate:
    """Refuse to spend below a reserve while nothing is attacking.

    THE MEASUREMENT. The bot spends ~105 elixir per episode against ~98 of
    income and sits below 3 elixir -- the cheapest card in the deck -- on 65.3%
    of all decisions, rising to 60.8% during a big push, where P(play) is then
    0.0% by arithmetic rather than by choice. It is not hoarding and not
    apathetic; conditioned on affordability its threat response is real
    (27.8% -> 34.9%). It is simply broke when the answer is needed.

    WHY A HARD GATE RATHER THAN SHAPING. The potential-based solvency term in
    elixir_shaping.py is policy-invariant by construction, which makes it safe
    but also means it cannot change the optimum -- only how fast one is found.
    Measured over ~80 PPO updates it moved the bankruptcy rate by -1.0 points
    (95% CI [-2.7, +0.6], p = 0.21), i.e. not at all. This gate does not wait
    for learning: it removes the spends that cause the insolvency.

    WHY IT IS CONDITIONED ON THREAT, which is the part that keeps it from
    capping the ceiling. Spending to zero is often correct -- on a counterpush,
    or to answer something. The gate only binds when NOTHING is attacking, which
    is exactly where the measurement found the waste: 739 of the bot's plays
    happen at zero threat against 126 during a big push. Under threat the gate
    opens completely and the policy may spend to zero as before.

    A FLAT RESERVE IS ANTI-OFFENSE, and that had to be fixed before this was
    usable. Building a push means spending exactly when nothing is attacking,
    which is the only situation a flat gate blocks. Measured over 6 paired
    openings, a flat reserve of 4 cut tower damage DEALT from 8,739 to 4,094 per
    episode -- 0 of 6 pairs better, sign p = 0.03 -- while fixing bankruptcy
    exactly as designed. It bought solvency by refusing to attack.

    So the reserve is held against what the opponent can actually PUNISH with:
    `reserve_effective = min(reserve, opponent_elixir)`. If they are broke they
    cannot make us pay for committing, and the gate steps out of the way. The
    opponent-elixir estimate comes from the network's own auxiliary head, whose
    MAE is ~0.83-1.0 against a predict-the-mean baseline of ~1.35 (CLAUDE.md) --
    a real capability, not a guess, and one already paid for.
    """

    def __init__(self, reserve=4.0, threat_hp=400.0):
        # 4.0 = the cost of every dedicated defensive answer in DEFAULT_DECK
        # (Valkyrie, Musketeer, Mini PEKKA, Fireball). Below it the agent cannot
        # respond to anything that matters.
        self.reserve = reserve
        # Enemy HP on our half above which the gate opens. 400 is just under a
        # single Musketeer (721) and above a lone Minion (230), so any real
        # commitment releases the reserve.
        self.threat_hp = threat_hp

    def effective_reserve(self, opp_elixir=None):
        if opp_elixir is None:
            return self.reserve
        return min(self.reserve, max(0.0, float(opp_elixir)))

    def allows(self, obs, cost, opp_elixir=None):
        """May a card of `cost` be played from this state?"""
        if threat_map(obs).sum() >= self.threat_hp:
            return True                      # something is attacking: spend freely
        return own_elixir(obs) - cost >= self.effective_reserve(opp_elixir)

    def mask(self, obs, hand_costs, opp_elixir=None):
        """Bool list over hand slots + no-op, ANDed with affordability by caller."""
        return [self.allows(obs, c, opp_elixir) for c in hand_costs] + [True]


class TacticalOverride:
    """Deterministic tactical plays layered on a policy, SOLVENCY-GATED.

    The gate is the whole design, and it is there because the ungated version
    was measured and lost. Forcing the dead cards in at a naive threshold played
    5.5 Cannons and 2.75 Fireballs per episode -- about 27 extra elixir against
    an agent that already spends ~105 of its ~98 elixir income per episode and
    sits below 3 elixir 65.3% of the time. Both forced arms lost to the control,
    and the comparison was measuring bankruptcy rather than placement quality.

    So an override here may only spend elixir the agent can actually spare:
    `reserve` is what must REMAIN after the play, and the Cannon is rate-limited
    to roughly its own 300-tick lifetime so it cannot be stacked.

    Thresholds are set a priori from card stats, never tuned on win rate:
    Fireball needs at least a 3-cost squad's worth of HP in the blast (Minions
    are 3x230=690) and the Cannon needs at least a Musketeer (721) approaching.
    """

    def __init__(self, legality_table, reserve=4.0,
                 fireball_min_catch=690.0, cannon_min_cover=721.0,
                 cannon_cooldown_steps=30):
        self._legal = legality_table
        self.reserve = reserve
        self.fireball_min_catch = fireball_min_catch
        self.cannon_min_cover = cannon_min_cover
        self.cannon_cooldown_steps = cannon_cooldown_steps
        self._cannon_cd = 0

    def reset(self):
        self._cannon_cd = 0

    def _legal_for(self, card_id):
        return np.asarray(self._legal[card_id]).astype(bool)

    def __call__(self, obs, hand, affordable, default):
        """Return (slot, x, y). `default` is the policy's own greedy action.

        `hand` is the card id per slot (-1 empty), `affordable` the
        affordability mask over slots. Both must be read from the SAME
        observation the decision is made on -- the hand rotates the instant a
        card is played.
        """
        self._cannon_cd = max(0, self._cannon_cd - 1)
        elixir = own_elixir(obs)

        def solvent(cost):
            return elixir - cost >= self.reserve

        if FIREBALL_ID in hand:
            s = hand.index(FIREBALL_ID)
            if s < len(affordable) and affordable[s] and solvent(4.0):
                fx, fy, val = best_spell_cell(obs, legal=self._legal_for(FIREBALL_ID))
                if val >= self.fireball_min_catch:
                    return s, fx, fy

        if CANNON_ID in hand and self._cannon_cd == 0:
            s = hand.index(CANNON_ID)
            if s < len(affordable) and affordable[s] and solvent(3.0):
                cx, cy, cover = best_building_cell(obs, legal=self._legal_for(CANNON_ID))
                if cover >= self.cannon_min_cover:
                    self._cannon_cd = self.cannon_cooldown_steps
                    return s, cx, cy

        return default


def building_score_map(obs, legal=None, rng_range=CANNON_RANGE):
    """The full per-cell score `best_building_cell` takes the argmax of.

    Split out because the argmax alone is a BAD SUPERVISION TARGET and the map
    is a good one. Coverage is accumulated by scattering each enemy's HP over a
    disc of radius `rng_range`, so every cell that reaches the same set of
    enemies scores EXACTLY the same, and the only tie-breakers are two step
    functions (the y>=6 depth bonus and the lane multiplier). The surface is
    therefore made of large exact plateaus, and `np.argmax` resolves them by
    row-major order -- it returns the top-left cell of the winning plateau,
    which jumps to a completely different cell when the plateau shifts by one
    tile, while the advisor is genuinely INDIFFERENT across all of them.

    Fitting a head to that argmax asks it to learn a tie-break rule that carries
    no value. Measured 2026-08-14 on 594 held-out states: Fireball, whose target
    comes from a smooth blast-coverage map, reached 54.3% exact-cell match after
    the hi-res branch; the Cannon stayed at 0.2% while its top-1 probability
    rose 10x and its modal share fell 89.8% -> 54.9% -- i.e. the head DID become
    state-dependent and sharp, against a target whose exact cell is noise.

    Returns (BOARD_H, BOARD_W) float32 with -inf on illegal/across-river cells,
    which is what `masked_kl`-style losses expect.
    """
    return _building_score(obs, legal, rng_range)[0]


def best_building_cell(obs, legal=None, rng_range=CANNON_RANGE):
    """(x, y, score) -- where a defensive building actually defends.

    Scores each legal cell by how much approaching enemy HP it would bring into
    its own attack range, with two corrections that matter more than the raw
    coverage:

    * a cell BEHIND our princess towers is discounted, because anything it can
      reach from there has already hit the tower;
    * a cell far from the threatened lane is discounted, since the building
      cannot move.

    With no threat on the board it returns the classic defensive pocket -- in
    front of the King, between the two Princess towers -- which is where a
    building wants to be pre-placed anyway.
    """
    score, kind = _building_score(obs, legal, rng_range)
    if kind == "none":
        return float(OWN_KING[0]), 9.0, 0.0
    i = int(np.argmax(score))
    value = 0.0 if kind == "pocket" else float(score.reshape(-1)[i])
    return float(i % BOARD_W), float(i // BOARD_W), value


def _building_score(obs, legal=None, rng_range=CANNON_RANGE):
    """(score map, kind). kind is 'coverage', 'pocket' or 'none'.

    One implementation shared by `best_building_cell` and
    `building_score_map`, so the cell that is PLAYED and the surface that is
    DISTILLED can never come from two drifting copies of this arithmetic.
    """
    hp = enemy_hp_map(obs)
    # Look ahead: score against where the push WILL be in ~2s, not where it is.
    future = advance(hp, enemy_speed_map(obs), 20)
    lane = threat_lane(obs)

    # Coverage by SCATTERING from each occupied enemy cell onto the cells that
    # could reach it, rather than looping every candidate cell against every
    # enemy. Same result, O(n_enemy * disc) instead of O(612 * n_enemy) -- the
    # difference between ~10 ms and ~0.2 ms per call, which is what makes this
    # usable inside a training loop rather than only in offline analysis.
    cover = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
    offs = _disc_offsets(rng_range)
    ys, xs = np.nonzero(future)
    for ey, ex in zip(ys, xs):
        v = future[ey, ex]
        for dy, dx in offs:
            cy, cx = ey - dy, ex - dx
            if 0 <= cy < BOARD_H and 0 <= cx < BOARD_W:
                cover[cy, cx] += v

    # Forward of our own princess towers is worth more than behind them, and a
    # building cannot chase, so the threatened lane matters.
    depth = np.where(np.arange(BOARD_H) >= 6, 1.0, 0.35).astype(np.float32)
    score = cover * depth[:, None]
    if lane != 0:
        side = np.where(np.arange(BOARD_W) < BOARD_W // 2, -1, 1)
        score = score * np.where(side == lane, 1.0, 0.5).astype(np.float32)[None, :]

    # A building cannot be placed across the river.
    score[int(np.ceil(RIVER_Y)):] = -np.inf
    if legal is not None:
        score = np.where(np.asarray(legal, bool).reshape(BOARD_H, BOARD_W),
                         score, -np.inf)

    if not np.isfinite(score).any():
        return score, "none"
    if float(np.nanmax(score[np.isfinite(score)])) <= 0.0:
        # Nothing is approaching: fall back to the classic pocket -- central,
        # forward of the King, not across the river -- which is where a building
        # wants to be pre-placed anyway.
        gy, gx = np.mgrid[0:BOARD_H, 0:BOARD_W]
        pocket = -(np.abs(gx - OWN_KING[0]) + np.abs(gy - 9.0)).astype(np.float32)
        pocket[int(np.ceil(RIVER_Y)):] = -np.inf
        if legal is not None:
            pocket = np.where(np.asarray(legal, bool).reshape(BOARD_H, BOARD_W),
                              pocket, -np.inf)
        return pocket, "pocket"

    return score, "coverage"
