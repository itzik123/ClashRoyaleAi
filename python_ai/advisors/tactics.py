"""Deterministic tactical advisor: spell aiming, defensive building placement and
win-condition timing.

Cheap, engine-validated answers for where cards want to go, used as candidates
for search and the teacher, as a supervised target for cards the policy has
stopped playing (whose placement heads get no gradient otherwise), and as a
live-path override.

It reads the observation, not the engine, so it behaves identically in
simulation and on a real screen (the live encoder is pinned bit-equal to
`getObservationForTeam(0)`). Constants are read from the bindings; the few that
are not exposed are typed once with their header named, and
`tests/test_tactics.py` pins them against those headers.
"""
from functools import lru_cache

import numpy as np

import clash_royale_env as E

from python_ai import engine_constants as EC
from python_ai.rewards import weights as W

CE = E.ClashRoyaleEnv

# --- engine geometry / normalizers ---
BOARD_H = CE.BOARD_HEIGHT
BOARD_W = CE.BOARD_WIDTH
N_CH = CE.NUM_CHANNELS
SPATIAL = N_CH * BOARD_H * BOARD_W
MAX_TROOP_HP = CE.MAX_TROOP_HP
MAX_MATCH_ELIXIR = CE.MAX_MATCH_ELIXIR   # normaliser for the two spend scalars
MAX_CELL_UNITS = 5.0          # ClashEnv.h, not bound
MAX_UNIT_SPEED = 1.5          # ClashEnv.h, not bound

# Enemy-side channels (side offset 1).
CH_ENEMY_TROOP = (4, 5, 6)    # melee / ranged / building-targeter
CH_ENEMY_BUILDING = 7
CH_ENEMY_COUNT = CE.CH_COUNT + 1
CH_ENEMY_SPEED = CE.CH_SPEED + 1
CH_ENEMY_DPS = CE.CH_DPS + 1
CH_ENEMY_FLYING = CE.CH_FLYING + 1

# River start edge, derived: getOwnHalfMaxY() is getRiverStart() -
# OWN_HALF_RIVER_BUFFER, and only the buffer is unbound.
_OWN_HALF_RIVER_BUFFER = 0.5   # GameManager.h
_probe = CE(list(range(8)), list(range(8)), 100)
RIVER_Y = _probe.get_own_half_max_y() + _OWN_HALF_RIVER_BUFFER
del _probe

# From ArenaLayout.
OWN_PRINCESS = ((EC.LEFT_LANE_X, EC.princess_y(0)),
                (EC.RIGHT_LANE_X, EC.princess_y(0)))
OWN_KING = (EC.BOARD_CENTER_X, EC.king_y(0))

# Fireball. Id and damage come from `rewards.weights`; the radius is typed here
# with its header.
FIREBALL_ID = W.FIREBALL_CARD_ID
FIREBALL_RADIUS = 2.5         # CardRegistry.h: spell(7, ..., 2.5f, 689, 10, 'O')
FIREBALL_DAMAGE = W.FIREBALL_DAMAGE
FIREBALL_DELAY_TICKS = 10
CANNON_ID = 25
CANNON_RANGE = 5.5            # CardRegistry.h: building(25, ..., 5.5f, 202, 10)


def spatial(obs):
    """(N_CH, 34, 18) view of the spatial half of a flat observation."""
    a = np.asarray(obs, dtype=np.float32)
    return a[:SPATIAL].reshape(N_CH, BOARD_H, BOARD_W)


def enemy_hp_map(obs):
    """Approximate enemy troop HP per cell, in absolute HP.

    Channels 4-6 overwrite rather than accumulate, so a stacked cell
    under-reports; multiplying by CH_COUNT (which accumulates) recovers most of
    it.
    """
    sp = spatial(obs)
    hp = sp[list(CH_ENEMY_TROOP)].sum(axis=0) * MAX_TROOP_HP
    count = np.maximum(1.0, sp[CH_ENEMY_COUNT] * MAX_CELL_UNITS)
    return hp * count


def enemy_speed_map(obs):
    """Enemy move speed per cell in tiles/tick (denormalized)."""
    return spatial(obs)[CH_ENEMY_SPEED] * MAX_UNIT_SPEED


def advance(hp, speed, ticks):
    """Shift enemy mass `ticks` ticks toward our towers (decreasing y in the
    team-0 frame).

    Mass is split between the two rows it lands between, so sub-tile movement
    is not rounded away. First-order: ignores pathing and retargeting.
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
    """Cell offsets within `radius`, as a tuple.

    Memoized over the few radii this module uses; a tuple because every caller
    shares the same object.
    """
    r = int(np.ceil(radius))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Units are summarized at their cell centre (the encoder truncates
            # positions into cells); the aim point is the integer cell the
            # engine receives.
            if np.hypot(dx + 0.5, dy + 0.5) <= radius:
                offs.append((dy, dx))
    return tuple(offs)


def spell_catch_map(obs, radius=FIREBALL_RADIUS, lead_ticks=0,
                    damage=FIREBALL_DAMAGE):
    """(34, 18) map of the enemy value a spell aimed at each cell would catch.

    Each unit contributes at most `damage`: overkill is wasted, and uncapped
    the map would chase raw HP (a Giant) rather than the clump the spell kills.

    `lead_ticks` defaults to 0 on purpose. Leading the target by the cast delay
    measured worse: the blast radius is larger than a second of movement, and
    engaged units standing still get led out of it. Weighting by what the spell
    kills instead of damages measured the same.
    """
    hp = enemy_hp_map(obs)
    count = np.maximum(1.0, spatial(obs)[CH_ENEMY_COUNT] * MAX_CELL_UNITS)
    if lead_ticks:
        hp = advance(hp, enemy_speed_map(obs), lead_ticks)
    effective = np.minimum(hp, damage * count)

    out = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
    ys, xs = np.nonzero(effective)
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
    """(x, y, caught_hp): where to aim, and what it is worth.

    `legal` is an optional (34*18,) bool mask; pass the net's `placement_mask`
    so the advisor never proposes a cell the engine refuses.
    """
    m = spell_catch_map(obs, radius, lead_ticks, damage)
    flat = m.reshape(-1).copy()
    if legal is not None:
        flat[~np.asarray(legal, dtype=bool).reshape(-1)] = -1.0
    i = int(np.argmax(flat))
    return float(i % BOARD_W), float(i // BOARD_W), float(flat[i])


GIANT_ID = 2
# Our side of the river, so a win condition placed here crosses at once.
BRIDGE_ROW = 15
# Bridge cells, truncated from the engine's bridge centres as cell_to_xy does
# (never round()): a bridge centred on 2.5 spans cells 2 and 3.
BRIDGE_XS = (int(EC.LEFT_BRIDGE_X), int(EC.RIGHT_BRIDGE_X))


def best_giant_cell(obs, legal=None):
    """(x, y, 0.0): where to commit the win condition: the bridge, on the lane the
    enemy defends less.

    Measured by injecting a Giant and scoring enemy tower damage over 600
    ticks: the bridge on the weaker lane beat the policy's own back-row cell by
    ~160x and a random legal cell by ~5x.
    """
    hp = enemy_hp_map(obs)
    left, right = hp[:, :BOARD_W // 2].sum(), hp[:, BOARD_W // 2:].sum()
    # Away from the enemy's mass; ties go left.
    x = BRIDGE_XS[0] if right >= left else BRIDGE_XS[1]
    y = BRIDGE_ROW
    if legal is not None:
        flat = np.asarray(legal, bool).reshape(-1)
        if not flat[y * BOARD_W + x]:
            # Walk along the row to the nearest legal column rather than emit a
            # cell the engine refuses.
            for dx in range(1, BOARD_W):
                for cand in (x - dx, x + dx):
                    if 0 <= cand < BOARD_W and flat[y * BOARD_W + cand]:
                        return float(cand), float(y), 0.0
    return float(x), float(y), 0.0


# --- Hog Rider: the 2.6 win condition ---
# Hardcoded (CardRegistry.h: id 15, cost 4) because this module must not import
# gym_wrapper; test_hog_id_matches_the_registry pins it.
HOG_ID = 15
HOG_COST = 4.0

# Elixir kept back after paying for the Hog. 0: a cycle deck rarely banks
# enough for a larger reserve (a 3-elixir reserve opened the gate on 0 of 542
# states), and HOG_MAX_THREAT already covers the risk it guarded.
HOG_DEFENSIVE_RESERVE = 0.0

# Enemy HP already on our half (absolute) above which we defend instead of
# committing: admits a clear board or one small unit being handled (a Musketeer
# is ~600 HP), refuses a real push.
HOG_MAX_THREAT = 800.0

# Estimated opponent elixir above which we do not commit: at 7+ they can answer
# and counter-push.
HOG_MAX_OPP_ELIXIR = 7.0

# ClashEnv.h values, not bound.
ELIXIR_REGEN_RATE = 0.035
STARTING_ELIXIR = 5.0
MAX_ELIXIR = 10.0
TRAINING_MAX_TICKS = 3600.0


# Forward offsets into the extra scalars: elapsed time / max_ticks, our and the
# opponent's cumulative spend / MAX_MATCH_ELIXIR, then six tower HPs.
_S = EC.EXTRA_SCALARS_START
IDX_ELAPSED = _S + 0
IDX_OWN_SPEND = _S + 1
IDX_OPP_SPEND = _S + 2


def elapsed_ticks(obs):
    """Match time in ticks, from the extra-scalar tail."""
    return float(np.asarray(obs, dtype=np.float32)[IDX_ELAPSED]) * TRAINING_MAX_TICKS


def opp_elixir_estimate(obs, multiplier=1.0):
    """The opponent's current elixir: start + regen * time - observed cumulative
    spend, clamped.

    The observation carries no elixir multiplier, so at the default 1.0 this
    underestimates a boosted opponent and the gate opens too often;
    HOG_MAX_OPP_ELIXIR is set well below the cap to compensate.
    """
    a = np.asarray(obs, dtype=np.float32)
    spent = float(a[IDX_OPP_SPEND]) * MAX_MATCH_ELIXIR
    gained = STARTING_ELIXIR + ELIXIR_REGEN_RATE * elapsed_ticks(obs) * multiplier
    return float(np.clip(gained - spent, 0.0, MAX_ELIXIR))


def best_hog_cell(obs, legal=None):
    """(x, y, 0.0): the bridge row, on the lane the enemy defends less.

    From y=15 the Hog crosses at once, minimising the opponent's reaction time;
    placed deeper it walks our whole half first. The same rule as the Giant's,
    and engine-scored separately.
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
    """Is now the moment to send the win condition? Timing, not placement.

    All three must hold:
      1. Our half is clear: committing into a push leaves too little to defend.
      2. We can afford it (plus HOG_DEFENSIVE_RESERVE).
      3. They are not banked: a full opponent answers and counter-pushes.

    False makes `hog_advice` emit no target, so that row falls back to the
    entropy term; always returning a cell would teach a constant.
    """
    # No enemy anywhere: the lane choice carries no information and the rule
    # would emit a fixed cell, which distils into a constant.
    if float(enemy_hp_map(obs).sum()) <= 0.0:
        return False
    if threat_level(obs) > HOG_MAX_THREAT:
        return False
    # The walking win condition's own cost (a Royal Giant is 6); None means the
    # Hog.
    if own_elixir(obs) < (HOG_COST if cost is None else float(cost)) + HOG_DEFENSIVE_RESERVE:
        return False
    if opp_elixir_estimate(obs, multiplier) > HOG_MAX_OPP_ELIXIR:
        return False
    return True


def hog_advice(obs, legal=None, multiplier=1.0, cost=None):
    """(x, y) to commit the win condition now, or None."""
    if not hog_should_commit(obs, multiplier, cost):
        return None
    x, y, _rank = best_hog_cell(obs, legal)
    return x, y


def threat_map(obs):
    """Enemy HP already on our half: what we must answer."""
    hp = enemy_hp_map(obs)
    out = hp.copy()
    out[int(np.ceil(RIVER_Y)):] = 0.0
    return out


def threat_level(obs):
    """Scalar: total enemy troop HP on our half."""
    return float(threat_map(obs).sum())


def air_siege_map(obs):
    """(34, 18) enemy HP that only an anti-air card can answer: flying
    building-targeters (Balloon, Lava Hound).

    Flyers that chase troops (Minions, Baby Dragon) are excluded, since a
    ground unit still distracts them.
    """
    sp = spatial(obs)
    bt = sp[CH_ENEMY_TROOP[2]] * MAX_TROOP_HP
    return np.where(sp[CH_ENEMY_FLYING] > 0.0, bt, 0.0).astype(np.float32)


#: Enemy HP on our half above which the deck-coverage floor may push: the same
#: threshold as SolvencyGate's `threat_hp` (above a lone Minion, just under a
#: Musketeer).
DECK_COVERAGE_THREAT_HP = 400.0


def threat_level_batch(obs):
    """`threat_level` over a (B, obs_dim) torch batch, in absolute enemy HP.

    For the PPO update, which cannot afford a numpy round-trip per row. Mirrors
    the scalar version step for step;
    `tests/test_threat_gated_deck_coverage.py` pins the two together.
    """
    import torch
    sp = obs[:, :SPATIAL].reshape(-1, N_CH, BOARD_H, BOARD_W)
    hp = sp[:, list(CH_ENEMY_TROOP)].sum(dim=1) * MAX_TROOP_HP
    count = torch.clamp(sp[:, CH_ENEMY_COUNT] * MAX_CELL_UNITS, min=1.0)
    per_cell = hp * count
    per_cell = per_cell[:, :int(np.ceil(RIVER_Y)), :]
    return per_cell.sum(dim=(1, 2))


def threat_lane(obs):
    """-1 left, +1 right, 0 none: which side the push is on.

    Counts enemy mass up to six rows past the river, not just what has crossed:
    a building placed once the push is on our side is too late.
    """
    hp = enemy_hp_map(obs)
    approaching = hp[: int(RIVER_Y) + 6]
    left = approaching[:, : BOARD_W // 2].sum()
    right = approaching[:, BOARD_W // 2:].sum()
    if left <= 0 and right <= 0:
        return 0
    return -1 if left > right else 1


def own_elixir(obs):
    """Our current elixir, from the scalars (ClashEnv divides by 10)."""
    return float(np.asarray(obs, dtype=np.float32)[SPATIAL]) * 10.0


class SolvencyGate:
    """Refuse to spend below a reserve while nothing is attacking.

    For an agent that spends itself broke and cannot answer a push. A hard gate
    rather than shaping: the potential-based solvency term cannot change the
    optimum and measured no effect.

    Conditioned on threat: under attack it opens completely, since spending to
    zero can be right. The reserve is `min(reserve, opponent elixir)`: a flat
    reserve blocks exactly the spends that build a push (it halved tower damage
    dealt), and a broke opponent cannot punish a commitment.
    """

    def __init__(self, reserve=4.0, threat_hp=400.0):
        # The cost of a dedicated defensive answer; below it nothing that
        # matters can be answered.
        self.reserve = reserve
        # Enemy HP on our half above which the gate opens: just under a
        # Musketeer (721), above a lone Minion (230).
        self.threat_hp = threat_hp

    def effective_reserve(self, opp_elixir=None):
        if opp_elixir is None:
            return self.reserve
        return min(self.reserve, max(0.0, float(opp_elixir)))

    def allows(self, obs, cost, opp_elixir=None):
        """May a card of `cost` be played from this state?"""
        if threat_map(obs).sum() >= self.threat_hp:
            return True                      # under attack: spend freely
        return own_elixir(obs) - cost >= self.effective_reserve(opp_elixir)

    def mask(self, obs, hand_costs, opp_elixir=None):
        """Bool list over hand slots + no-op; the caller ANDs it with
        affordability.
        """
        return [self.allows(obs, c, opp_elixir) for c in hand_costs] + [True]


class TacticalOverride:
    """Deterministic tactical plays layered on a policy, solvency-gated.

    Ungated, forced Cannons and Fireballs bankrupted an already-broke agent and
    lost to the control. So an override only spends what leaves `reserve` in
    the bar, and the Cannon is rate-limited to about its lifetime. Thresholds
    come from card stats, not win rate: Fireball needs a 3-cost squad's HP in
    the blast (Minions 3 x 230), the Cannon a Musketeer (721) approaching.
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
        """Return (slot, x, y); `default` is the policy's own greedy action.

        `hand` (card id per slot, -1 empty) and `affordable` must come from the
        observation being decided on: the hand rotates as soon as a card is
        played.
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
    """The per-cell score `best_building_cell` takes the argmax of.

    A better supervision target than the argmax: coverage forms large exact
    plateaus, and argmax picks a plateau's top-left cell, which jumps as the
    plateau shifts while the advisor is indifferent across it.

    Returns (BOARD_H, BOARD_W) float32, -inf on illegal or across-river cells.
    """
    return _building_score(obs, legal, rng_range)[0]


def best_building_cell(obs, legal=None, rng_range=CANNON_RANGE):
    """(x, y, score): where a defensive building actually defends.

    Scores each legal cell by the approaching enemy HP it brings into range,
    discounting cells behind our Princess Towers (reached too late) and away
    from the threatened lane (a building cannot move). With no threat, the
    classic pocket in front of the King.
    """
    score, kind = _building_score(obs, legal, rng_range)
    if kind == "none":
        return float(OWN_KING[0]), 9.0, 0.0
    i = int(np.argmax(score))
    value = 0.0 if kind == "pocket" else float(score.reshape(-1)[i])
    return float(i % BOARD_W), float(i // BOARD_W), value


def _building_score(obs, legal=None, rng_range=CANNON_RANGE):
    """(score map, kind), kind in 'coverage', 'pocket', 'none'. Shared by
    `best_building_cell` and `building_score_map`, so the played cell and the
    distilled surface agree.
    """
    hp = enemy_hp_map(obs)
    # Score where the push will be in ~2 s.
    future = advance(hp, enemy_speed_map(obs), 20)
    lane = threat_lane(obs)

    # Scatter from each occupied enemy cell onto the cells that reach it:
    # O(n_enemy * disc) instead of O(612 * n_enemy).
    cover = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
    offs = _disc_offsets(rng_range)
    ys, xs = np.nonzero(future)
    for ey, ex in zip(ys, xs):
        v = future[ey, ex]
        for dy, dx in offs:
            cy, cx = ey - dy, ex - dx
            if 0 <= cy < BOARD_H and 0 <= cx < BOARD_W:
                cover[cy, cx] += v

    # Forward of our Princess Towers is worth more, and a building cannot
    # chase, so the threatened lane counts more.
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
        # Nothing approaching: the classic pocket, central and in front of the
        # King.
        gy, gx = np.mgrid[0:BOARD_H, 0:BOARD_W]
        pocket = -(np.abs(gx - OWN_KING[0]) + np.abs(gy - 9.0)).astype(np.float32)
        pocket[int(np.ceil(RIVER_Y)):] = -np.inf
        if legal is not None:
            pocket = np.where(np.asarray(legal, bool).reshape(BOARD_H, BOARD_W),
                              pocket, -np.inf)
        return pocket, "pocket"

    return score, "coverage"
