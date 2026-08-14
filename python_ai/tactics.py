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
train.py's `mb_placed`): both the actor loss and the placement entropy bonus
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

Every engine constant here is READ from the bindings, never re-typed --
CLAUDE.md's rule about second copies, which this project has paid for twice.
"""
import numpy as np

import clash_royale_env as E

CE = E.ClashRoyaleEnv

# --- engine geometry / normalizers, pulled live -----------------------------
BOARD_H = CE.BOARD_HEIGHT
BOARD_W = CE.BOARD_WIDTH
N_CH = CE.NUM_CHANNELS
SPATIAL = N_CH * BOARD_H * BOARD_W
MAX_TROOP_HP = CE.MAX_TROOP_HP
MAX_CELL_UNITS = 5.0          # ClashEnv.h MAX_CELL_UNITS -- not bound; see below
MAX_UNIT_SPEED = 1.5          # ClashEnv.h MAX_UNIT_SPEED -- not bound; see below
# MAX_CELL_UNITS and MAX_UNIT_SPEED are the two normalizers ClashEnv.h uses that
# pybind does NOT expose (unlike MAX_TROOP_HP / MAX_BUILDING_HP / CH_*). They are
# hardcoded here with this comment naming the header, which is exactly the
# fallback CLAUDE.md prescribes for values that are not derivable. If either
# changes in ClashEnv.h, `test_tactics.py::test_normalizers_match_header` fails.

# Enemy-side channel indices (side offset 1 = enemy, see ClashEnv.h).
CH_ENEMY_TROOP = (4, 5, 6)    # melee / ranged / building-targeter
CH_ENEMY_BUILDING = 7
CH_ENEMY_COUNT = CE.CH_COUNT + 1
CH_ENEMY_SPEED = CE.CH_SPEED + 1
CH_ENEMY_DPS = CE.CH_DPS + 1

RIVER_Y = 15.5
OWN_PRINCESS = ((4.0, 6.0), (14.0, 6.0))
OWN_KING = (9.0, 2.5)

# Fireball, read from the registry.
FIREBALL_ID = 7
FIREBALL_RADIUS = 2.5         # CardRegistry.h:752 spell(7,...,2.5f,689,10,'O')
FIREBALL_DAMAGE = 689.0
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
    and target re-acquisition. `test_tactics.py` measures it against the
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


def _disc_offsets(radius):
    r = int(np.ceil(radius))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Units are summarized at their CELL CENTRE, because the encoder
            # truncates a float position into a cell; the aim point is the
            # integer cell coordinate the engine actually receives.
            if np.hypot(dx + 0.5, dy + 0.5) <= radius:
                offs.append((dy, dx))
    return offs


_FIREBALL_DISC = _disc_offsets(FIREBALL_RADIUS)


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
    for y, x in zip(ys, xs):
        v = effective[y, x]
        for dy, dx in _disc_offsets(radius):
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
BRIDGE_XS = (4, 14)          # the two bridge columns, Board geometry in CLAUDE.md


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


def threat_map(obs):
    """Enemy HP already on our half, the part we actually have to answer."""
    hp = enemy_hp_map(obs)
    out = hp.copy()
    out[int(np.ceil(RIVER_Y)):] = 0.0
    return out


def threat_level(obs):
    """Scalar: total enemy troop HP on our half."""
    return float(threat_map(obs).sum())


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
        return float(OWN_KING[0]), 9.0, 0.0
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
        i = int(np.argmax(pocket))
        return float(i % BOARD_W), float(i // BOARD_W), 0.0

    i = int(np.argmax(score))
    return float(i % BOARD_W), float(i // BOARD_W), float(score.reshape(-1)[i])
