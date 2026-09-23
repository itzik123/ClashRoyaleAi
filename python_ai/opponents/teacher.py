"""The utility-search teacher: phase 1's opponent, at a symmetric 1.0x economy.

Difficulty is competence (lookahead, candidate width, epsilon;
`TEACHER_STAGES`), never an elixir handicap, which inverts which strategies
pay. Hand-written rather than the neural search, which scores with the net's
own critic and so is no expert at random init.

Rules propose, simulation ranks. `tactics.py`'s rules supply a few cells per
card; each candidate, including short two-card sequences, is rolled forward on
`env.snapshot()` and scored against the no-op rollout, so a score is the
marginal value of playing and holding scores exactly 0.0. Elixir spent is still
charged (`w_cost`): the baseline removes what would happen anyway, not the
cost. `positional_advantage` stands in for a value bootstrap, since a win
condition at the bridge deals no damage within a few seconds; without it the
teacher turtles.

Positions are read from the observation, as in tactics.py, so behaviour is the
same on a real screen; `env` is used only for snapshots, legality and
hand/elixir.

Side-agnostic: it works in its own observation frame and stepSelfPlay mirrors
team 1's action back. But `is_valid_placement` takes absolute y for both teams,
and `to_absolute_y` is the one conversion
(test_teacher_candidates_are_all_legal_for_either_team); getting it wrong makes
team 1 silently never play.
"""
import functools
import itertools

import numpy as np

import clash_royale_env as E
from python_ai import engine_constants as EC
from python_ai.advisors import card_probes, tactics

CE = E.ClashRoyaleEnv

# Rollouts need the observation-free step binding. Fail at import on a stale
# .pyd rather than fall back and silently lose the speedup.
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

# Ally troop channels (0-2), the mirror of tactics.py's enemy side.
CH_ALLY_TROOP = (0, 1, 2)
CH_ALLY_COUNT = CE.CH_COUNT           # ally is the base index, enemy is base+1

# Above this the bar discards income, so spending is nearly free
# (W_ELIXIR_OVERFLOW's threshold).
ELIXIR_OVERFLOW_AT = 9.0

# Where effective_play_margin's taper starts: mid-bar. Tapering from the
# overflow line kept the full margin up to 9 elixir, and the teacher sat on its
# bar while losing towers.
MARGIN_TAPER_START = 6.0


# --- card roles, derived from the engine ---
@functools.lru_cache(maxsize=64)
def _card_table_cached(deck_key):
    return _card_table_uncached(list(deck_key))


def card_peak_hp(deck):
    """{card_id: single-body HP} for the deck's troops, from the engine.

    The peak cell, not the sum: the question is how much one body absorbs.
    Spells and buildings are absent.
    """
    return dict(_card_table_cached(tuple(deck))["hp"])


def tank_id(deck):
    """The card that goes in front of the win condition: the highest-HP troop that
    is not the win condition, or None.
    """
    table = _card_table_cached(tuple(deck))
    roles, hp = table["roles"], table["hp"]
    troops = [c for c in hp if roles.get(c) != "wincon"]
    if not troops:
        return None
    return max(troops, key=lambda c: (hp[c], -c))


def card_roles(deck):
    """{card_id: "wincon"|"spell"|"building"|"ranged"|"melee"} for one deck.

    The registry exposes no archetype, so each troop is injected and classified
    by which type channel lights up (0 melee, 1 ranged, 2 building-targeter).
    The win condition comes from `resolve_win_condition`. Memoized per deck;
    returns a copy.
    """
    return dict(_card_table_cached(tuple(deck))["roles"])


# Long enough for a siege building to deploy, acquire and out-range the tower.
TOWER_THREAT_PROBE_TICKS = 1200
#: Probe deck; the card under test is injected, so the contents are irrelevant.
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
    """The furthest-forward row a building may occupy (the siege row), from the
    engine.
    """
    return int(_probe_env().get_own_half_max_y())


@functools.lru_cache(maxsize=256)
def siege_reach(card_id):
    """Enemy tower HP a building takes from the furthest-forward legal row.

    Answers whether the card has any route to a tower at all: a Mortar damages
    the tower from 34 of its 170 legal cells (the front rows) and from none of
    the rest. Uses step_self_play, since plain step runs the defending
    heuristic, and scores tower HP actually lost (get_tower_damage_dealt is
    nonzero on an empty board).
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
    """Peak friendly bodies this spell puts on the board.

    The discriminator for a spell win condition is the spawn, not the damage:
    every direct spell hurts a tower it is cast on, but only spells like Goblin
    Barrel deliver bodies.
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
    """(radius, damage) to aim `card_id` with: the card's own, measured by
    card_probes.

    A spell the probe declines (a roller like The Log, whose value is a
    corridor) falls back to Fireball's disc.
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
    """Peak friendly troop bodies a building puts on an empty board.

    Separates siege buildings (0: X-Bow, Mortar), which fire at the tower
    themselves, from spawners (Barbarian Hut, Tombstone, ...), whose bodies
    walk there. Both damage a tower from our half.
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
    """A building that attacks the enemy tower itself from our half: reaches a
    tower and spawns nothing.

    Deploy-anywhere buildings (Goblin Drill) are played beside the tower
    instead. One definition, shared with `card_probes.building_defends`.
    """
    info = E.get_card_info(card_id)
    if not info["is_building"] or info.get("deploy_anywhere", False):
        return False
    return siege_reach(card_id) > 0.0 and building_spawns_bodies(card_id) == 0


#: Short on purpose: over a long window almost anything that walks takes an
#: undefended tower, and the probe stops separating a Hog from an Ice Golem.
WINCON_PROBE_TICKS = 300

#: Minimum tower HP per elixir (alone, in WINCON_PROBE_TICKS) to count as a win
#: condition. It only excludes cards with no route to a tower: real win
#: conditions and cheap tanks overlap (Skeleton Barrel 81 vs Ice Golem 84), so
#: ranking, not the floor, picks the card. A weak best candidate is still used,
#: and validate_deck warns about it.
WINCON_MIN_DAMAGE_PER_ELIXIR = 50.0
#: Below this the resolved win condition is reported as WEAK by validate_deck.
WINCON_WEAK_DAMAGE_PER_ELIXIR = 200.0


def _roll_idle(env, ticks):
    for _ in range(ticks // 10):
        env.step_self_play(HAND_SIZE, 0.0, 0.0, HAND_SIZE, 0.0, 0.0, 10,
                           False, False, False, False)


def _attack_cell(card_id, env):
    """Where the card is played to attack, or None.

    A walker starts on our side of the left bridge, a deploy-anywhere troop
    goes beside the enemy tower, a body-spawning spell lands on it.
    """
    info = E.get_card_info(card_id)
    lane_x = float(int(EC.LEFT_LANE_X))
    tower_y = float(EC.princess_y(1))
    if info["is_spell"]:
        # Only a cell the card can actually be cast on: inject bypasses
        # legality, and a rolling spell is confined to our half and the river.
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
    """Enemy tower HP per elixir this card takes alone, from its attack cell.

    Measured as tower HP actually lost, never ranked by cost. Buildings reuse
    `siege_reach`.
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

    Eligibility is by class (building-targeter, deploy-anywhere, siege
    building, body-spawning spell): a Knight also damages an empty tower, so
    measured damage only ranks within the class. Spawner buildings reach a
    tower only by walking bodies there, like a Knight.
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
    """The deck's win condition, or None.

    The one definition, shared with `gym_wrapper._find_win_condition`, which
    feeds the agent's `W_WIN_CONDITION_DAMAGE` term.
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
    # Ranked by absolute tower damage, then per elixir. Per elixir first would
    # favour the cheapest card, since the probe saturates at one Princess
    # (2534).
    return max(scored, key=lambda t: (t[0], t[1], -t[2]))[2]


def _card_table_uncached(deck):
    """One injection pass per card yields both its role (which type channel lights
    up) and its HP.
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
    # One measured resolver; see resolve_win_condition.
    wincon = resolve_win_condition(deck, set(targeters))
    if wincon is not None:
        roles[wincon] = "wincon"
    return {"roles": roles, "hp": hp}


# --- cycle tracking ---
class CycleTracker:
    """How many plays until a given card returns to hand.

    A played card goes to the back of the queue, so the order cards leave the
    hand is the queue order and our own cycle is exact from watching the hand.
    The initial queue's order is unknown (deck order is assumed until one cycle
    corrects it), which is why `cycle_value` is a scored term rather than a
    gate. Assumes no duplicate card ids.
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


# --- weight profiles ---
# Units are elixir-equivalents (one elixir destroyed = 1.0):
#
#   w_twr / w_def  per point of tower HP (100 HP ~ 1 elixir at 0.01)
#   w_trade        per elixir of value killed / lost
#   w_crown        per net surviving tower
#   w_pos          per unit of positional advantage
#   w_cost         per elixir spent (not absorbed by the no-op baseline)
#   w_cycle        per unit of cycle value
#
# Starting values chosen by reasoning; `prove_teacher.py --sweep` re-selects
# them against win rate and confirms on an independent run.
PROFILES = {
    "aggressive": dict(w_twr=0.015, w_def=0.007, w_trade=0.8, w_crown=5.0,
                       w_pos=25.0, w_cost=1.0, w_cycle=0.40),
    "balanced":   dict(w_twr=0.010, w_def=0.010, w_trade=1.0, w_crown=5.0,
                       w_pos=20.0, w_cost=1.0, w_cycle=0.30),
    "defensive":  dict(w_twr=0.006, w_def=0.016, w_trade=1.2, w_crown=5.0,
                       w_pos=15.0, w_cost=1.0, w_cycle=0.20),
}

# The competence ladder; each rung moves one axis.
#
#   horizon_ticks  rollout lookahead, the main axis. Stops at 10 s: past ~12 s
#                  a mostly no-op rollout stops resembling the game.
#   epsilon        chance of a uniformly random legal action; 0 at the top
#   k_cells        cells proposed per card
#   max_combos     two-card sequences simulated per decision; 0 below
#                  COMBO_MIN_HORIZON_TICKS, where the follow-up is invisible
#   reactive       whether the rollout opponent answers attacks; off low down,
#                  since an episode-0 agent cannot defend
#
# Rung 3 moves k_cells and combos together only because combos are inert below
# it. The old six stages are rungs 0, 1, 4, 6, 8 and 10 (see
# remap_legacy_stage).
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

#: The six-rung table this replaced, by horizon: a saved stage indexes the
#: table live when it was written, so legacy checkpoints are remapped by
#: horizon.
_LEGACY_STAGE_HORIZONS = [0, 10, 30, 50, 70, 100]


def remap_legacy_stage(stage):
    """Old six-rung index -> the eleven-rung index with the same horizon.

    Returns `stage` unchanged if it cannot be a legacy index. Called only from
    `CurriculumManager.load_state_dict`, for unstamped checkpoints.
    """
    stage = int(stage)
    if not 0 <= stage < len(_LEGACY_STAGE_HORIZONS):
        return min(stage, len(TEACHER_STAGES) - 1)
    want = _LEGACY_STAGE_HORIZONS[stage]
    for i, cfg in enumerate(TEACHER_STAGES):
        if cfg["horizon_ticks"] == want:
            return i
    return min(stage, len(TEACHER_STAGES) - 1)


# Own back-half cells for `wincon_mode="cycle"`. Several, because the engine
# may refuse one, which would silently turn "cycle" into "ban".
WINCON_DUD_CELLS = [(2.0, 2.0), (15.0, 2.0), (2.0, 4.0), (15.0, 4.0), (9.0, 5.0)]


# --- multi-card combos ---
# A combo is a sequence of placements across decisions: the teacher returns one
# placement per 10-tick decision, and a 0-tick step places nothing.
COMBO_FOLLOWUP_DELAY_TICKS = 10

# The rollout opponent answers one decision after an attacking placement, the
# fastest a real opponent could. Answering every chunk scored worse at twice
# the cost.
COUNTER_DELAY_TICKS = 10

# The rollout opponent may answer only when the rollout is long enough to also
# see the attack's payoff: the answer lands at +10 ticks, but a Hog needs ~130
# to reach a tower, so a short rollout charges the answer and credits nothing,
# and the teacher freezes against a passive opponent.
COUNTER_MIN_HORIZON_TICKS = 100

#: The counter switches off after this many consecutive decisions (~8 s) in
#: which the real opponent spent nothing, and back on when they play.
#: Unconditional, it froze the top rung against an opponent sitting on a full
#: bar.
COUNTER_PASSIVE_DECISIONS = 8

# Follow-up gaps, searched rather than fixed. The second card is paid from what
# regenerates during the gap, which is what makes a pair affordable at all (the
# bar rarely holds two cards at once). Offered shortest first: the tighter
# escort scores better whenever it is affordable.
COMBO_FOLLOWUP_DELAYS = (10, 30, 50)

# Minimum horizon to propose a combo: a rollout that never plays the second
# card would charge both costs and credit one.
COMBO_MIN_HORIZON_TICKS = 20

# Elixir regenerated across one decision.
ELIXIR_PER_DECISION = tactics.ELIXIR_REGEN_RATE * COMBO_FOLLOWUP_DELAY_TICKS

# Charge (elixir-equivalents) on a spend that would leave a committed plan's
# second card unaffordable while it waits out its gap. Set to outbid a 1-cost
# cycle card's positional value (~1.2). A flat "save toward a push" charge
# measured as doing nothing. `combo_reserve=0.0` disables it.
COMBO_RESERVE = 1.5

#: Every combo family, in offer order; configurable so one can be ablated.
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
    """A sequence of placements the teacher is willing to simulate.

    `.slot/.x/.y/.card_id` describe the first step, the one `act()` returns
    this decision. The empty sequence is the no-op, which scores exactly 0.
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
    """Ally troop HP per cell (absolute): the mirror of `tactics.enemy_hp_map`.
    """
    sp = tactics.spatial(obs)
    hp = sp[list(CH_ALLY_TROOP)].sum(axis=0) * MAX_TROOP_HP
    count = np.maximum(1.0, sp[CH_ALLY_COUNT] * tactics.MAX_CELL_UNITS)
    return hp * count


_ROW_WEIGHT = np.arange(BOARD_H, dtype=np.float32) / float(BOARD_H - 1)


def positional_advantage(obs):
    """Board control: troop HP (in full-troop units) weighted by progress up the
    board, ours minus theirs.

    Stands in for a value bootstrap: a win condition at the bridge deals no
    damage inside a short rollout, so on damage alone the teacher would never
    attack. High y is toward the enemy in the observer's frame.
    """
    ally = ally_hp_map(obs) / MAX_TROOP_HP
    enemy = tactics.enemy_hp_map(obs) / MAX_TROOP_HP
    return float((ally.sum(axis=1) * _ROW_WEIGHT).sum()
                 - (enemy.sum(axis=1) * _ROW_WEIGHT[::-1]).sum())


class UtilityTeacher:
    """A deterministic opponent strong enough to make attacking pay.

    Plays either side: `act(env, obs_own)` returns `(slot, x, y)` in its own
    frame, ready for step_self_play.

    `wincon_mode` exists for `prove_environment.py`:

      "attack"  the win condition goes to a bridge (the real rule)
      "cycle"   it is still played and cycled, but placed in our back half
                where it is a dud
      "ban"     never played

    "cycle" is the falsifier's arm: "ban" also clogs a hand slot, which
    confounds the comparison.
    """

    def __init__(self, deck, team, profile=None, horizon_ticks=30, k_cells=2,
                 epsilon=0.0, seed=None, wincon_mode="attack", max_combos=3,
                 combo_reserve=None, reactive_rollout=True):
        self.deck = list(deck)
        self.team = int(team)
        #: Whether the rollout opponent answers; see `counter_schedule`.
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
        # The second half of a chosen combo, offered at a later decision; see
        # `commit`.
        self.pending = None
        # Ticks until the pending step is due, so it is played at the gap it
        # was scored at.
        self.pending_ticks = 0
        # Diagnostic: the candidate kind the last `act` chose.
        self.last_kind = "noop"
        # A profile name from PROFILES or an explicit weight dict (for sweeps),
        # copied so a shared dict is never mutated.
        self._fixed_profile = (dict(profile) if isinstance(profile, dict)
                               else profile)
        self.rng = np.random.default_rng(seed)
        self.cycle = CycleTracker(self.deck)
        self.profile = self._resolve_profile(profile)
        self.lane_bias = 0
        # A play must beat holding by more than this: the teacher's one economy
        # control. 3.0 was selected by sweep and confirmed on an independent
        # run; lower dumps elixir, higher wastes income for no gain. Lowering
        # w_pos instead does not work: it prunes escorted pushes before cheap
        # cards.
        self.play_margin = 3.0

    # -- lifecycle ---------------------------------------------------------
    def reset(self, rng=None):
        """New match. Redraws the profile and lane bias unless a profile was
        pinned, so the opponent cannot be memorised.
        """
        if rng is not None:
            self.rng = rng
        self.cycle.reset()
        self.pending = None
        self.pending_ticks = 0
        #: Opponent reactivity, read from the engine each decision; see
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
        """A name, an explicit weight set, or None ("balanced") -> a copy, so
        PROFILES is never mutated.
        """
        if isinstance(profile, dict):
            return dict(profile)
        return dict(PROFILES[profile] if profile else PROFILES["balanced"])

    def set_deck(self, deck):
        """Point the teacher at a different deck, rebuilding its role table and
        cycle tracker.

        The opponent deck changes per episode under the deck pool. `card_roles`
        is memoized, so a deck already seen costs nothing.
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
        """Apply one rung of TEACHER_STAGES; never touches elixir."""
        cfg = TEACHER_STAGES[int(np.clip(stage, 0, len(TEACHER_STAGES) - 1))]
        self.horizon_ticks = cfg["horizon_ticks"]
        self.epsilon = cfg["epsilon"]
        self.k_cells = cfg["k_cells"]
        self.max_combos = cfg["max_combos"]
        self.reactive_rollout = cfg["reactive"]

    # -- frames ------------------------------------------------------------
    def to_absolute_y(self, y_own):
        """Own-frame y -> the absolute y `is_valid_placement` expects.

        step_self_play mirrors team 1's y itself, but is_valid_placement does
        not; skipping this makes every team-1 candidate illegal.
        """
        return float(y_own) if self.team == 0 else MIRROR_Y - float(y_own)

    # -- candidate generation ---------------------------------------------
    def candidates(self, env, obs_own, elixir=None):
        """Every candidate worth simulating this step, plus the no-op.

        Cells come from `tactics.py`'s rules and are filtered by the engine's
        own legality predicate.
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

        # The pending second half of last decision's plan, re-scored like any
        # candidate; see `commit`.
        follow = self._followup_candidate(env, hand, elixir)
        if follow is not None:
            out.append(follow)

        out.extend(self._legal_combos(env, obs_own, hand, elixir))
        return out

    # -- the plan ----------------------------------------------------------
    def commit(self, cand):
        """Record a chosen combo's second step for a later decision.

        It is offered then and re-scored, not played blindly: by then the first
        card is on the board, so a plain rollout of the second already sees the
        escort, and a blind play could walk into whatever the board became.
        What the plan carries is the exact cell and lane.
        """
        if not cand.is_combo:
            # Not a clear: a plan waiting out its gap survives single plays
            # made meanwhile.
            return
        self.pending = cand.steps[1]
        self.pending_ticks = self.pending.delay_ticks

    def tick_plan(self):
        """One decision passes; returns True while a plan is still waiting.

        Called before `candidates`, so a plan is offered on the decision it
        comes due.
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

        A handful of named tactics rather than all pairs x cells, since each
        candidate costs a whole rollout. Families are taken round-robin under
        `max_combos`.
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
        """Every step must be legal, or the combo is charged for two cards and
        plays one.
        """
        if cand.steps[0].slot == cand.steps[1].slot:
            # Playing a slot refills it, so the second step would place a
            # different card.
            return False
        return all(env.is_valid_placement(st.card_id, st.x,
                                          self.to_absolute_y(st.y), self.team)
                   for st in cand.steps)

    def _gaps(self):
        """The follow-up gaps this rung's rollout can see, shortest first."""
        return [d for d in COMBO_FOLLOWUP_DELAYS
                if d + COMBO_FOLLOWUP_DELAY_TICKS <= self.horizon_ticks]

    def _pairs(self, slots, first, second, cell1, cell2, kind, elixir):
        """Every affordable gap for one (first -> second) tactic.

        The second card is paid from what has regenerated by then, capped at
        the engine's maximum.
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
        """Tank first, win condition behind it in the same lane, a searched gap
        later.

        The tank absorbs its deploy time first so the tower locks onto it; win
        condition first is the naked push the engine punishes, so that order is
        not offered. Two variants, on the tank's cell and one tile behind; the
        simulator picks.
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
        """A cheap escort: the cheapest body in front of the win condition,
        affordable far more often than the tank. Skipped when the cheapest body
        is the tank.
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
        """A building to hold the push, then a body to kill it.

        The building goes first, since it must deploy and start pulling early.
        The second variant uses the centre-pull cell in front of our King.
        """
        if tactics.threat_level(obs) <= 0.0:
            return []
        buildings = [c for c in slots if self.roles.get(c) == "building"]
        bodies = [c for c in slots if self.roles.get(c) in ("melee", "ranged")]
        if not buildings or not bodies:
            return []
        b = min(buildings, key=lambda c: E.get_card_info(c)["cost"])
        # The cheapest body; any play advances the cycle.
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
        """Two cheap bodies onto the same threat, one gap apart, the second a tile
        back so one splash cannot catch both.

        Affordable where the expensive families are not. The win condition is
        excluded.
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
        """Clear the lane with a spell, then push into it; only when the spell's
        catch map catches something.
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
        """The win condition, then a spell on defenders already in its lane.

        A predictive spell on defenders not yet played cannot score, because
        the rollout opponent never plays them; this proposes only what the
        rollout can value.
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
            # Buildings go to the siege row, except deploy-anywhere ones
            # (Goblin Drill), which go beside the tower like a Miner.
            if info["is_building"] and not info.get("deploy_anywhere", False):
                return self._siege_cells(obs)[:k]
            if info["is_spell"]:
                return self._tower_cells(obs)[:k]
            if info.get("deploy_anywhere", False):
                # A Miner's value is that it skips the bridge.
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
                # The centre pull: in front of the King, between the Princess
                # Towers, where a Cannon drags a lane-committed win condition
                # off its path.
                cells.append((float(tactics.OWN_KING[0]), 11.0))
            return cells[:k]

        # Plain troops: meet the deepest threat as far forward as legal, then
        # support our most advanced unit. An anti-air card meets a flying
        # building-targeter first, since nothing else can.
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
            # Nothing to answer or support: offer the bridge on this match's
            # lane; the no-op baseline decides whether it is worth it.
            cells.append((float(tactics.BRIDGE_XS[self.lane_bias]),
                          float(tactics.BRIDGE_ROW)))
        return cells[:k]

    def _siege_cells(self, obs):
        """Where a Mortar or X-Bow must stand to threaten anything: the
        furthest-forward row (one row back is worth nothing).

        Centre first, which reaches both Princess Towers, then the threatened
        lane, then the other.
        """
        row = float(own_half_max_row())
        bx, _by, _ = tactics.best_hog_cell(obs)
        cells = [(float(int(EC.BOARD_CENTER_X)), row), (float(int(bx)), row)]
        other = (tactics.BRIDGE_XS[1] if int(bx) == tactics.BRIDGE_XS[0]
                 else tactics.BRIDGE_XS[0])
        cells.append((float(other), row))
        return cells

    def _beside_tower_cells(self, card_id, obs):
        """Legal cells just in front of each enemy Princess, weaker tower first,
        for a deploy-anywhere troop. Steps toward the river until the engine
        accepts the cell.
        """
        out = []
        for x, y in self._tower_cells(obs):
            for dy in (3.0, 4.0, 2.0, 5.0):
                if self.env_valid(card_id, x, y - dy):
                    out.append((x, y - dy))
                    break
        return out or self._tower_cells(obs)

    def env_valid(self, card_id, x, y):
        """`is_valid_placement` for an own-frame cell, asked as team 0 on a shared
        probe board.

        Legality is mirror-symmetric, and independent of board state on the
        enemy half.
        """
        return _probe_env_cached().is_valid_placement(card_id, float(x), float(y), 0)

    def _tower_cells(self, obs):
        """Where a body-spawning spell lands: on an enemy Princess Tower, weaker
        first (finish one tower rather than chip two).
        """
        y = float(EC.princess_y(1))
        lanes = [(float(int(EC.LEFT_LANE_X)), y, 1), (float(int(EC.RIGHT_LANE_X)), y, 2)]
        lanes.sort(key=lambda c: self._enemy_tower_fraction(obs, c[2]))
        return [(x, yy) for x, yy, _slot in lanes]

    def _enemy_tower_fraction(self, obs, slot):
        """Enemy Princess Tower HP (normalised) from the observation's extra
        scalars (6-8: king/left/right), read forward from EXTRA_SCALARS_START.
        """
        return float(obs[EC.EXTRA_SCALARS_START + 6 + slot])

    def _top_spell_cells(self, obs, k, card_id):
        """Top-k cells of this spell's catch map, with non-maximum suppression so
        the second candidate is a different decision. Uses the card's own
        geometry.
        """
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
        """(x, y) of the deepest flying building-targeter, or (None, None): the
        threat a ground-only card cannot touch.
        """
        m = tactics.air_siege_map(obs)
        if m.sum() <= 0.0:
            return None, None
        ys, xs = np.nonzero(m)
        deepest = int(np.argmin(ys))
        return float(xs[deepest]), float(ys[deepest])

    def _deepest_threat(self, obs):
        """(x, y) of the enemy mass furthest toward our towers, or (None, None).

        Scans the whole board, so a push is noticed before it crosses.
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
        """`play_margin`, tapered to zero between MARGIN_TAPER_START and a full
        bar.

        A fixed margin looks strong against an active opponent but freezes the
        teacher against a passive one, and an episode-0 agent is passive.
        `score`'s overflow relief stays at 9.0: it refunds the cost charge only
        when income is actually being thrown away.
        """
        relief = max(0.0, float(elixir) - MARGIN_TAPER_START) / (
            10.0 - MARGIN_TAPER_START)
        return float(self.play_margin) * (1.0 - min(1.0, relief))

    def margin_for(self, cand, elixir):
        """The bar this candidate must clear.

        Zero for a plan's due second half: its first half is already paid for,
        so holding is not free. It still has to win the argmax.
        """
        if cand.kind == "followup":
            return 0.0
        return self.effective_play_margin(elixir)

    def rollout_ticks(self):
        """How long every rollout in one decision runs: one length for candidates
        and baseline alike, or the difference would carry extra time as well as
        extra cards.
        """
        return max(COMBO_FOLLOWUP_DELAY_TICKS, self.horizon_ticks)

    def execute_steps(self, s, cand, ticks):
        """Play `cand`'s scheduled placements while advancing `s` by `ticks`.

        In 10-tick chunks (the teacher's real decision granularity), cut short
        to land exactly on each scheduled offset.
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
            # The same advance without building observations nobody reads;
            # rollout_stats reads the one it needs.
            if self.team == 0:
                s.step_self_play_fast(slot, x, y, oslot, ox, oy, nxt - t)
            else:
                s.step_self_play_fast(oslot, ox, oy, slot, x, y, nxt - t)
            t = nxt
        return s

    # -- the reacting opponent ---------------------------------------------
    def counter_schedule(self, cand):
        """{tick: (x, y)}: when and where the rollout opponent answers.

        Open-loop, derived once from our own candidate: closed-loop responders
        measured no better and cost more. Only an attacking placement draws an
        answer; answering defence would penalise it.
        """
        if not self.reactive_rollout:
            return {}
        if self.horizon_ticks < COUNTER_MIN_HORIZON_TICKS:
            return {}
        if getattr(self, "_opp_idle_decisions", 0) >= COUNTER_PASSIVE_DECISIONS:
            # The real opponent is not answering anything; do not score attacks
            # against an answer they are not giving.
            return {}
        out = {}
        for st in cand.steps:
            if st.y < tactics.BRIDGE_ROW - 1:
                continue
            out[int(st.delay_ticks) + COUNTER_DELAY_TICKS] = (st.x, st.y)
        return out

    def _observe_opponent_spend(self, env):
        """Count consecutive decisions in which the opponent spent no elixir, from
        the engine's cumulative spend (a refused play spends nothing).
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
        # Our attacking cell in their frame, clamped to their own half.
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
        """Roll one candidate forward on a snapshot and read the engine.

        Apart from the counter, both sides no-op after the plays, as in search;
        the horizons here stay inside the ~12 s where that still resembles the
        game.
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
        # Opportunity cost, relieved above the overflow line where holding
        # wastes income.
        overflow_relief = max(0.0, elixir - ELIXIR_OVERFLOW_AT) / (
            10.0 - ELIXIR_OVERFLOW_AT)
        u -= p["w_cost"] * cost * (1.0 - min(1.0, overflow_relief))
        u -= self.plan_reserve_penalty(
            cand, obs_own, list(env.get_hand_for_team(self.team)), elixir)
        return float(u)

    def plan_reserve_penalty(self, cand, obs, hand, elixir):
        """Elixir-equivalent charge for spending a committed plan's money.

        A charge, not a gate, and only while a plan waits out its gap. Exempt
        under a real threat (above `tactics.HOG_MAX_THREAT`): holding elixir
        through a push loses the tower. A `> 0.0` threshold was an off switch,
        since something is almost always on our half.
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
        """Total elixir a candidate commits, both cards of a combo; charging one
        would make escorting look free.
        """
        return float(sum(E.get_card_info(c)["cost"] for c in cand.cards))

    def sequence_cycle_value(self, card_ids):
        """Cycle value of a whole sequence, with the queue advancing between
        plays, so a combo is not paid twice for one step of cycle.
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
        """The value of playing `card_id` for cycling back to the win condition:
        plays remaining / cost.

        A scored term, never a gate, so a wrong estimate costs only ranking
        quality.
        """
        if self.wincon_id is None:
            return 0.0
        if card_id == self.wincon_id:
            return 0.0
        d = self.cycle.distance_to(self.wincon_id)
        if d <= 0:
            return 0.0
        cost = float(E.get_card_info(card_id)["cost"])
        # Any play advances the queue by one, so a cheap card pulls the win
        # condition closer most efficiently.
        return float(d) / max(1.0, cost)

    # -- the decision ------------------------------------------------------
    def ability_flags(self, env, obs_own):
        """(activate slot 1, activate slot 2) for this decision.

        Activate a ready Champion ability when the opponent has a real force on
        the board. Readiness alone would fire the moment a Champion lands.
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
        # Advance the plan clock first; see `tick_plan`.
        waiting = self.tick_plan()
        cands = self.candidates(env, obs_own)
        # A due plan is offered once and then dropped: a stale plan places a
        # card against a board that no longer exists.
        due = self.pending is not None and not waiting

        if self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon:
            # A uniformly random legal action; the no-op stays in the
            # distribution.
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
        """Rung 0 only (horizon 0): no rollouts at all.

        Plays only on a clear reason; a bot that plays on every affordable step
        is the elixir-dumping opponent this replaced.
        """
        threat = tactics.threat_level(obs_own)
        # With a flying building-targeter on our half, an anti-air card
        # outranks everything, and a ground-only card plays only if a ground
        # threat also exists (a Skeleton under a Balloon is a gift).
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
                # Cast only when the catch uses the spell's full damage, from
                # its own map and damage.
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
