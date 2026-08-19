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

import numpy as np

import clash_royale_env as E
import tactics

CE = E.ClashRoyaleEnv

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


# --------------------------------------------------------------------------
# card roles -- derived from the engine, never a hardcoded id list
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _card_roles_cached(deck_key):
    return _card_roles_uncached(list(deck_key))


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

    `_find_win_condition` is deliberately NOT refactored into this. It answers a
    narrower question, it sits in the reward path, and changing it to serve a
    new caller is risk with no benefit.
    """
    return dict(_card_roles_cached(tuple(deck)))


def _card_roles_uncached(deck):
    roles = {}
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
        if 2 in lit:
            targeters.append(cid)
            roles[cid] = "melee"          # provisional; promoted below
        elif 1 in lit:
            roles[cid] = "ranged"
        else:
            roles[cid] = "melee"
    if targeters:
        # More than one building-targeter is normal (2.6 has Hog AND Ice Golem).
        # The win condition is the one that actually threatens a tower, i.e. the
        # most expensive -- the same tiebreak _find_win_condition uses.
        roles[max(targeters, key=lambda c: E.get_card_info(c)["cost"])] = "wincon"
    return roles


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
# Stages 0-1 run with NO search at all, so the early curriculum is free.
TEACHER_STAGES = [
    {"horizon_ticks":  0, "epsilon": 0.30, "k_cells": 1},
    {"horizon_ticks":  0, "epsilon": 0.15, "k_cells": 1},
    {"horizon_ticks": 30, "epsilon": 0.10, "k_cells": 2},
    {"horizon_ticks": 30, "epsilon": 0.05, "k_cells": 2},
    {"horizon_ticks": 60, "epsilon": 0.02, "k_cells": 3},
    {"horizon_ticks": 60, "epsilon": 0.00, "k_cells": 3},
]


# Own-back-half cells used by `wincon_mode="cycle"` -- see UtilityTeacher's
# docstring. Several offered because `candidates()` filters through the engine's
# legality predicate and a single cell could be refused (tower footprint, back-row
# dead zone), which would silently turn "cycle" into "ban" and confound the
# experiment those two modes exist to separate.
WINCON_DUD_CELLS = [(2.0, 2.0), (15.0, 2.0), (2.0, 4.0), (15.0, 4.0), (9.0, 5.0)]


class Candidate:
    """One (slot, cell) the teacher is willing to consider. slot == HAND_SIZE
    is the no-op, which is always present and always scores exactly 0."""

    __slots__ = ("slot", "card_id", "x", "y", "role")

    def __init__(self, slot, card_id, x, y, role):
        self.slot = int(slot)
        self.card_id = int(card_id)
        self.x = float(x)
        self.y = float(y)
        self.role = role

    def __repr__(self):
        if self.slot >= HAND_SIZE:
            return "Candidate(no-op)"
        return (f"Candidate(slot={self.slot}, card={self.card_id}, "
                f"{self.role}, x={self.x:.1f}, y={self.y:.1f})")


NOOP = Candidate(HAND_SIZE, -1, 0.0, 0.0, "noop")


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
                 epsilon=0.0, seed=None, wincon_mode="attack"):
        self.deck = list(deck)
        self.team = int(team)
        self.wincon_mode = wincon_mode
        self.roles = card_roles(self.deck)
        self.wincon_id = next((c for c, r in self.roles.items() if r == "wincon"),
                              None)
        self.horizon_ticks = int(horizon_ticks)
        self.k_cells = int(k_cells)
        self.epsilon = float(epsilon)
        self._fixed_profile = profile
        self.rng = np.random.default_rng(seed)
        self.cycle = CycleTracker(self.deck)
        self.profile = PROFILES[profile] if profile else PROFILES["balanced"]
        self.lane_bias = 0
        # A play must beat holding by more than this. Strictly positive so
        # rollout noise on a dead board cannot talk the bot into dumping.
        self.play_margin = 0.05

    # -- lifecycle ---------------------------------------------------------
    def reset(self, rng=None):
        """New match. Redraws the profile and lane bias unless a profile was
        pinned at construction -- a FULLY deterministic opponent is memorizable,
        which is the single-counter-line version of the echo chamber this whole
        pivot exists to avoid."""
        if rng is not None:
            self.rng = rng
        self.cycle.reset()
        if self._fixed_profile:
            self.profile = PROFILES[self._fixed_profile]
        else:
            names = list(PROFILES)
            self.profile = PROFILES[names[int(self.rng.integers(len(names)))]]
        self.lane_bias = int(self.rng.integers(2))

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
        self.cycle = CycleTracker(self.deck)
        self.cycle.reset()

    def set_stage(self, stage):
        """Apply one rung of TEACHER_STAGES. Difficulty is competence only --
        this never touches elixir."""
        cfg = TEACHER_STAGES[int(np.clip(stage, 0, len(TEACHER_STAGES) - 1))]
        self.horizon_ticks = cfg["horizon_ticks"]
        self.epsilon = cfg["epsilon"]
        self.k_cells = cfg["k_cells"]

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
                    out.append(Candidate(slot, cid, xi, yi, role))
        return out

    def _cells_for(self, role, card_id, obs):
        """The rule layer: 1-3 tactically sensible cells for one card."""
        k = max(1, self.k_cells)
        if role == "wincon":
            if self.wincon_mode == "cycle":
                return list(WINCON_DUD_CELLS)
            x, y, _ = tactics.best_hog_cell(obs)
            cells = [(x, y)]
            if k > 1:
                other = (tactics.BRIDGE_XS[1] if int(x) == tactics.BRIDGE_XS[0]
                         else tactics.BRIDGE_XS[0])
                cells.append((float(other), float(tactics.BRIDGE_ROW)))
            return cells[:k]

        if role == "spell":
            return self._top_spell_cells(obs, k)

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
        # (second) support our own most advanced unit.
        cells = []
        tx, ty = self._deepest_threat(obs)
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

    def _top_spell_cells(self, obs, k):
        """Top-k cells of the catch map with non-maximum suppression, so the
        second candidate is a different DECISION and not the same blast shifted
        one tile."""
        m = tactics.spell_catch_map(obs).copy()
        cells = []
        for _ in range(k):
            i = int(np.argmax(m))
            v = float(m.reshape(-1)[i])
            x, y = float(i % BOARD_W), float(i // BOARD_W)
            cells.append((x, y))
            if v <= 0.0:
                break
            for dy, dx in tactics._disc_offsets(tactics.FIREBALL_RADIUS):
                ay, ax = int(y) + dy, int(x) + dx
                if 0 <= ay < BOARD_H and 0 <= ax < BOARD_W:
                    m[ay, ax] = 0.0
        return cells

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
    def rollout_stats(self, env, cand):
        """Roll one candidate forward on a SNAPSHOT and read the engine.

        Both sides no-op after the play, the same assumption `search_ab_test`
        makes. CLAUDE.md records that past ~12 s of that a rollout "stops
        resembling the game", so the 3-6 s horizons here sit well inside the
        validated regime.
        """
        s = env.snapshot()
        me, opp = self.team, 1 - self.team
        slot = cand.slot if cand.slot < HAND_SIZE else -1
        if self.team == 0:
            s.step_self_play(slot, cand.x, cand.y, -1, 0.0, 0.0, 10)
        else:
            s.step_self_play(-1, 0.0, 0.0, slot, cand.x, cand.y, 10)
        remaining = max(0, self.horizon_ticks - 10)
        while remaining > 0 and not s.is_game_over():
            step = min(10, remaining)
            s.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, step)
            remaining -= step
        killed = sum(s.get_elixir_value_killed_by(c, me) for c in self.deck)
        lost = sum(s.get_elixir_value_killed_by(c, opp) for c in self.deck)
        return {
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
        if cand.slot >= HAND_SIZE:
            return 0.0
        st = self.rollout_stats(env, cand)
        p = self.profile
        cost = float(E.get_card_info(cand.card_id)["cost"])
        elixir = tactics.own_elixir(obs_own)

        u = (p["w_twr"] * (st["tower_dealt"] - baseline["tower_dealt"])
             - p["w_def"] * (st["tower_taken"] - baseline["tower_taken"])
             + p["w_trade"] * ((st["killed"] - baseline["killed"])
                               - (st["lost"] - baseline["lost"]))
             + p["w_crown"] * (st["crowns"] - baseline["crowns"])
             + p["w_pos"] * (st["pos"] - baseline["pos"])
             + p["w_cycle"] * self.cycle_value(cand.card_id))
        # Opportunity cost. Above the overflow line the bar is discarding income,
        # so holding is not actually free and the charge is relieved -- the same
        # 9.0 threshold train.W_ELIXIR_OVERFLOW uses.
        overflow_relief = max(0.0, elixir - ELIXIR_OVERFLOW_AT) / (
            10.0 - ELIXIR_OVERFLOW_AT)
        u -= p["w_cost"] * cost * (1.0 - min(1.0, overflow_relief))
        return float(u)

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
    def act(self, env, obs_own):
        """(slot, x, y) in our own frame. slot == HAND_SIZE means hold."""
        self.cycle.observe(env.get_hand_for_team(self.team))
        cands = self.candidates(env, obs_own)

        if self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon:
            # A uniformly random LEGAL action, no-op included -- waiting is a
            # real move and must stay in the noise distribution.
            c = cands[int(self.rng.integers(len(cands)))]
            return c.slot, c.x, c.y

        playable = [c for c in cands if c.slot < HAND_SIZE]
        if not playable:
            return HAND_SIZE, 0.0, 0.0

        if self.horizon_ticks <= 0:
            return self._rules_only(obs_own, playable)

        baseline = self.rollout_stats(env, NOOP)
        best, best_score = None, self.play_margin
        for c in playable:
            sc = self.score(env, c, baseline, obs_own)
            if sc > best_score:
                best, best_score = c, sc
        if best is None:
            return HAND_SIZE, 0.0, 0.0
        return best.slot, best.x, best.y

    def _rules_only(self, obs_own, playable):
        """Stages 0-1: no rollout at all, so the curriculum's easy rungs cost
        nothing. The gate matters more than the ranking here -- a bot that plays
        on every affordable step is the elixir-dumping opponent this project
        already replaced once."""
        threat = tactics.threat_level(obs_own)
        best, best_pri = None, 0.0
        for c in playable:
            pri = 0.0
            if c.role in ("melee", "ranged") and threat > 0.0:
                pri = 3.0
            elif c.role == "building" and threat > 0.0:
                pri = 2.5
            elif c.role == "spell":
                caught = tactics.spell_catch_map(obs_own).max()
                # Only cast when it actually catches something worth 4 elixir --
                # forcing Fireball once dropped win rate 97% -> 23%.
                pri = 2.0 if caught >= tactics.FIREBALL_DAMAGE else 0.0
            elif c.role == "wincon" and tactics.hog_should_commit(obs_own):
                pri = 1.5
            if pri > best_pri:
                best, best_pri = c, pri
        if best is None:
            return HAND_SIZE, 0.0, 0.0
        return best.slot, best.x, best.y
