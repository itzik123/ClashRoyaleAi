"""Four hand-written opponents, driven purely from the observation vector.

No card-ID-specific logic anywhere here -- only elixir/cost from the scalar tail
and enemy positions from the spatial channels -- so these behave sensibly under
the randomized deck `set_scripted_opponent` gives them, unlike a historical
checkpoint which only ever learned to play one deck.

Split out of the env because they are a POLICY, not an environment: the env's
job is to step the engine, and a 120-line heuristic living inside a method of it
was the clearest single case of a class doing two unrelated jobs.
"""
import numpy as np

from python_ai.engine_constants import (
    BOARD_H, BOARD_W, HAND_SIZE, N_CHANNELS, SPATIAL_SIZE,
)

# --- Diverse scripted opponents (industry precedent: OpenAI Five bootstrapped
# against scripted bots before self-play -- pure self-play alone tends to
# converge onto whatever beats a narrow, self-similar pool rather than
# generalizing). Added as PERMANENT members of the SAME PFSP pool/weighting
# used for historical checkpoints (see _sample_pfsp_opponent) -- not a
# separate curriculum stage, since PFSP's own (1-winrate)^exponent weighting
# with a floor already does exactly what's wanted here: never fully drops
# out, gets sampled more if the trainee is currently weak against it. Tagged
# "scripted:<name>" (never a real file path) so _sample_pfsp_opponent's
# dispatch and pfsp_stats' win-rate keying both work unchanged -- see
# _set_opponent's prefix check.
SCRIPTED_OPPONENTS = ["scripted:Rusher", "scripted:Defender", "scripted:Cycler", "scripted:Counter"]

# Defender/Counter specifically model defensive play. PFSP's own win-rate-
# based weighting works AGAINST deliberately seeing more of them: once the
# trainee reliably beats a given opponent, (1-winrate)^PFSP_EXPONENT pushes
# its weight toward PFSP_MIN_WEIGHT same as anything else already mastered --
# so simply having them in the pool doesn't increase exposure on its own,
# and if anything actively suppresses it the better the trainee gets against
# them. The actual reason to keep facing them isn't "the trainee is
# currently weak against them" (PFSP's own criterion) -- it's that defensive
# pressure looked underrepresented in what's deciding average game length
# (see the ~300-tick average game length discussion this responds to), a
# property PFSP has no way to see or weight for on its own. This floor
# overrides PFSP_MIN_WEIGHT for these two specifically, independent of
# measured win-rate.
#
# First tried at 4x the base floor (0.20). Confirmed NOT enough: with a
# ~98-member pool where the other 96 sit at PFSP_MIN_WEIGHT once mastered
# (empirically true here -- aggregate decisive win rate stayed >=0.77
# throughout), 0.20 each gives Defender+Counter only ~7.7% combined sampling
# share -- against ep_len_history's maxlen=50 window, that's ~1-4 games/
# window, statistically invisible in Progress/Episode_Length_Ticks_50 (no
# trend after 3000+ episodes at that setting). A follow-up isolated eval
# (current greedy policy vs ONLY Defender/Counter, bypassing PFSP sampling
# entirely) confirmed the escalation logic itself does work -- AvgTicks 240
# and 339 respectively vs the live aggregate's ~250-290, Counter games up to
# 1000 ticks -- so the fix here is exposure, not the opponent logic. 0.8
# retargets combined share to ~25% (2*0.8 / (96*0.05 + 2*0.8) under the same
# all-others-floored assumption) -- large enough to actually move the
# aggregate if the isolated-eval numbers hold at scale. Even at 25% exposure
# the games are individually bimodal (most of both isolated runs still ended
# fast, 110-230 ticks) -- so if THIS still doesn't move AvgTicks, the next
# suspect is the bots' purely-reactive posture (no proactive early-game
# stance), not sampling weight again.
DEFENSIVE_SCRIPTED_OPPONENTS = {"scripted:Defender", "scripted:Counter"}

DEFENSIVE_SCRIPTED_MIN_WEIGHT = 0.8


def find_incursion(spatial, max_y):
    """(x, y, is_heavy) for the most urgent enemy incursion, or None.

    Channels 4-6 = enemy (team 0) melee/ranged/tank troops, from THIS observer's
    own mirrored point of view.

    Scans the WHOLE board, not just this observer's own half. It previously
    scanned only rows up to int(max_y) -- the enforced own-half placement bound
    -- meaning Defender/Counter only noticed a threat once it had already
    crossed the river, often most of the way to the tower by the time a response
    spawned and reached it. An isolated eval found 55-70% of head-to-head games
    still ending in an early blowout (110-230 ticks) regardless of exposure:
    sampling weight was not the bottleneck, REACTION LATENCY was.

    `is_heavy` is True iff the incursion includes a channel-6 (building-targeter
    / tank archetype) unit -- the same category `scenarios._WIN_CONDITION_IDS`
    treats as "the real threat". A plain melee/ranged squad troop does not set
    it even if it is also in range; that distinction is what Defender escalates
    on.
    """
    def clamp_y(y):
        # Placement is only ever legal within our own half, so for a threat
        # still crossing from the enemy's half this meets it right at the
        # bridge -- the earliest legal interception point -- instead of only
        # reacting once it is already deep in our own territory.
        return min(float(y), float(int(max_y)))

    tank_nz = np.nonzero(spatial[6])
    if tank_nz[0].size > 0:
        ys, xs = tank_nz
        # Smallest y = closest to team 1's own tower = most urgent.
        deepest = int(np.argmin(ys))
        return float(xs[deepest]), clamp_y(ys[deepest]), True
    other_nz = np.nonzero(spatial[4] + spatial[5])
    if other_nz[0].size == 0:
        return None
    ys, xs = other_nz
    deepest = int(np.argmin(ys))
    return float(xs[deepest]), clamp_y(ys[deepest]), False


def scripted_action(kind, obs1, lane, max_x, max_y):
    """(card_index, x, y, ability1, ability2) for one scripted opponent.

    Never activates a Champion ability -- there is no ability logic in these
    heuristics at all, so both slots are always False.
    """
    spatial = obs1[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    scalar = obs1[SPATIAL_SIZE:]
    elixir = float(scalar[0])
    costs = scalar[1:1 + HAND_SIZE]
    # cost <= 0 marks an empty/invalid hand slot (ClashEnv::
    # extractObservationForTeam writes `card ? card->cost/10.0f : 0.0f`) --
    # never a real, free card.
    affordable = [i for i in range(HAND_SIZE)
                  if costs[i] > 0.0 and costs[i] <= elixir + 1e-6]

    NO_OP = (HAND_SIZE, 0.0, 0.0, False, False)

    def lane_x():
        return max_x * (0.2 if lane == "left" else 0.8)

    if kind == "Rusher":
        if not affordable:
            return NO_OP
        return max(affordable, key=lambda i: costs[i]), lane_x(), max_y, False, False

    if kind == "Cycler":
        if not affordable:
            return NO_OP
        return (min(affordable, key=lambda i: costs[i]),
                max_x * 0.5, max_y * 0.5, False, False)

    incursion = find_incursion(spatial, max_y)

    if kind == "Defender":
        if incursion is None or not affordable:
            return NO_OP
        x, y, is_heavy = incursion
        # Escalate to the strongest affordable answer against a real
        # win-condition-style threat; an ordinary squad troop still gets the
        # cheapest efficient trade. A Defender that always reaches for its
        # cheapest card regardless of what is attacking loses to any
        # sufficiently strong rush no matter how often it is sampled -- this is
        # what makes facing it more often actually worth anything.
        slot = (max(affordable, key=lambda i: costs[i]) if is_heavy
                else min(affordable, key=lambda i: costs[i]))
        return slot, x, y, False, False

    if kind == "Counter":
        if not affordable:
            return NO_OP
        slot = max(affordable, key=lambda i: costs[i])
        if incursion is not None:
            x, y, _is_heavy = incursion
            return slot, x, y, False, False
        return slot, lane_x(), max_y, False, False

    return NO_OP
