"""The dense reward terms. Weights live in `rewards/weights.py`.

Every function is a pure function of the engine's MatchStatistics (the env's
info dict, as (num_envs,) arrays) and returns (num_envs,) float32.
"""
import numpy as np

from python_ai.engine_constants import (
    BOARD_H, BOARD_W, MAX_BUILDING_HP, MAX_TROOP_HP, N_CHANNELS,
    OWN_TOWER_HP_TOTAL, SPATIAL_SIZE,
)
from python_ai.rewards.elixir_shaping import solvency_shaping
from python_ai.rewards.weights import (
    ELIXIR_OVERFLOW_THRESHOLD,
    FLAWLESS_REQUIRES_CROWN, MAX_ELIXIR_PER_STEP, SOLVENCY_COEF,
    SOLVENCY_ENABLED, SPELL_VALUE_ANNEAL_EPISODES,
    SPELL_VALUE_ANNEAL_START, W_BLDG, W_ELIXIR_OVERFLOW, W_ELIXIR_TRADE,
    W_FLAWLESS_DEFENSE, W_LETHAL_SPELL, W_SPELL_VALUE_FINAL,
    W_SPELL_VALUE_START, W_TOWER_DESTROYED, W_TROOPS,
    W_WIN_CONDITION_DAMAGE,
)

#: Counters that only increase within an episode; a decrease means the vector
#: env auto-reset between the two readings.
_MONOTONE_UP = (
    "team0_troop_damage", "team1_troop_damage",
    "team0_building_damage", "team1_building_damage",
    "team0_tower_damage", "team1_tower_damage",
    "team0_wincon_damage",
    "team0_elixir_spent", "team1_elixir_spent",
    "spell_value_killed", "spell_elixir_spent",
)
#: Counters that only decrease. Needed on their own: an episode that ended with
#: towers lost may leave every counter above at a value the reset does not
#: lower.
_MONOTONE_DOWN = ("team0_towers_alive", "team1_towers_alive")


def auto_reset_mask(stats, prev_stats):
    """(num_envs,) bool: True where these two readings straddle an auto-reset.

    Counters restart at 0 on reset, so the potential-based terms would
    otherwise emit a large spurious reward on a step where nothing happened.
    Detected from the counters themselves so no caller has to remember to mask
    it.
    """
    n = np.asarray(stats["team0_troop_damage"]).shape[0]
    reset = np.zeros(n, dtype=bool)
    for key in _MONOTONE_UP:
        if key in stats and key in prev_stats:
            reset |= np.asarray(stats[key]) < np.asarray(prev_stats[key])
    for key in _MONOTONE_DOWN:
        if key in stats and key in prev_stats:
            reset |= np.asarray(stats[key]) > np.asarray(prev_stats[key])
    return reset


def flawless_defense_bonus(dones, step_rewards, stats, prev_stats,
                           w=None):
    """Rank wins by how little we gave up: w * (fraction of our tower HP left), on
    the step that ended a won episode, else 0.

    Uses np.maximum(prev, cur) because on a done step the env has already
    auto-reset and the counter may read the new episode's 0.
    """
    if w is None:
        w = W_FLAWLESS_DEFENSE
    is_win = dones & (step_rewards > 0.5)
    if FLAWLESS_REQUIRES_CROWN:
        # A win with all three enemy towers standing is a timeout win on tower
        # HP, the turtle this bonus must not pay.
        is_win = is_win & (np.asarray(stats["team1_towers_alive"]) < 3)
    taken = stats["team1_tower_damage"]
    if prev_stats is not None:
        taken = np.maximum(prev_stats["team1_tower_damage"], taken)
    hp_left = np.clip(1.0 - taken / OWN_TOWER_HP_TOTAL, 0.0, 1.0)
    return (w * is_win.astype(np.float32) * hp_left.astype(np.float32)).astype(np.float32)

def lethal_spell_potential(stats, w=W_LETHAL_SPELL):
    """Phi(s): w when an enemy tower is within the deck spell's damage and the
    spell is in hand and affordable, else 0.

    Moves credit for "cycle the spell in, then finish" to the moment it becomes
    reachable. `spell_damage` is the spell's measured damage to a Crown Tower.
    The keys are required: a default would silently be some other spell's
    numbers. A spell-less deck publishes 0.0, which makes the term exactly
    zero.
    """
    hp = stats["enemy_tower_hp"]                       # (num_envs, 3), absolute
    damage = np.reshape(np.asarray(stats["spell_damage"], dtype=np.float32), (-1, 1))
    cost = np.asarray(stats["spell_cost"], dtype=np.float32)
    in_range = np.any((hp > 0.0) & (hp <= damage), axis=1)
    actionable = (stats["spell_in_hand"] > 0.5) & \
                 (stats["team0_elixir_current"] >= cost)
    return w * (in_range & actionable).astype(np.float32)

def spell_value_weight(eps_done, start=None, length=None):
    """The spell-value weight at `eps_done`, annealing START -> FINAL.

    The term biases the optimum, so it must reach zero. `start` slides the
    schedule onto a resumed run; both knobs are env-overridable.
    """
    start = SPELL_VALUE_ANNEAL_START if start is None else start
    length = SPELL_VALUE_ANNEAL_EPISODES if length is None else length
    frac = min(1.0, max(0.0, (eps_done - start) / float(max(1, length))))
    return W_SPELL_VALUE_START + frac * (W_SPELL_VALUE_FINAL - W_SPELL_VALUE_START)

def spell_value_shaping(stats, prev_stats, w):
    """Pays for the elixir the deck's spell destroys, minus one unit per cast.

    Charging the cast makes this a trade ratio centred on break-even (kill 8
    elixir with a 4-cost spell: +1; kill nothing: -1). Without it a whiffed
    spell costs exactly zero and spam becomes safe. The denominator is the
    spell's own cost, so a 6-cost Rocket is priced as one.

    A positive trade counts only if we can still afford the spell afterwards; a
    negative one always counts. A spell-less deck (cost 0) is guarded rather
    than divided by, since a nan reward would silently destroy the net.
    """
    cost = np.asarray(stats["spell_cost"], dtype=np.float32)
    killed = np.maximum(0.0, stats["spell_value_killed"] - prev_stats["spell_value_killed"])
    spent = np.maximum(0.0, stats["spell_elixir_spent"] - prev_stats["spell_elixir_spent"])
    have_spell = cost > 0.0
    safe_cost = np.where(have_spell, cost, 1.0)
    casts = spent / safe_cost
    traded = (killed / safe_cost - casts) * have_spell.astype(np.float32)
    solvent = (stats["team0_elixir_current"] >= safe_cost).astype(np.float32)
    return (w * np.where(traded > 0.0, traded * solvent, traded)).astype(np.float32)

def tower_potential(stats, w_bldg=W_BLDG):
    """Phi(s): the tower-damage differential, normalized.

    Towers only. Damage to deployed buildings is priced with the troops in
    compute_shaping; at the tower rate a Cannon had to kill 5.3x its own HP to
    break even, and hiding it in a corner was free.
    """
    return w_bldg * (stats["team0_tower_damage"] - stats["team1_tower_damage"]) / MAX_BUILDING_HP

def compute_shaping(stats, prev_stats, gamma, w_bldg=W_BLDG, w_troops=W_TROOPS,
                     w_elixir=W_ELIXIR_TRADE, w_overflow=W_ELIXIR_OVERFLOW,
                     w_tower=W_TOWER_DESTROYED, w_spell=W_SPELL_VALUE_START,
                     w_wincon=W_WIN_CONDITION_DAMAGE):
    """Dense shaping (num_envs,), excluding the sparse win/loss reward.

    `stats` / `prev_stats` hold cumulative per-match counters from
    MatchStatistics (team0 = us, team1 = opponent). `team0_elixir_current` is
    instantaneous and read only from `stats`.

    `gamma` is required, not defaulted: potential-based shaping is
    policy-invariant only with the same gamma GAE uses, and `rewards/` may not
    import `rl/` to read it.
    """
    if prev_stats is None:
        return np.zeros(stats["team0_troop_damage"].shape[0], dtype=np.float32)

    # Counters only increase within an episode, so a negative delta is an
    # auto-reset; clamp it to zero.
    def delta(key):
        return np.maximum(0, stats[key] - prev_stats[key])

    # Deployed buildings trade HP like units, so they are priced with the
    # troops (break-even 1:1). The engine's building counter is towers plus
    # deployed buildings.
    def deployed_building(team):
        return (stats[f"team{team}_building_damage"] - stats[f"team{team}_tower_damage"],
                prev_stats[f"team{team}_building_damage"] - prev_stats[f"team{team}_tower_damage"])

    e_now, e_prev = deployed_building(0)
    a_now, a_prev = deployed_building(1)
    enemy_troops_damage = (delta("team0_troop_damage") + np.maximum(0, e_now - e_prev)) / MAX_TROOP_HP
    ally_troops_damage = (delta("team1_troop_damage") + np.maximum(0, a_now - a_prev)) / MAX_TROOP_HP
    enemy_elixir_spent = delta("team1_elixir_spent") / MAX_ELIXIR_PER_STEP

    ally_elixir_current = stats["team0_elixir_current"]
    overflow = np.maximum(0.0, ally_elixir_current - ELIXIR_OVERFLOW_THRESHOLD) / (10.0 - ELIXIR_OVERFLOW_THRESHOLD)

    # Potential-based: the gamma is load-bearing. Without it the difference is
    # a different, biased shaping that can pay for prolonging a game.
    tower_shaping = gamma * tower_potential(stats, w_bldg) - tower_potential(prev_stats, w_bldg)

    lethal_shaping = (gamma * lethal_spell_potential(stats)
                      - lethal_spell_potential(prev_stats))

    # A tower falling is a drop in the alive count; the clamp ignores the
    # post-reset jump back to 3.
    towers_taken = np.maximum(0, prev_stats["team1_towers_alive"] - stats["team1_towers_alive"])
    towers_lost = np.maximum(0, prev_stats["team0_towers_alive"] - stats["team0_towers_alive"])
    tower_events = w_tower * (towers_taken - towers_lost)

    solvency = (solvency_shaping(stats, prev_stats, gamma, w=SOLVENCY_COEF)
                if SOLVENCY_ENABLED else 0.0)

    # Not potential-based; see W_WIN_CONDITION_DAMAGE. Zero when the key is
    # absent (no building-targeter in the deck).
    wincon_damage = (delta("team0_wincon_damage") / MAX_BUILDING_HP
                     if "team0_wincon_damage" in stats and "team0_wincon_damage" in prev_stats
                     else 0.0)

    shaping = (tower_shaping
               + lethal_shaping
               + spell_value_shaping(stats, prev_stats, w_spell)
               + solvency
               + tower_events
               + w_wincon * wincon_damage
               + w_troops * (enemy_troops_damage - ally_troops_damage)
               + w_elixir * enemy_elixir_spent
               - w_overflow * overflow)

    # A step that straddles an auto-reset is not a transition.
    shaping = np.where(auto_reset_mask(stats, prev_stats), 0.0, shaping)

    return shaping.astype(np.float32)

def building_hp_end(obs_vec):
    """Remaining normalized building HP (ally, enemy) in one observation."""
    spatial = obs_vec[:SPATIAL_SIZE].reshape(N_CHANNELS, BOARD_H, BOARD_W)
    return float(spatial[3].sum()), float(spatial[7].sum())
