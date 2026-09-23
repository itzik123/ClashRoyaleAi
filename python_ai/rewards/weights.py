"""Reward-shaping weights.

Potential-based in form (gamma*Phi(s') - Phi(s)): W_BLDG, W_LETHAL_SPELL and
SOLVENCY_COEF. Phi(terminal) is not zeroed, so each leaves a small terminal
residue; the tower term's acts as a tower-margin bonus of roughly 10-17% of the
+/-1 outcome.

Deliberately biasing: W_TOWER_DESTROYED, W_FLAWLESS_DEFENSE,
W_WIN_CONDITION_DAMAGE, W_SPELL_VALUE_START and DRAW_PENALTY. Each exists
because the policy-invariant version left pure defence optimal.

Keep an episode's shaping sum comparable to the terminal +/-1.
"""
import os

import clash_royale_env

from python_ai.rewards.elixir_shaping import W_SOLVENCY

W_BLDG = 0.5     # tower HP, potential-based

W_TROOPS = 0.1   # troop HP deltas

# Pays when the opponent spends elixir ("won the trade"). One-sided on purpose:
# taxing our own spend made doing nothing the safe choice. Kept small because
# at 0.15 it outweighed winning.
W_ELIXIR_TRADE = 0.03

# Paid once per tower taken or lost. tests/test_reward_horizon_invariant.py
# pins its ratio to a discounted win; retune it only against a training run.
W_TOWER_DESTROYED = 0.6

# Per-step pressure against sitting on a full bar.
ELIXIR_OVERFLOW_THRESHOLD = 9.0

W_ELIXIR_OVERFLOW = 0.1

# Terminal penalty for a timeout without a winner. As costly as a loss, so
# stalling never pays.
DRAW_PENALTY = 1.0

# Paid only on a win, scaled by the fraction of our tower HP still standing.
# Gating on the win means it can only rank wins against each other; weighting
# damage taken above damage dealt per step would reward turtling instead.
W_FLAWLESS_DEFENSE = 0.5

# Pay the flawless bonus only when at least one enemy tower fell. Gated on any
# win, it paid more for never attacking than a successful Hog earned.
FLAWLESS_REQUIRES_CROWN = True

# Extra reward per point of damage dealt by the win-condition card.
# Outcome-gated, so a suicidal push earns nothing. Anneal toward zero once the
# win condition is played.
W_WIN_CONDITION_DAMAGE = float(os.environ.get("CLASH_W_WINCON_DAMAGE", 1.0))

# Upper bound on one step's spend; normalises the trade term.
MAX_ELIXIR_PER_STEP = 10.0

# Both spell terms follow the deck's own damage spell, published by the envs as
# `spell_damage` / `spell_cost`. The Fireball constants below only serve as
# advisors.tactics' default spell geometry.
FIREBALL_CARD_ID = 7

FIREBALL_DAMAGE = float(clash_royale_env.get_card_info(FIREBALL_CARD_ID)["damage"]) \
    if "damage" in clash_royale_env.get_card_info(FIREBALL_CARD_ID) else 689.0

FIREBALL_COST = float(clash_royale_env.get_card_info(FIREBALL_CARD_ID)["cost"])

W_LETHAL_SPELL = 0.15

# Biasing on purpose, and annealed to zero so training ends on the true
# objective.
W_SPELL_VALUE_START = 0.08

W_SPELL_VALUE_FINAL = 0.0

SPELL_VALUE_ANNEAL_EPISODES = int(os.environ.get(
    "CLASH_SPELL_ANNEAL_EPISODES", 40000))

# Episode at which the anneal begins. 0 for a from-scratch run.
SPELL_VALUE_ANNEAL_START = int(os.environ.get("CLASH_SPELL_ANNEAL_START", 0))

# Potential-based solvency term; see elixir_shaping.py. Both knobs are
# env-overridable for ablations.
SOLVENCY_ENABLED = os.environ.get("CLASH_SOLVENCY", "1") != "0"
SOLVENCY_COEF = float(os.environ.get("CLASH_SOLVENCY_COEF", W_SOLVENCY))
