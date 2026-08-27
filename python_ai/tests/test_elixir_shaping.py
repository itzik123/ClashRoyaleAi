"""elixir_shaping.py -- potential-based solvency.

Split out of the old single `test_python_ai.py` on 2026-08-20. The bodies are
unchanged -- only the shared header moved into `tests/conftest.py`, so the set of
test node ids is the same modulo the file name.

    python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
import clash_royale_env as E  # noqa: E402
from python_ai import engine_constants as EC  # noqa: E402
from python_ai.advisors import advisor_target as AT  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.envs import scenario_offense  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.rewards import shaping as T  # noqa: E402
from python_ai.rewards import shaping as train_shaping  # noqa: E402
from python_ai.rewards import weights as TW  # noqa: E402
from python_ai.rewards import weights as train_weights  # noqa: E402
from python_ai.rewards.elixir_shaping import (  # noqa: E402
    SOLVENCY_RESERVE, W_SOLVENCY, bankruptcy_rate, solvency_potential,
    solvency_shaping,
)
from python_ai.rl.coverage import (  # noqa: E402
    PLACEMENT_COVERAGE_COEF, placement_coverage_slots,
)
from python_ai.trainers.distill_tactics import masked_kl  # noqa: E402

#: The discount these arithmetic tests are written against. Deliberately a
#: FIXED fixture value and NOT `PPOConfig.gamma`: these cases assert exact
#: numbers out of `gamma*Phi(s') - Phi(s)`, so reading the live config would
#: make their expected values move every time someone tunes the discount --
#: a test that changes its own answer cannot pin anything. The separate
#: question of whether the TRAINER passes its real gamma is pinned by
#: tests/test_rl_config.py and tests/test_reward_horizon_invariant.py.
SHAPING_TEST_GAMMA = 0.99


CE = clash_royale_env.ClashRoyaleEnv


# ==========================================================================
# elixir_shaping.py -- potential-based solvency
# (was test_elixir_shaping.py)
# ==========================================================================
# Tests for the elixir solvency term.
#
#     python_ai/venv/Scripts/python.exe -m pytest python_ai/test_elixir_shaping.py -q
#
# The property that matters most is TELESCOPING: a potential-based term must
# contribute ~0 to an episode's return, or it is not policy-invariant and every
# safety argument in elixir_shaping.py's docstring evaporates.

GAMMA = 0.99


def _stats(e):
    return {"team0_elixir_current": np.asarray(e, dtype=np.float32)}


def test_potential_is_zero_at_and_above_the_reserve():
    """No charge in the healthy band -- the agent must be free to play."""
    phi = solvency_potential([SOLVENCY_RESERVE, 5.0, 7.0, 10.0])
    assert np.allclose(phi, 0.0)


def test_potential_falls_linearly_to_minus_w_at_zero():
    assert solvency_potential([0.0])[0] == pytest.approx(-W_SOLVENCY)
    assert solvency_potential([SOLVENCY_RESERVE / 2])[0] == pytest.approx(-W_SOLVENCY / 2)
    # Strictly monotone below the reserve: a step function would make every
    # broke state identical and give no reason to prefer 3.9 to 0.1.
    phi = solvency_potential([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.all(np.diff(phi) > 0)


def test_spending_below_the_reserve_is_charged_immediately():
    """The whole point: the cost lands at the spend, not seconds later."""
    f = solvency_shaping(_stats([1.0]), _stats([5.0]), GAMMA)
    assert f[0] < 0.0
    # 5 -> 1 crosses 3 elixir of the reserve band
    expected = GAMMA * (-W_SOLVENCY * 3.0 / 4.0) - 0.0
    assert f[0] == pytest.approx(expected, rel=1e-5)


def test_spending_inside_the_healthy_band_is_free():
    """9 -> 5 must cost nothing, or the term becomes a tax on acting at all."""
    f = solvency_shaping(_stats([5.0]), _stats([9.0]), GAMMA)
    assert f[0] == pytest.approx(0.0)


def test_regenerating_back_up_pays_it_back():
    f = solvency_shaping(_stats([2.0]), _stats([1.0]), GAMMA)
    assert f[0] > 0.0


def test_telescopes_to_approximately_zero_over_an_episode():
    """Policy-invariance in practice: the term must not add return.

    A realistic trace -- saving up, dumping to zero, recovering -- repeated many
    times. The sum must equal gamma^T*Phi(s_T) - Phi(s_0) up to the discounting,
    NOT accumulate.
    """
    rng = np.random.default_rng(0)
    e = 5.0
    trace = [e]
    for _ in range(400):
        e = min(10.0, e + 0.35)                      # regen per decision step
        if rng.random() < 0.3:
            e = max(0.0, e - rng.choice([3.0, 4.0, 5.0]))
        trace.append(e)

    total = 0.0
    for prev, cur in zip(trace[:-1], trace[1:]):
        total += float(solvency_shaping(_stats([cur]), _stats([prev]), GAMMA)[0])

    # Undiscounted telescoping bound: |sum| <= |Phi| range, and with gamma<1 the
    # residual is bounded by (1-gamma) * sum|Phi| which is small for a potential
    # capped at W_SOLVENCY.
    assert abs(total) < 0.5 * W_SOLVENCY * len(trace) * (1 - GAMMA) + W_SOLVENCY, total
    # And crucially it must not be a large one-sided drift.
    assert abs(total) < 0.6, f"term accumulated {total}, so it is not telescoping"


def test_a_pure_hoarder_earns_nothing():
    """Doing nothing must not be paid.

    This is the failure mode CLAUDE.md records three times -- a term that makes
    passivity a guaranteed-positive outcome. Sitting at full elixir keeps Phi at
    0, so every step's shaping is exactly 0.
    """
    total = sum(float(solvency_shaping(_stats([10.0]), _stats([10.0]), GAMMA)[0])
                for _ in range(300))
    assert total == pytest.approx(0.0)


def test_vectorized_over_envs():
    f = solvency_shaping(_stats([1.0, 5.0, 0.0]), _stats([5.0, 5.0, 4.0]), GAMMA)
    assert f.shape == (3,)
    assert f[0] < 0 and f[1] == pytest.approx(0.0) and f[2] < 0
    assert f.dtype == np.float32


def test_bankruptcy_rate_matches_the_reported_statistic():
    assert bankruptcy_rate([0.0, 1.0, 2.9, 3.0, 5.0]) == pytest.approx(0.6)
    assert bankruptcy_rate([5.0, 6.0]) == 0.0


def test_matches_compute_shaping_when_wired_in():
    """Integration: the term must be additive and leave everything else alone."""
    from python_ai.rewards import shaping as train_shaping, weights
    if not getattr(weights, "SOLVENCY_ENABLED", False):
        pytest.skip("solvency term not wired into compute_shaping yet")
    n = 2
    base = {k: np.zeros(n, dtype=np.float32) for k in (
        "team0_troop_damage", "team1_troop_damage", "team0_building_damage",
        "team1_building_damage", "team0_tower_damage", "team1_tower_damage",
        "team0_elixir_spent", "team1_elixir_spent", "fireball_value_killed",
        "fireball_elixir_spent", "fireball_in_hand")}
    base["team0_towers_alive"] = np.full(n, 3.0, dtype=np.float32)
    base["team1_towers_alive"] = np.full(n, 3.0, dtype=np.float32)
    base["enemy_tower_hp"] = np.full((n, 3), 2534.0, dtype=np.float32)

    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    prev["team0_elixir_current"] = np.array([8.0, 8.0], dtype=np.float32)
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["team0_elixir_current"] = np.array([8.0, 1.0], dtype=np.float32)

    out = train_shaping.compute_shaping(cur, prev, SHAPING_TEST_GAMMA)
    # env 0 stayed solvent, env 1 dropped to 1 elixir -> strictly worse
    assert out[1] < out[0]


# --- the bankruptcy floor's justification, pinned against the engine -------
#
# Added 2026-08-27. `bankruptcy_rate`'s docstring justified floor=3.0 as "the
# cost of the cheapest card in DEFAULT_DECK, so below it the action space is
# literally empty". That was true of the Giant deck and is false of the 2.6 Hog
# Cycle, whose cheapest card costs 1. The floor is deliberately unchanged (the
# 65.3% baseline is quoted against it); only the claim was wrong. These pin the
# real numbers so the comment cannot drift again.

def test_the_cheapest_default_deck_card_costs_one_not_three():
    import clash_royale_env as E
    from python_ai.envs.gym_wrapper import DEFAULT_DECK

    def _cost(cid):
        info = E.get_card_info(cid)
        return float(info["cost"] if isinstance(info, dict) else info.cost)

    costs = [_cost(c) for c in DEFAULT_DECK]
    assert min(costs) == 1.0, (
        f"DEFAULT_DECK costs are {costs}; bankruptcy_rate's docstring explains "
        "floor=3.0 in terms of the cheapest card and must be re-checked")


def test_the_action_space_is_NOT_empty_below_the_bankruptcy_floor():
    """The specific claim that was false: at 2.0 elixir several cards are still
    affordable, so 'bankrupt' cannot mean 'nothing is playable'."""
    import numpy as np
    import torch

    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.net import MicroRoyaleNet
    import clash_royale_env as E

    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    g = E.ClashRoyaleEnv(deck, deck, 20000)
    g.reset()
    # Returns None, unlike set_hand_for_team which is documented `-> bool`.
    g.set_elixir_for_team(0, 2.0)
    obs = torch.tensor(
        np.asarray(g.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)

    mask = net.affordability_mask(obs)[0]
    # last column is the always-legal no-op; the real cards are before it
    playable = int(mask[:net.hand_size].sum())
    assert playable > 0, (
        "no card affordable at 2.0 elixir -- if the deck changed so that this "
        "is now true, bankruptcy_rate's floor and docstring both need revisiting")
