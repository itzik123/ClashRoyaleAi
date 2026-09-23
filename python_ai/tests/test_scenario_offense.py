"""scenario_offense.py: an accelerator, kept default-OFF."""
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

CE = clash_royale_env.ClashRoyaleEnv


# --- scenario_offense.py, default-OFF ---

def test_offensive_scenarios_are_off_by_default():
    """Injection changes the state distribution, not the payoff, so enabling it
    before the payoff is right just pays a negative price more often. Off until
    prove_environment.py says otherwise.
    """
    from python_ai.envs import scenario_offense
    assert scenario_offense.OFFENSIVE_SCENARIO_PROB == 0.0
    env = gym_wrapper.MicroRoyaleEnv()
    assert env.offensive_scenario_prob == 0.0
    env.reset()
    assert env.last_scenario is None


def test_offensive_scenario_reports_a_stale_pyd_instead_of_an_attributeerror():
    """The state setters may be missing from a stale .pyd (the post-build copy can
    fail with MSB3073); report that at the entry point, not as an
    AttributeError mid-episode.
    """
    from python_ai.envs import scenario_offense
    if scenario_offense.HAS_STATE_SETTERS:
        pytest.skip("this .pyd exports the state setters")
    rng = np.random.default_rng(0)
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    with pytest.raises(RuntimeError, match="post-build copy"):
        scenario_offense.apply_offensive_scenario(
            env, rng, gym_wrapper.DEFAULT_DECK, prob=1.0)


@pytest.mark.skipif(
    not scenario_offense.HAS_STATE_SETTERS,
    reason="needs set_elixir_for_team/set_hand_for_team (stale .pyd)")
def test_punish_window_actually_builds_the_position_it_claims():
    """set_hand_for_team returns False rather than raising, so a rejected setup
    would count as practice that never happened.
    """
    from python_ai.envs import scenario_offense
    rng = np.random.default_rng(0)
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    name = scenario_offense.punish_window(env, rng, gym_wrapper.DEFAULT_DECK)
    assert name is not None and name.startswith("punish_window")
    assert 15 in list(env.get_hand_for_team(0)), "the win condition must be in hand"
    assert env.get_elixir_for_team(0) == pytest.approx(8.0)
    assert env.get_elixir_for_team(1) == pytest.approx(1.0)
    # Their commitment is on the board, visible to us as enemy mass.
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    plane = CE.BOARD_HEIGHT * CE.BOARD_WIDTH
    enemy = sum(float(obs[c * plane:(c + 1) * plane].sum()) for c in (4, 5, 6))
    assert enemy > 0.0, "punish_window injected nothing the agent can see"


@pytest.mark.skipif(
    not scenario_offense.HAS_STATE_SETTERS,
    reason="needs set_elixir_for_team/set_hand_for_team (stale .pyd)")
def test_counter_push_leaves_our_own_units_alive_on_our_side():
    from python_ai.envs import scenario_offense
    rng = np.random.default_rng(1)
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    name = scenario_offense.counter_push(env, rng, gym_wrapper.DEFAULT_DECK)
    assert name is not None and name.startswith("counter_push")
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    plane = CE.BOARD_HEIGHT * CE.BOARD_WIDTH
    ally = sum(float(obs[c * plane:(c + 1) * plane].sum()) for c in (0, 1, 2))
    assert ally > 0.0, "counter_push injected no survivors to push behind"


@pytest.mark.skipif(
    not scenario_offense.HAS_STATE_SETTERS,
    reason="needs set_elixir_for_team/set_hand_for_team (stale .pyd)")
def test_scenario_injection_reobserves_after_rewriting_the_state():
    """reset() returns the observation from before the rewrite; the env must
    re-read it.
    """
    env = gym_wrapper.MicroRoyaleEnv({"offensive_scenario_prob": 1.0,
                                      "scenario_seed": 0})
    obs, _ = env.reset()
    assert env.last_scenario is not None
    live = np.asarray(env.game.get_observation_for_team(0), np.float32)
    assert np.allclose(obs, live), "the returned observation is pre-scenario"

def test_teacher_follows_a_deck_change():
    """The opponent deck changes during phase 1; a teacher holding the old deck's
    role table would misread the new win condition and count cycle over cards
    it no longer holds.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=1)
    assert t.wincon_id == 15
    other = [2, 6, 25, 40, 24, 72, 33, 7]        # Giant instead of Hog Rider
    t.set_deck(other)
    assert t.deck == other
    assert t.wincon_id == 2, "the Giant is the new deck's building-targeter"
    assert set(t.roles) == set(other)
    assert t.cycle.distance_to(15) == len(other), "the old wincon is gone"


def test_env_deck_changes_propagate_to_the_teacher():
    """Two paths write the opponent deck: set_opponent_deck() and reset()'s
    randomize_opp_deck branch, which writes to self.game directly.
    """
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher"})
    assert env.teacher.wincon_id == 15
    env.set_opponent_deck([2, 6, 25, 40, 24, 72, 33, 7])
    assert env.teacher.wincon_id == 2

    rnd = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher",
                                      "randomize_opp_deck": True})
    for _ in range(3):
        rnd.reset()
        assert rnd.teacher.deck == rnd.current_opp_deck, (
            "the teacher is playing a different deck than the engine dealt it")
        assert set(rnd.teacher.roles) == set(rnd.current_opp_deck)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
