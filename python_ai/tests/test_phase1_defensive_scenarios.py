"""Phase 1 presents defensive emergencies, on the right tiles.

Defensive cards are worth almost nothing on a quiet board, so without
manufactured "defend or lose the tower" moments the policy never learns them.
This changes the state distribution, not the reward. The bridge lanes are
derived from the engine.
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai import engine_constants as EC
from python_ai.envs import gym_wrapper, scenarios

CE = clash_royale_env.ClashRoyaleEnv


def _enemy_troop_mass(obs):
    """Total enemy-troop occupancy: channels 4/5/6 (enemy melee / ranged /
    building-targeter), as `advisors/tactics.py` groups them.
    """
    spatial = np.asarray(obs)[:EC.SPATIAL_SIZE].reshape(
        EC.N_CHANNELS, EC.BOARD_H, EC.BOARD_W)
    return float(spatial[[4, 5, 6]].sum())


def test_bridge_lanes_are_the_engine_s_bridges():
    """Derived, not restated."""
    assert sorted(scenarios._BRIDGE_LANES) == pytest.approx(
        sorted([EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X]))


def test_every_bridge_spawn_lands_on_a_bridge_column():
    """The behavioural form of the same rule: a two-tile bridge spans cells 2-3
    and 14-15, and cell i covers [i-0.5, i+0.5], so a lane centre must sit
    inside one of those spans.
    """
    for lane in scenarios._BRIDGE_LANES:
        cells = {int(np.floor(lane - 0.25)), int(np.floor(lane + 0.25))}
        assert cells <= {2, 3, 14, 15}, f"lane {lane} is not on a bridge"


def test_phase1_reset_injects_a_threat_when_enabled():
    env = gym_wrapper.MicroRoyaleEnv({"defensive_scenario_prob": 1.0})
    obs, _ = env.reset(seed=7)

    assert env.last_scenario is not None
    assert _enemy_troop_mass(obs) > 0.0, \
        "the injected threat must be visible in the FIRST observation"


def test_phase1_injects_nothing_when_disabled():
    """Off by default, so an existing run's distribution is unchanged."""
    env = gym_wrapper.MicroRoyaleEnv()
    obs, _ = env.reset(seed=7)

    assert env.last_scenario is None
    assert _enemy_troop_mass(obs) == 0.0


def test_the_scenario_window_truncates_rather_than_terminating():
    """A window expiring is not the world ending: base_trainer bootstraps
    V(final_obs) on `truncated` and 0 on `terminated`, so reporting it as
    terminated would teach the critic that surviving a defence is worth zero.
    """
    env = gym_wrapper.MicroRoyaleEnv({"defensive_scenario_prob": 1.0})
    # Engine seed too: the opening hand comes from the engine's own generator,
    # which every earlier env in the session advances.
    env.game.seed(3)
    env.reset(seed=3)
    assert env.scenario_max_steps is not None

    noop = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}
    for _ in range(env.scenario_max_steps):
        _, _, terminated, truncated, _ = env.step(noop)
        if terminated or truncated:
            break

    assert truncated and not terminated
