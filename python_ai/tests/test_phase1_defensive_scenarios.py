"""Phase 1 must present defensive emergencies, and on the RIGHT tiles.

WHY PHASE 1 NEEDS THIS AT ALL
-----------------------------
Scenario injection was a pipeline-2 feature; `trainers/train.py`'s own comment
says it "never runs here". So across the 2026-08-28 phase-1 run (32,680
episodes) the agent never faced a manufactured "defend or lose the tower"
moment -- and a Cannon, a Log and a Fireball are worth almost nothing on a quiet
board. Measured on that run's final checkpoint, P(play | in hand) was 0.0053 for
Cannon, 0.0091 for The Log and 0.0011 for Fireball, while a well-placed Cannon
prevents a full Princess Tower (2,536 HP) in a real threat state.

The policy was not wrong. It was answering a question the state distribution
never asked. This changes the distribution, not the reward -- the same reason
`envs/scenarios.py` gives for existing.

AND THE GEOMETRY HAD DRIFTED
----------------------------
`_BRIDGE_LANES` was `[3.5, 13.5]`, written when the arena put bridges at x =
4.0 and 14.0. The 2026-08-21 re-centring moved them to 2.5 and 14.5 (cells 2-3
and 14-15), so 13.5 is the boundary of cell 13 -- WATER. Half of every injected
bridge push was landing off the bridge. That is the eighth stale copy of arena
geometry this project has found, and the rule CLAUDE.md states for all of them
applies here: derive from the bindings, never restate.
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai import engine_constants as EC
from python_ai.envs import gym_wrapper, scenarios

CE = clash_royale_env.ClashRoyaleEnv


def _enemy_troop_mass(obs):
    """Total enemy-troop occupancy in an observation.

    Channels 4/5/6 are enemy melee / ranged / building-targeter -- the same
    grouping `advisors/tactics.py` uses (CH_ENEMY_TROOP).
    """
    spatial = np.asarray(obs)[:EC.SPATIAL_SIZE].reshape(
        EC.N_CHANNELS, EC.BOARD_H, EC.BOARD_W)
    return float(spatial[[4, 5, 6]].sum())


def test_bridge_lanes_are_the_engine_s_bridges():
    """Derive, do not restate. 13.5 is a water cell under the current arena."""
    assert sorted(scenarios._BRIDGE_LANES) == pytest.approx(
        sorted([EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X]))


def test_every_bridge_spawn_lands_on_a_bridge_column():
    """The behavioural form of the same rule, independent of the constant.

    A two-tile bridge spans cells 2-3 and 14-15, and cell i covers
    [i-0.5, i+0.5], so a lane centre must sit inside one of those spans.
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
    """The default must stay off so an existing run's distribution is unchanged
    unless a caller asks for the new behaviour."""
    env = gym_wrapper.MicroRoyaleEnv()
    obs, _ = env.reset(seed=7)

    assert env.last_scenario is None
    assert _enemy_troop_mass(obs) == 0.0


def test_the_scenario_window_truncates_rather_than_terminating():
    """A window expiring is NOT the world ending.

    `rl/base_trainer.py` bootstraps V(final_obs) on `truncated` and 0 on
    `terminated`, reading the FLAG and never the reward's magnitude. Reporting a
    window expiry as terminated would teach the critic that surviving a defence
    is worth zero.
    """
    env = gym_wrapper.MicroRoyaleEnv({"defensive_scenario_prob": 1.0})
    # ENGINE seed, not just the gym seed. reset(seed=) drives the scenario
    # sampler; the opening HAND comes from the engine's own unseeded mt19937,
    # which every ClashRoyaleEnv built earlier in the session advances. So this
    # test's outcome depended on what else the suite had constructed before it,
    # and it failed intermittently under -q with no change to its own subject.
    env.game.seed(3)
    env.reset(seed=3)
    assert env.scenario_max_steps is not None

    noop = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}
    for _ in range(env.scenario_max_steps):
        _, _, terminated, truncated, _ = env.step(noop)
        if terminated or truncated:
            break

    assert truncated and not terminated
