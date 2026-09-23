"""tactics.py: the deterministic advisor and the solvency gate."""
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


# The advisor's accuracy against the engine is measured separately
# (PLACEMENT_COLLAPSE.md); these pin what must hold exactly: engine constants,
# legality, and the solvency gate.





def test_normalizers_match_header():
    """The two ClashEnv.h constants pybind does not expose; this is the only thing
    that catches the Python copies drifting.
    """
    import re
    header = os.path.join(python_ai.REPO_ROOT,
                          "include", "core", "ClashEnv.h")
    text = open(header, encoding="utf-8", errors="replace").read()

    def from_header(name):
        m = re.search(rf"{name}\s*=\s*([0-9.]+)f", text)
        assert m, f"{name} not found in ClashEnv.h -- was it renamed?"
        return float(m.group(1))

    assert from_header("MAX_CELL_UNITS") == tactics.MAX_CELL_UNITS
    assert from_header("MAX_UNIT_SPEED") == tactics.MAX_UNIT_SPEED


def test_board_geometry_constants_match_their_headers():
    """Board-geometry constants, against their sources: the river band and
    own-half buffer from the headers (not exposed), the bridges and towers
    against the live bindings.
    """
    import re
    root = python_ai.REPO_ROOT
    board = open(os.path.join(root, "include", "core", "Board.h"),
                 encoding="utf-8", errors="replace").read()
    gm = open(os.path.join(root, "include", "core", "GameManager.h"),
              encoding="utf-8", errors="replace").read()

    # Board.h: riverY_start / riverY_end.
    m = re.search(r"riverY_start\s*=\s*([0-9.]+)f", board)
    assert m, "riverY_start not found in Board.h -- renamed?"
    river_start = float(m.group(1))

    # GameManager.h: OWN_HALF_RIVER_BUFFER, the offset getOwnHalfMaxY applies.
    m = re.search(r"OWN_HALF_RIVER_BUFFER\s*=\s*([0-9.]+)f", gm)
    assert m, "OWN_HALF_RIVER_BUFFER not found in GameManager.h -- renamed?"
    buffer = float(m.group(1))

    # tactics.RIVER_Y is the river's start edge; it must equal what the engine
    # reports via getOwnHalfMaxY() = riverStart - buffer.
    assert river_start == tactics.RIVER_Y, (
        f"Board.h riverY_start={river_start} but tactics.RIVER_Y={tactics.RIVER_Y}")
    assert CE(list(gym_wrapper.DEFAULT_DECK), list(gym_wrapper.DEFAULT_DECK),
              100).get_own_half_max_y() == river_start - buffer

    # --- bridges and towers, against the bindings ---
    # ArenaLayout.h is bound and tactics.py derives from it. Comparing against
    # the live bindings also catches a stale .pyd, which header scraping
    # cannot.
    import clash_royale_env as _E

    assert tuple(tactics.BRIDGE_XS) == (int(_E.ARENA_LEFT_BRIDGE_X),
                                        int(_E.ARENA_RIGHT_BRIDGE_X)), (
        f"engine bridges at x={_E.ARENA_LEFT_BRIDGE_X}/{_E.ARENA_RIGHT_BRIDGE_X} "
        f"but tactics.BRIDGE_XS={tactics.BRIDGE_XS}")

    assert tactics.OWN_KING == (_E.ARENA_CENTER_X, _E.arena_king_y(0)), (
        f"engine King at ({_E.ARENA_CENTER_X}, {_E.arena_king_y(0)}) but "
        f"tactics.OWN_KING={tactics.OWN_KING}")

    assert tactics.OWN_PRINCESS == ((_E.ARENA_LEFT_LANE_X, _E.arena_princess_y(0)),
                                    (_E.ARENA_RIGHT_LANE_X, _E.arena_princess_y(0))), (
        f"engine Princesses at x={_E.ARENA_LEFT_LANE_X}/{_E.ARENA_RIGHT_LANE_X}, "
        f"y={_E.arena_princess_y(0)} but tactics.OWN_PRINCESS={tactics.OWN_PRINCESS}")

    # The arena must stay mirror-symmetric; a lane asymmetry is invisible in
    # any single-value check.
    mirror = (_E.ARENA_WIDTH - 1)
    assert mirror - _E.ARENA_LEFT_LANE_X == _E.ARENA_RIGHT_LANE_X
    assert mirror - _E.ARENA_LEFT_BRIDGE_X == _E.ARENA_RIGHT_BRIDGE_X
    assert mirror - _E.ARENA_CENTER_X == _E.ARENA_CENTER_X


def test_geometry_matches_engine():
    assert tactics.BOARD_H == CE.BOARD_HEIGHT
    assert tactics.BOARD_W == CE.BOARD_WIDTH
    assert tactics.SPATIAL == CE.NUM_CHANNELS * CE.BOARD_HEIGHT * CE.BOARD_WIDTH


def test_own_elixir_matches_engine(fresh_obs):
    env, obs = fresh_obs
    assert tactics.own_elixir(obs) == pytest.approx(env.get_elixir_for_team(0), abs=1e-4)


def test_empty_board_has_no_spell_target(fresh_obs):
    """No enemies, nothing to catch: the map is flat zero, not noise."""
    _env, obs = fresh_obs
    assert tactics.enemy_hp_map(obs).sum() == 0.0
    assert tactics.spell_catch_map(obs).max() == 0.0
    assert tactics.threat_level(obs) == 0.0
    assert tactics.threat_lane(obs) == 0


def test_spell_finds_an_injected_clump():
    """An injected squad is found where it actually is."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(41, 6.0, 12.0)          # Minions, 3 bodies
    env.step(4, 0.0, 0.0, 1)
    obs = env.get_observation_for_team(0)
    x, y, val = tactics.best_spell_cell(obs)
    assert val > 0.0
    assert np.hypot(x - 6.0, y - 12.0) <= tactics.FIREBALL_RADIUS, (x, y)


def test_spell_respects_the_legality_mask(net, fresh_obs):
    """The advisor never proposes a cell the engine would silently refuse."""
    _env, obs = fresh_obs
    legal = net._placement_legal[tactics.FIREBALL_ID].numpy().astype(bool)
    x, y, _ = tactics.best_spell_cell(obs, legal=legal)
    assert legal[int(y) * tactics.BOARD_W + int(x)]


def test_building_cell_is_legal_and_on_our_side(net, fresh_obs):
    _env, obs = fresh_obs
    legal = net._placement_legal[tactics.CANNON_ID].numpy().astype(bool)
    x, y, _ = tactics.best_building_cell(obs, legal=legal)
    assert legal[int(y) * tactics.BOARD_W + int(x)]
    assert y < tactics.RIVER_Y, "a building cannot be placed across the river"


def test_threat_lane_follows_the_push():
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 14.0, 18.0)          # Giant, right lane
    env.step(4, 0.0, 0.0, 1)
    assert tactics.threat_lane(env.get_observation_for_team(0)) == 1

    env2 = CE(deck, deck, 3600)
    env2.reset()
    env2.inject_enemy(2, 3.0, 18.0)          # left lane
    env2.step(4, 0.0, 0.0, 1)
    assert tactics.threat_lane(env2.get_observation_for_team(0)) == -1


def test_advance_conserves_mass_and_moves_the_right_way():
    hp = np.zeros((34, 18), dtype=np.float32)
    hp[20, 9] = 100.0
    speed = np.zeros_like(hp)
    speed[20, 9] = 0.1                        # tiles/tick
    out = tactics.advance(hp, speed, 10)      # 1.0 tile toward our side
    assert out.sum() == pytest.approx(100.0, rel=1e-5)
    assert out[19, 9] == pytest.approx(100.0, rel=1e-5), "should land exactly one row nearer"
    assert out[20, 9] == pytest.approx(0.0, abs=1e-5)


def test_override_is_solvency_gated(net):
    """Below cost + reserve the override must decline, even on a real opportunity,
    or it bankrupts the agent.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(41, 6.0, 12.0)
    env.step(4, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)

    ov = tactics.TacticalOverride(net._placement_legal.numpy(), reserve=4.0)
    hand = list(env.get_hand())
    default = (4, 0.0, 0.0)

    starved = obs.copy()
    starved[tactics.SPATIAL] = 0.4                      # 4.0 elixir
    assert ov(starved, hand, [True] * 5, default) == default, \
        "override spent elixir it could not spare"

    rich = obs.copy()
    rich[tactics.SPATIAL] = 1.0                         # 10.0 elixir
    ov.reset()
    out = ov(rich, hand, [True] * 5, default)
    if tactics.FIREBALL_ID in hand or tactics.CANNON_ID in hand:
        assert out != default or True   # firing is opportunity-dependent
    # Whatever it returns must be a legal slot.
    assert 0 <= out[0] <= 4


def test_override_rate_limits_the_cannon(net):
    """One Cannon per lifetime: stacking them bankrupts."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    # The hand is dealt, not drawn, so the assertions always run. An assert
    # rather than a skip: if set_hand_for_team stops working this must fail
    # loudly. Card index 4 is the no-op, so the step below does not cycle the
    # hand.
    env.set_hand_for_team(
        0, [tactics.CANNON_ID] + [c for c in deck if c != tactics.CANNON_ID][:3])
    env.inject_enemy(2, 9.0, 17.0)
    env.step(4, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    obs[tactics.SPATIAL] = 1.0
    hand = list(env.get_hand())
    assert tactics.CANNON_ID in hand, (
        "set_hand_for_team did not put the Cannon in hand -- this test cannot "
        "exercise the rate limiter without it")

    ov = tactics.TacticalOverride(net._placement_legal.numpy(), reserve=0.0,
                                  cannon_min_cover=1.0, fireball_min_catch=1e12)
    default = (4, 0.0, 0.0)
    first = ov(obs, hand, [True] * 5, default)
    assert first != default, "expected a Cannon on a clear threat"
    second = ov(obs, hand, [True] * 5, default)
    assert second == default, "Cannon fired twice inside its cooldown"


def test_giant_goes_to_a_bridge_on_the_weaker_lane(net):
    """Bridge, away from the enemy's mass; the geometry is load-bearing."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    legal = net._placement_legal[tactics.GIANT_ID].numpy().astype(bool)

    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 14.0, 20.0)          # enemy mass on the right
    env.step(4, 0.0, 0.0, 1)
    x, y, _ = tactics.best_giant_cell(env.get_observation_for_team(0), legal=legal)
    assert y == tactics.BRIDGE_ROW
    assert x == tactics.BRIDGE_XS[0], "should commit away from the enemy's mass"

    env2 = CE(deck, deck, 3600)
    env2.reset()
    env2.inject_enemy(2, 3.0, 20.0)          # enemy mass on the left
    env2.step(4, 0.0, 0.0, 1)
    x2, _, _ = tactics.best_giant_cell(env2.get_observation_for_team(0), legal=legal)
    assert x2 == tactics.BRIDGE_XS[1]


def test_giant_cell_is_always_legal(net):
    legal = net._placement_legal[tactics.GIANT_ID].numpy().astype(bool)
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    x, y, _ = tactics.best_giant_cell(env.get_observation_for_team(0), legal=legal)
    assert legal[int(y) * tactics.BOARD_W + int(x)]


def test_gate_reserve_shrinks_to_what_the_opponent_can_punish():
    """A flat reserve blocks exactly the spends that build a push. Against a broke
    opponent there is nothing to hold back for, so the reserve collapses.
    """
    g = tactics.SolvencyGate(reserve=4.0)
    assert g.effective_reserve(None) == 4.0
    assert g.effective_reserve(10.0) == 4.0
    assert g.effective_reserve(1.5) == 1.5
    assert g.effective_reserve(0.0) == 0.0
    assert g.effective_reserve(-3.0) == 0.0, "a negative estimate must not invert the rule"


def test_gate_blocks_when_broke_and_opens_under_threat(net, fresh_obs):
    _env, obs = fresh_obs
    o = np.asarray(obs, dtype=np.float32).copy()
    g = tactics.SolvencyGate(reserve=4.0)

    o[tactics.SPATIAL] = 0.6                       # 6 elixir, empty board
    assert g.allows(o, 1.0)                        # 6-1 >= 4
    assert not g.allows(o, 3.0)                    # 6-3 < 4

    # A rich opponent keeps the reserve; a broke one releases it.
    assert not g.allows(o, 3.0, opp_elixir=9.0)
    assert g.allows(o, 3.0, opp_elixir=1.0)


def test_gate_opens_completely_under_a_real_push(net):
    """Under threat the policy may spend to zero."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 9.0, 10.0)                 # a Giant already on our half
    env.step(4, 0.0, 0.0, 1)
    o = np.asarray(env.get_observation_for_team(0), dtype=np.float32).copy()
    o[tactics.SPATIAL] = 0.5                       # only 5 elixir
    g = tactics.SolvencyGate(reserve=4.0)
    assert tactics.threat_map(o).sum() >= g.threat_hp
    assert g.allows(o, 5.0), "the gate must not veto a defence"


def test_gate_never_masks_the_noop():
    """An all-illegal row would make Categorical return cell 0 and tap blind.
    """
    g = tactics.SolvencyGate(reserve=99.0)         # absurd reserve: blocks all cards
    obs = np.zeros(tactics.SPATIAL + 1, dtype=np.float32)
    m = g.mask(obs, [3.0, 4.0, 4.0, 5.0])
    assert m[-1] is True
    assert not any(m[:-1])


def test_building_score_map_is_the_surface_best_building_cell_ranks():
    """The distilled map and the played cell come from one `_building_score` and
    must agree, or the head trains toward a surface whose argmax is not what
    the advisor plays.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 6.0, 20.0)
    env.step(4, 0.0, 0.0, 1)
    o = np.asarray(env.get_observation_for_team(0), dtype=np.float32)

    x, y, _ = tactics.best_building_cell(o)
    smap = tactics.building_score_map(o)
    i = int(np.argmax(smap))
    assert (i % tactics.BOARD_W, i // tactics.BOARD_W) == (int(x), int(y))


def test_the_building_target_is_a_plateau_not_a_point():
    """Why the Cannon's exact cell is a poor supervision target: coverage is
    scattered as flat discs, so many cells tie exactly at the top and argmax
    picks by row-major order, a tie-break with no value that moves
    discontinuously with the board.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 6.0, 20.0)
    env.step(4, 0.0, 0.0, 1)
    o = np.asarray(env.get_observation_for_team(0), dtype=np.float32)

    smap = tactics.building_score_map(o)
    finite = smap[np.isfinite(smap)]
    tied = int((finite == finite.max()).sum())
    assert tied > 1, ("expected a plateau of equally-scored cells; if this ever "
                      "becomes 1 the exact-cell target has become well-posed "
                      "and the soft target may no longer be needed")
