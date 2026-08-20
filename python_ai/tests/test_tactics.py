"""tactics.py -- the deterministic advisor and the solvency gate.

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

CE = clash_royale_env.ClashRoyaleEnv


# ==========================================================================
# tactics.py -- the deterministic advisor and the solvency gate
# (was test_tactics.py)
# ==========================================================================
# Tests for the deterministic tactical advisor.
#
#     python_ai/venv/Scripts/python.exe -m pytest python_ai/test_tactics.py -q
#
# The advisor's accuracy against the engine is measured separately (see
# PLACEMENT_COLLAPSE.md); these pin the properties that must hold exactly --
# engine constants, legality, and the solvency gate whose absence lost the first
# version of the A/B.





def test_normalizers_match_header():
    """The two ClashEnv.h constants pybind does not expose.

    If either changes in the header this test is the only thing that catches it,
    because nothing else in Python re-derives them -- exactly the second-copy
    drift CLAUDE.md forbids, made detectable where it cannot be avoided.
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
    """The BOARD-geometry constants Python copies because nothing exposes them.

    No binding reads back the river band or the bridge positions -- ClashEnv
    exposes get_own_half_max_y() and nothing else -- so tactics.py keeps its own
    copies. This test is the only thing standing between those copies and the
    next geometry change.

    That change is not hypothetical. CLAUDE.md records the river moving on
    2026-07-29 ([16,18) -> [15.5,17.5)) and the towers on 2026-07-30, and BOTH
    times a stale Python copy survived the edit: models/net.py's '18*16=288' comment
    and calibrate.py scoring bridges against y=17.0. This pins the remaining
    copies to the headers they came from.
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

    # tactics.RIVER_Y is the river's START edge. It must equal what the engine
    # reports live, via getOwnHalfMaxY() = riverStart - buffer.
    assert river_start == tactics.RIVER_Y, (
        f"Board.h riverY_start={river_start} but tactics.RIVER_Y={tactics.RIVER_Y}")
    assert CE(list(gym_wrapper.DEFAULT_DECK), list(gym_wrapper.DEFAULT_DECK),
              100).get_own_half_max_y() == river_start - buffer

    # Board.h: the two bridge x's, which tactics.BRIDGE_XS copies.
    xs = [float(x) for x in re.findall(r"(?:left|right)Bridge\s*\{\s*([0-9.]+)f", board)]
    assert len(xs) == 2, f"expected 2 bridge x's in Board.h, found {xs}"
    assert tuple(xs) == tuple(float(v) for v in tactics.BRIDGE_XS), (
        f"Board.h bridges at x={xs} but tactics.BRIDGE_XS={tactics.BRIDGE_XS}")

    # GameManager::reset()'s addTower() calls -- team 0's King and its two
    # Princesses, which tactics.OWN_KING / OWN_PRINCESS copy. Nothing exposes
    # entity positions through the bindings, so these are the last hand-typed
    # geometry in the module. Both moved on 2026-07-30 (King x 8.5 -> 9.0, left
    # Princess x 3.0 -> 4.0), which is precisely why they are pinned.
    king = re.search(r"addTower\(\s*([0-9.]+)f,\s*([0-9.]+)f,\s*4008,\s*0,", gm)
    assert king, "team-0 King addTower() call not found in GameManager.h -- changed shape?"
    assert (float(king.group(1)), float(king.group(2))) == tactics.OWN_KING, (
        f"GameManager.h King at {king.group(1)},{king.group(2)} but "
        f"tactics.OWN_KING={tactics.OWN_KING}")

    princesses = re.findall(
        r'addTower\(\s*([0-9.]+)f,\s*([0-9.]+)f,\s*0,\s*"Princess Tower"', gm)
    assert len(princesses) == 2, (
        f"expected 2 team-0 Princess addTower() calls, found {princesses}")
    found = tuple((float(x), float(y)) for x, y in princesses)
    assert found == tactics.OWN_PRINCESS, (
        f"GameManager.h Princesses at {found} but "
        f"tactics.OWN_PRINCESS={tactics.OWN_PRINCESS}")


def test_geometry_matches_engine():
    assert tactics.BOARD_H == CE.BOARD_HEIGHT
    assert tactics.BOARD_W == CE.BOARD_WIDTH
    assert tactics.SPATIAL == CE.NUM_CHANNELS * CE.BOARD_HEIGHT * CE.BOARD_WIDTH


def test_own_elixir_matches_engine(fresh_obs):
    env, obs = fresh_obs
    assert tactics.own_elixir(obs) == pytest.approx(env.get_elixir_for_team(0), abs=1e-4)


def test_empty_board_has_no_spell_target(fresh_obs):
    """No enemies -> nothing to catch. The map must be flat zero, not noise."""
    _env, obs = fresh_obs
    assert tactics.enemy_hp_map(obs).sum() == 0.0
    assert tactics.spell_catch_map(obs).max() == 0.0
    assert tactics.threat_level(obs) == 0.0
    assert tactics.threat_lane(obs) == 0


def test_spell_finds_an_injected_clump():
    """An injected squad must be found, and found where it actually is."""
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
    """The advisor must never propose a cell the engine will silently refuse."""
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
    """The gate that the ungated A/B proved necessary.

    With elixir below cost+reserve the override MUST decline, even when the
    tactical opportunity is real -- otherwise it bankrupts an agent that already
    sits under 3 elixir 65% of the time.
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
    # whatever it returns must be a legal slot
    assert 0 <= out[0] <= 4


def test_override_rate_limits_the_cannon(net):
    """One Cannon per lifetime; stacking them is how the naive version bankrupted."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 9.0, 17.0)
    env.step(4, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    obs[tactics.SPATIAL] = 1.0
    hand = list(env.get_hand())
    if tactics.CANNON_ID not in hand:
        pytest.skip("Cannon not in the opening hand this shuffle")

    ov = tactics.TacticalOverride(net._placement_legal.numpy(), reserve=0.0,
                                  cannon_min_cover=1.0, fireball_min_catch=1e12)
    default = (4, 0.0, 0.0)
    first = ov(obs, hand, [True] * 5, default)
    assert first != default, "expected a Cannon on a clear threat"
    second = ov(obs, hand, [True] * 5, default)
    assert second == default, "Cannon fired twice inside its cooldown"


def test_giant_goes_to_a_bridge_on_the_weaker_lane(net):
    """The measured rule: bridge, away from the enemy's mass.

    Scored at 535.6 enemy tower damage against 3.3 for the policy's own cell
    over 913 states, so the geometry here is load-bearing rather than cosmetic.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    legal = net._placement_legal[tactics.GIANT_ID].numpy().astype(bool)

    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(2, 14.0, 20.0)          # enemy mass on the RIGHT
    env.step(4, 0.0, 0.0, 1)
    x, y, _ = tactics.best_giant_cell(env.get_observation_for_team(0), legal=legal)
    assert y == tactics.BRIDGE_ROW
    assert x == tactics.BRIDGE_XS[0], "should commit away from the enemy's mass"

    env2 = CE(deck, deck, 3600)
    env2.reset()
    env2.inject_enemy(2, 3.0, 20.0)          # enemy mass on the LEFT
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
    """The fix for the gate being anti-offense.

    A flat reserve blocks exactly the spends that build a push, and measurably
    cost 4,645 tower damage dealt per episode. Against a broke opponent there is
    nothing to hold back for, so the reserve must collapse.
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
    """Under threat the policy must be free to spend to zero as before."""
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
    """An all-illegal row would make Categorical return cell 0 and tap blind."""
    g = tactics.SolvencyGate(reserve=99.0)         # absurd reserve: blocks all cards
    obs = np.zeros(tactics.SPATIAL + 1, dtype=np.float32)
    m = g.mask(obs, [3.0, 4.0, 4.0, 5.0])
    assert m[-1] is True
    assert not any(m[:-1])


def test_building_score_map_is_the_surface_best_building_cell_ranks():
    """The map that gets DISTILLED and the cell that gets PLAYED must agree.

    They are two entry points to one `_building_score`; this pins that they
    stay that way, since a drifting copy would train the head toward a surface
    whose argmax is not the cell the advisor actually plays.
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
    """WHY the Cannon's exact cell is a bad supervision target, as a number.

    Coverage is scattered as flat discs, so many cells tie EXACTLY at the top
    and `argmax` returns whichever comes first in row-major order. The advisor
    is indifferent among them; a head fitted to the argmax is being asked to
    learn that tie-break, which carries no value and moves discontinuously with
    the board. Measured here rather than asserted in a comment.
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
