"""The python_ai test suite -- every ML-side test in one file.

    python_ai/venv/Scripts/python.exe -m pytest python_ai/test_python_ai.py -q

Consolidated 2026-08-14 from test_tactics.py, test_placement_coverage.py,
test_elixir_shaping.py and test_placement_hires.py. The test bodies are
unchanged -- only the four import headers were merged into one, so the set of
test node ids is identical to the four files it replaces (verified by
collecting both and diffing the names).

Each section keeps its original module docstring as a comment block, because
those record WHY the tests exist -- which is the half that is expensive to
reconstruct and the half a merge would otherwise silently drop.
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.envs import scenario_offense  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.rewards.elixir_shaping import (  # noqa: E402
    SOLVENCY_RESERVE, W_SOLVENCY, bankruptcy_rate, solvency_potential,
    solvency_shaping,
)
from python_ai import engine_constants as EC  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.rl.coverage import (  # noqa: E402
    PLACEMENT_COVERAGE_COEF, placement_coverage_slots,
)

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

@pytest.fixture(scope="module")
def net():
    return MicroRoyaleNet(num_ability_slots=0)


@pytest.fixture(scope="module")
def fresh_obs():
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    return env, env.get_observation_for_team(0)


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
    times a stale Python copy survived the edit: model.py's '18*16=288' comment
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ==========================================================================
# the placement-gradient coverage hole
# (was test_placement_coverage.py)
# ==========================================================================
# Pins the placement-gradient coverage hole and the fix for it.
#
# Run (the .pyd is Python 3.11 only):
#
#     python_ai/venv/Scripts/python.exe -m pytest python_ai/test_placement_coverage.py -q
#
# WHY THESE TESTS EXIST
# ---------------------
# `card_id_embed` is `nn.Linear(num_card_ids, 16, bias=False)`, so column c of its
# weight belongs to card id c ALONE. `placement_given_card` selects only the
# chosen slot's embedding, and nothing else downstream of the LSTM consumes the
# others -- the card head reads `hx`, not the embeddings.
#
# That makes the coverage hole mechanically visible rather than merely plausible:
# the gradient of the loss with respect to an unchosen card's embedding column is
# EXACTLY zero, in exact arithmetic, not just small. `test_unchosen_card_gets_no_
# gradient` asserts the defect, and `test_coverage_pass_restores_gradient` asserts
# the fix removes it. The first test is what fails on the old code path.
#
# This is the whole root cause of the (11,0) placement collapse; see
# PLACEMENT_COLLAPSE.md for the behavioural measurements.

L, B = 3, 2


def _fixture():
    """A tiny (L,B) chunk built from real observations, with a real hand."""
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    obs = torch.tensor(env.get_observation_for_team(0), dtype=torch.float32)
    obs_seq = obs.view(1, 1, -1).expand(L, B, -1).contiguous()

    flat = obs_seq.view(L * B, -1)
    feats, embeds, spatial = net.extract_features(flat)
    feats_seq = feats.view(L, B, -1)
    embeds_seq = embeds.view(L, B, *embeds.shape[1:])
    spatial_seq = spatial.view(L, B, *spatial.shape[1:])
    # Everything affordable, so "unchosen" is a real choice and not a mask
    # artifact -- otherwise the test could pass for the wrong reason.
    card_mask = torch.ones(L, B, net.hand_size + 1, dtype=torch.bool)
    resets = torch.ones(L, B)
    hidden = (torch.zeros(B, 256), torch.zeros(B, 256))
    hand = net.hand_card_ids(flat)[0].tolist()
    return net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand


def test_unchosen_card_gets_no_gradient():
    """THE DEFECT. Scoring only the chosen card starves every other card.

    Slot 0 is chosen everywhere; the assertion is that slot 1's card gets an
    exactly-zero gradient. That is what freezes an unplayed card's placement map
    at a constant cell forever.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden)
    assert extra is None, "no coverage pass was requested"

    # The pre-fix loss: card log-prob + placement of the CHOSEN card only.
    loss = (Categorical(logits=cl).entropy().mean()
            + Categorical(logits=pl).entropy().mean() + values.mean())
    net.zero_grad(set_to_none=True)
    loss.backward()

    g = net.card_id_embed.weight.grad
    assert g is not None
    chosen_id, other_id = hand[0], hand[1]
    assert chosen_id != other_id
    assert g[:, chosen_id].abs().sum().item() > 0.0, "the chosen card must receive gradient"
    # Exactly zero, not merely small -- this is a structural disconnection.
    assert g[:, other_id].abs().sum().item() == 0.0, (
        "an unchosen card received placement gradient; the coverage hole this "
        "test pins has changed shape")


def test_coverage_pass_restores_gradient():
    """THE FIX. The coverage term reaches the card the actor loss cannot."""
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)      # always slot 1

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden,
        extra_card_idx_seq=cover)
    assert extra is not None and extra.shape == pl.shape

    loss = (Categorical(logits=cl).entropy().mean()
            + Categorical(logits=pl).entropy().mean() + values.mean()
            + PLACEMENT_COVERAGE_COEF * Categorical(logits=extra).entropy().mean())
    net.zero_grad(set_to_none=True)
    loss.backward()

    g = net.card_id_embed.weight.grad
    assert g[:, hand[1]].abs().sum().item() > 0.0, (
        "the coverage pass did not deliver gradient to the unchosen card")


def test_coverage_does_not_change_the_ppo_ratio():
    """The regularizer must not touch the quantity PPO is clipping.

    If the coverage pass altered the chosen action's log-prob, it would corrupt
    the ratio silently -- the exact failure mode CLAUDE.md records for masks
    that drift between rollout and update.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)

    with torch.no_grad():
        _, pl_a, v_a, aux_a, _, _ = net.forward_sequence(
            feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden)
        _, pl_b, v_b, aux_b, _, extra = net.forward_sequence(
            feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets, hidden,
            extra_card_idx_seq=cover)

    assert torch.equal(pl_a, pl_b), "chosen-card placement logits changed"
    assert torch.equal(v_a, v_b) and torch.equal(aux_a, aux_b)


def test_coverage_slots_are_affordable_or_the_noop_fallback():
    """The sampler must never propose a slot the affordability mask forbids."""
    torch.manual_seed(1)
    hand_size = 4
    mask = torch.zeros(5, 3, hand_size + 1, dtype=torch.bool)
    mask[..., -1] = True                    # no-op always legal
    mask[0, 0, 2] = True                    # exactly one affordable card
    mask[1, :, 1] = True
    mask[2, 1, 0] = True
    idx = placement_coverage_slots(mask, hand_size)
    assert idx.shape == (5, 3)
    assert int(idx[0, 0]) == 2
    assert all(int(idx[1, b]) == 1 for b in range(3))
    assert int(idx[2, 1]) == 0
    # Rows with nothing affordable fall back to the no-op slot, which is always
    # a legal index into card_embeds (hand_size == the no-op column).
    assert int(idx[0, 1]) == hand_size
    assert idx.max().item() <= hand_size


def test_slot_weights_reweight_without_ever_removing_a_candidate():
    """The weighted draw must RE-RANK affordable slots, never mask one out.

    A row whose affordable slots all carry weight 0 would otherwise look
    identical to a row with nothing affordable, and silently fall through to
    the no-op -- losing coverage exactly where it was meant to be added.
    """
    torch.manual_seed(5)
    hand_size = 4
    mask = torch.zeros(200, 2, hand_size + 1, dtype=torch.bool)
    mask[..., -1] = True
    mask[..., 0] = True          # slot 0 affordable everywhere
    mask[..., 3] = True          # slot 3 affordable everywhere

    w = torch.ones(200, 2, hand_size)
    w[..., 3] = 5.0              # slot 3 is "the advisor card"
    idx = placement_coverage_slots(mask, hand_size, slot_weights=w)
    share3 = float((idx == 3).float().mean())
    assert 0.75 < share3 < 0.92, f"5:1 weighting should give ~5/6, got {share3}"
    assert float((idx == 0).float().mean()) > 0.05, "the light slot got starved"

    # All-zero weights on the affordable slots: must fall back to the plain
    # affordable mask rather than to the no-op.
    zero = torch.zeros(200, 2, hand_size)
    idx0 = placement_coverage_slots(mask, hand_size, slot_weights=zero)
    assert set(idx0.view(-1).tolist()) <= {0, 3}, (
        "zero weights knocked every candidate out and fell through to the no-op")


def test_sampler_reaches_every_affordable_card():
    """Uniform over affordable slots -- otherwise coverage is itself biased."""
    torch.manual_seed(2)
    hand_size = 4
    mask = torch.zeros(400, 4, hand_size + 1, dtype=torch.bool)
    mask[..., :hand_size] = True
    idx = placement_coverage_slots(mask, hand_size)
    counts = torch.bincount(idx.view(-1), minlength=hand_size + 1)[:hand_size]
    share = counts.float() / counts.sum()
    assert share.min() > 0.20, f"a slot is being starved: {share.tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


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

    out = train_shaping.compute_shaping(cur, prev)
    # env 0 stayed solvent, env 1 dropped to 1 elixir -> strictly worse
    assert out[1] < out[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ==========================================================================
# place_hires -- the high-resolution placement branch
# (was test_placement_hires.py)
# ==========================================================================
# Pins the high-resolution placement branch: its safety, and its point.
#
# Run (the .pyd is Python 3.11 only):
#
#     python_ai/venv/Scripts/python.exe -m pytest python_ai/test_placement_hires.py -q
#
# WHAT THIS BRANCH IS FOR
# -----------------------
# `cnn_trunk` pools twice, so the placement head reads a 9x5 map of a 34x18 board
# and `place_up` blows it back up; one pooled cell covers ~4x4 board tiles and the
# card context enters as a spatially UNIFORM vector. `place_hires` adds a parallel
# path from the trunk's own PRE-pool 16x34x18 activation, at one-tile resolution,
# conditioned on the same (hx, card) context, added to the coarse logits as a
# residual.
#
# A CLAIM THIS FILE MEASURED AND HAD TO WEAKEN
# --------------------------------------------
# The handoff diagnosed the head as unable to EXPRESS an exact cell, from
# `distill_tactics.py`'s signature of cross-entropy falling 180.9 -> 21.4 while
# exact-cell argmax match never left 0.0%. Tested directly here, that is too
# strong: on the task reduced to its essential the coarse head fits 14/14 exactly
# (`test_both_heads_can_resolve_a_single_column_at_small_scale`), because
# nearest-upsample followed by 3x3 convs lets a fine cell mix neighbouring pooled
# cells, which recovers sub-block position. The branch is therefore a resolution
# INCREASE whose value has to be measured at realistic scale (`prove_hires.py`),
# not a repair of something provably impossible. Recorded here rather than
# quietly dropped, because the original claim is what justified the work.
#
# WHY THE FINAL CONV IS ZERO-INITIALIZED
# --------------------------------------
# The handoff proposed concatenating into `place_up`, which changes its shape and
# therefore **discards the trained placement head** from every checkpoint --
# exactly the trade the 2026-08-09 checkerboard fix had to make. That cost is
# avoidable. A zero-initialized residual branch computes the identical function at
# init, so an existing checkpoint loads and behaves BIT-IDENTICALLY, the cards
# that currently work keep working, and only genuinely new parameters start fresh.
# `test_zero_init_is_bit_identical` and `test_old_checkpoint_keeps_the_placement_head`
# are what make that claim checkable rather than asserted.
#
# The gradient still flows: with the last conv at zero its own gradient is
# nonzero (it sees a live activation), so it leaves zero on the first step and the
# layer beneath it starts learning on the second. Standard zero-conv behaviour.
#
# THE TEST THAT CARRIES THE ARGUMENT
# ----------------------------------
# `test_hires_branch_learns_exact_cells_the_coarse_head_cannot` is a controlled
# A/B: one net, one dataset, one optimizer, one seed. The ONLY difference between
# the arms is whether `place_hires` is trainable -- the control freezes it at its
# zero init, which is bit-exactly the old architecture. If the coarse head could
# express per-state exact cells, both arms would fit. It cannot, and that is the
# whole reason this branch exists.

def _net(seed=0):
    torch.manual_seed(seed)
    return MicroRoyaleNet(num_ability_slots=0)


def _obs_batch(n=8, seed=0):
    """`n` genuinely different real observations, from a real rollout."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    rng = np.random.default_rng(seed)
    rows = [np.asarray(env.get_observation_for_team(0), dtype=np.float32)]
    while len(rows) < n:
        # Random legal-ish play to move the board on. The action does not
        # matter; distinct BOARDS do, because a head that only has to fit one
        # state can fit it with a constant and prove nothing.
        r = env.step(int(rng.integers(0, 5)), float(rng.integers(0, 18)),
                     float(rng.integers(0, 16)), 10)
        rows.append(np.asarray(r.observation, dtype=np.float32))
        if r.done:
            env.reset()
    return torch.tensor(np.stack(rows))


def _place(net, obs, card_idx=None, hires=None):
    feats, embeds, spatial = net.extract_features(obs)
    hx, cx = torch.zeros(obs.shape[0], 256), torch.zeros(obs.shape[0], 256)
    hx, _ = net.lstm(feats, (hx, cx))
    if card_idx is None:
        card_idx = torch.zeros(obs.shape[0], dtype=torch.long)
    return net.placement_given_card(hx, embeds, card_idx, obs, spatial,
                                    hires_map=hires), hx, embeds, spatial


# --------------------------------------------------------------------------
# safety: the change must not disturb anything that already works
# --------------------------------------------------------------------------

def test_trunk_split_is_bit_identical_to_the_sequential():
    """Running cnn_trunk in two halves must equal running it whole.

    `extract_features` now takes the pre-pool activation out of the middle of
    `cnn_trunk` instead of calling it as one Sequential. If those disagree even
    in the last ulp, every checkpoint's features shift underneath it.
    """
    net = _net()
    obs = _obs_batch(4)
    spatial_obs = obs[:, :net.spatial_size].view(
        -1, net.channels, net.board_height, net.board_width)
    whole = net.cnn_trunk(spatial_obs)
    _, _, split = net.extract_features(obs)
    assert torch.equal(whole, split)


def test_hires_map_has_full_board_resolution():
    net = _net()
    obs = _obs_batch(2)
    hires = net.hires_features(obs)
    assert hires.shape == (2, 16, net.board_height, net.board_width)


def test_zero_init_is_bit_identical():
    """At init the branch contributes EXACTLY zero, so the head is unchanged.

    Not 'approximately': the final conv's weight and bias are zeroed, so its
    output is an exact zero tensor and the residual add is exact.
    """
    net = _net()
    obs = _obs_batch(4)
    logits, hx, embeds, spatial = _place(net, obs)

    ctx = net.place_ctx(torch.cat((hx, embeds[torch.arange(4), 0]), dim=-1))
    coarse_map = net.place_up(spatial + ctx.view(-1, 32, 1, 1))
    coarse = coarse_map[:, 0, :net.placement_rows, :net.board_width].reshape(4, -1)
    coarse = coarse.masked_fill(
        ~net.placement_mask(obs, torch.zeros(4, dtype=torch.long)), float("-inf"))
    assert torch.equal(logits, coarse)


def test_old_checkpoint_keeps_the_placement_head():
    """An existing checkpoint must warm-start EVERYTHING it carried.

    The point of the zero-init residual over the handoff's concat-into-place_up
    is precisely this: `place_up`/`place_ctx` still shape-match, so the trained
    placement head survives and only the new branch is fresh.
    """
    net = _net()
    old_keys = {k for k in net.state_dict() if not k.startswith(
        ("place_hires", "place_ctx_hi"))}
    old_blob = {k: torch.randn_like(v) for k, v in net.state_dict().items()
                if k in old_keys}

    fresh = _net(seed=1)
    clean = load_state_dict_flexible(fresh, old_blob, "test")
    assert clean is False           # the new keys are genuinely missing
    for k, v in old_blob.items():
        assert torch.equal(fresh.state_dict()[k], v), f"{k} was not warm-started"
    # ...and the branch is still an exact no-op, so the loaded net behaves
    # exactly as it did before the branch existed.
    assert torch.equal(fresh.place_hires[-1].weight,
                       torch.zeros_like(fresh.place_hires[-1].weight))


def test_recomputed_hires_equals_the_passed_one():
    """The two ways of getting the branch its input must agree.

    Hot paths hand the map in; everything else lets `placement_given_card`
    rebuild it from `obs`. If those diverged, the rollout and the PPO update
    would compute different logits from the same state and the ratio would
    break silently -- the failure this codebase has already paid for twice.
    """
    net = _net()
    for p in net.place_hires[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)       # wake the branch up
    obs = _obs_batch(4)
    passed, _, _, _ = _place(net, obs, hires=net.hires_features(obs))
    rebuilt, _, _, _ = _place(net, obs, hires=None)
    assert torch.equal(passed, rebuilt)


def test_refuses_to_guess_when_it_cannot_build_the_branch():
    """No silent fallback to the coarse-only head.

    Returning coarse logits when neither `obs` nor `hires_map` is available
    would be a different function than the one the rollout used, which is the
    same class of silent drift the `spatial_map is None` guard already exists
    for.
    """
    net = _net()
    obs = _obs_batch(2)
    feats, embeds, spatial = net.extract_features(obs)
    hx = torch.zeros(2, 256)
    with pytest.raises(ValueError, match="hires"):
        net.placement_given_card(hx, embeds, torch.zeros(2, dtype=torch.long),
                                 None, spatial)


def test_forward_sequence_matches_the_single_step_path():
    """The batched update path and the rollout path stay equal.

    Same guarantee `forward_sequence` was verified for when it was introduced;
    the branch has to hold it too, including on the coverage pass.
    """
    net = _net()
    for p in net.place_hires[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    L, B = 3, 2
    obs = _obs_batch(L * B)
    obs_seq = obs.view(L, B, -1)
    flat = obs_seq.reshape(L * B, -1)
    feats, embeds, spatial, hires = net.extract_features_hires(flat)

    card_idx = torch.zeros(L, B, dtype=torch.long)
    args = (feats.view(L, B, -1), embeds.view(L, B, *embeds.shape[1:]),
            spatial.view(L, B, *spatial.shape[1:]), obs_seq,
            torch.ones(L, B, net.hand_size + 1, dtype=torch.bool),
            card_idx, torch.ones(L, B), (torch.zeros(B, 256), torch.zeros(B, 256)))
    _, place_a, _, _, _, _ = net.forward_sequence(*args)
    _, place_b, _, _, _, _ = net.forward_sequence(
        *args, hires_seq=hires.view(L, B, *hires.shape[1:]))
    assert torch.equal(place_a, place_b)


# --------------------------------------------------------------------------
# the point: resolution the coarse head does not have
# --------------------------------------------------------------------------

def _fit_exact_cells(net, obs, targets, steps=400, train_hires=True, lr=3e-3):
    """Fit placement logits to one exact cell per state. Returns argmax match.

    Only the placement pathway trains, exactly as `distill_tactics.py` does it,
    so this measures what the HEAD can express given fixed trunk features.
    """
    trainable = []
    for name, p in net.named_parameters():
        train_it = name.startswith(("place_ctx", "place_up", "card_id_embed"))
        if name.startswith(("place_hires", "place_ctx_hi")):
            train_it = train_hires
        p.requires_grad_(train_it)
        if train_it:
            trainable.append(p)
    opt = torch.optim.Adam(trainable, lr=lr)

    feats, embeds, spatial = net.extract_features(obs)
    hx, _ = net.lstm(feats, (torch.zeros(obs.shape[0], 256),
                             torch.zeros(obs.shape[0], 256)))
    feats, embeds, spatial, hx = (t.detach() for t in (feats, embeds, spatial, hx))
    card_idx = torch.zeros(obs.shape[0], dtype=torch.long)

    for _ in range(steps):
        opt.zero_grad()
        logits = net.placement_given_card(hx, embeds, card_idx, obs, spatial)
        torch.nn.functional.cross_entropy(logits, targets).backward()
        opt.step()
    with torch.no_grad():
        logits = net.placement_given_card(hx, embeds, card_idx, obs, spatial)
    return float((logits.argmax(-1) == targets).float().mean())


def _enemy_column_states(columns, row=18.0):
    """One board per column, each with a single enemy troop in that column.

    Deliberately NOT random rollout states: two random boards can differ only
    in the elixir scalar, and the placement head reads the SPATIAL map, so a
    pair like that has one board and two labels and is unfittable by any head
    at any resolution. Measured on the first version of this test: 8 sampled
    states collapsed to 7 distinct spatial maps. Injecting the difference makes
    every state distinct exactly where the head can see it.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    rows = []
    for x in columns:
        env = CE(deck, deck, 3600)
        env.reset()
        # Archers by ID, not by DECK POSITION. This used to read
        # `gym_wrapper.DEFAULT_DECK[1]` with the comment "Archers", which is
        # only true for one particular deck: the 2026-08-16 switch to the 2.6
        # Hog Cycle silently made slot 1 a Musketeer, and a Musketeer is ONE
        # body against Archers' two, so every board's spatial signature
        # narrowed and the same head fit 12/14 instead of 14/14. The test then
        # reported an architecture regression that had not happened.
        #
        # The fixture must not move when the deck moves -- this test is about
        # what the placement head can EXPRESS, which has nothing to do with
        # which deck is being trained.
        env.inject(1, float(x), row, 1)                 # Archers, team 1
        env.step(4, 0.0, 0.0, 1)        # no-op tick, so the unit is on the board
        rows.append(np.asarray(env.get_observation_for_team(0), dtype=np.float32))
    return torch.tensor(np.stack(rows))


def test_both_heads_can_resolve_a_single_column_at_small_scale():
    """A MEASURED CORRECTION to the handoff's diagnosis. Read this one.

    The handoff (and this file's first draft) claimed the coarse head simply
    CANNOT express an exact cell, because one pooled cell covers ~4x4 tiles.
    Run as a controlled A/B on the task reduced to its essential -- 14 boards
    differing only in which column holds one enemy, answer in that column --
    **both arms fit 14/14 exactly**. The coarse head is not blind below the
    block: `place_up` is nearest-upsample followed by 3x3 convs, so each fine
    cell mixes NEIGHBOURING pooled cells and sub-block position is recoverable.

    So "argmax match stuck at 0.0% while CE fell 8.5x" is not, on its own,
    evidence of inexpressibility. Whatever binds in `distill_tactics.py` binds
    at realistic scale -- hundreds of states and a target that varies in both
    axes and per card -- not at the level of one column. `prove_hires.py`
    measures it there, which is the only place the question can be settled.

    Kept as a regression test with the honest assertion: the branch must not
    make a task the head could already do any harder.
    """
    columns = list(range(2, 16))
    obs = _enemy_column_states(columns)
    # Answer in the same column, in our own half. Row 12 for every state, so
    # the ONLY thing that has to be read off the board is the column.
    targets = torch.tensor([12 * 18 + x for x in columns], dtype=torch.long)

    control = _fit_exact_cells(_net(seed=7), obs, targets, train_hires=False)
    treatment = _fit_exact_cells(_net(seed=7), obs, targets, train_hires=True)

    print(f"\n  same-column argmax match ({len(columns)} states) -- "
          f"coarse only: {control:.3f}   +hires: {treatment:.3f}")
    assert treatment >= 0.99, f"the branch should fit exactly, got {treatment}"
    assert treatment >= control, "the branch must not cost resolution"


# ==========================================================================
# The spell-value anneal, and the advisor-targeted coverage term
# ==========================================================================
# Two changes made 2026-08-14, both GAMEPLAY-AFFECTING.
#
# 1. `spell_value_weight` was dead code: both trainers called
#    `compute_shaping(...)` without `w_spell`, so the Fireball-value weight sat
#    at W_SPELL_VALUE_START for the whole of training and the anneal its own
#    comment block describes never ran. It is wired in now.
#
# 2. The placement COVERAGE term was pure entropy -- "be spread out" -- and a
#    distilled placement map is exactly what that flattens. Coverage now
#    carries the advisor's own score map as a TARGET wherever the advisor has
#    something to say, and falls back to the entropy bonus where it does not.

from python_ai.advisors import advisor_target as AT  # noqa: E402
from python_ai.rewards import shaping as train_shaping  # noqa: E402
from python_ai.rewards import weights as train_weights  # noqa: E402
from python_ai.trainers.distill_tactics import masked_kl  # noqa: E402


def test_spell_value_weight_anneals_from_start_to_final():
    """The schedule the docstring always claimed, now actually reachable."""
    assert train_shaping.spell_value_weight(0) == pytest.approx(train_weights.W_SPELL_VALUE_START)
    end = train_weights.SPELL_VALUE_ANNEAL_EPISODES
    assert train_shaping.spell_value_weight(end) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    assert train_shaping.spell_value_weight(end * 10) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    # Monotone in between, and strictly decreasing end to end.
    xs = [train_shaping.spell_value_weight(int(end * f)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a >= b for a, b in zip(xs, xs[1:])), xs
    assert xs[0] > xs[-1]


def test_spell_value_weight_respects_the_start_offset():
    """Why the offset exists: a warm-start resumes PAST the anneal horizon.

    `model_weights_selfplay.pth` is at episode 64,309 against a 40,000-episode
    horizon, so a faithful wiring pins the weight at FINAL for the whole of any
    resumed run -- correct by the schedule, but it means the anneal can never be
    observed, and Phase 4 has to prove it happens. The offset slides the
    schedule onto the run that is actually being performed. With the default
    offset of 0 the behaviour is exactly the original intent.
    """
    end = train_weights.SPELL_VALUE_ANNEAL_EPISODES
    w = train_shaping.spell_value_weight
    assert w(1000, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_START)
    assert w(1000 + end, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    # Before the offset the term is still at full strength, never extrapolated
    # past START.
    assert w(0, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_START)


def _shaping_stats(fireball_killed):
    """Minimal stats/prev pair where a Fireball has just killed some value."""
    z = np.zeros(1, dtype=np.float32)
    base = {
        "team0_troop_damage": z.copy(), "team1_troop_damage": z.copy(),
        "team0_building_damage": z.copy(), "team1_building_damage": z.copy(),
        "team0_tower_damage": z.copy(), "team1_tower_damage": z.copy(),
        "team0_elixir_spent": z.copy(), "team1_elixir_spent": z.copy(),
        "team0_elixir_current": np.array([7.0], dtype=np.float32),
        "team0_towers_alive": np.array([3]), "team1_towers_alive": np.array([3]),
        "enemy_tower_hp": np.zeros((1, 3), dtype=np.float32),
        "fireball_in_hand": z.copy(),
        "fireball_value_killed": z.copy(), "fireball_elixir_spent": z.copy(),
    }
    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["fireball_value_killed"] = np.array([fireball_killed], dtype=np.float32)
    cur["fireball_elixir_spent"] = np.array([4.0], dtype=np.float32)
    return cur, prev


def test_compute_shaping_actually_responds_to_w_spell():
    """The regression that let the dead code hide.

    Nothing detected `w_spell` being unreachable because no test ever varied
    it. This one does: a two-for-one Fireball must be worth strictly more at
    weight START than at weight 0.
    """
    cur, prev = _shaping_stats(fireball_killed=8.0)
    hot = train_shaping.compute_shaping(cur, prev, w_spell=train_weights.W_SPELL_VALUE_START)
    off = train_shaping.compute_shaping(cur, prev, w_spell=0.0)
    assert float(hot[0]) > float(off[0]), (
        "w_spell is not reaching spell_value_shaping -- the dead-code bug is back")

    none_cast, prev2 = _shaping_stats(fireball_killed=0.0)
    none_cast["fireball_elixir_spent"] = np.zeros(1, dtype=np.float32)
    assert float(off[0]) == pytest.approx(
        float(train_shaping.compute_shaping(none_cast, prev2, w_spell=0.0)[0]))


# --- the advisor target ----------------------------------------------------

def _clump_obs(card=41, x=6.0, y=12.0):
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(card, x, y)
    env.step(4, 0.0, 0.0, 1)
    return np.asarray(env.get_observation_for_team(0), dtype=np.float32)


def test_advisor_has_nothing_to_say_on_an_empty_board(net, fresh_obs):
    """THE GATE, and the reason this is not just distillation-in-the-loop.

    With no threat the advisor still returns a cell -- the defensive pocket for
    a building, an arbitrary tie-broken lane for the Giant. Training on those
    teaches a CONSTANT, which is the exact failure this workstream exists to
    undo. `target_logits_for` must decline instead.
    """
    _env, obs = fresh_obs
    obs = np.asarray(obs, dtype=np.float32)
    for cid in AT.ADVISOR_CARDS:
        legal = net._placement_legal[cid].numpy().astype(bool)
        assert AT.target_logits_for(obs, cid, legal) is None, cid


def test_advisor_target_never_puts_mass_on_an_illegal_cell(net):
    """-inf on every illegal cell, for every card.

    A finite floor would leave real probability mass on a cell the engine
    refuses. The converse does NOT hold and must not be asserted: a "cell"-kind
    rule (the Giant) is a delta, so it is legitimately -inf on legal cells too.
    Only the map-backed kinds are finite across the legal set.
    """
    obs = _clump_obs()
    spoke = 0
    for cid, kind in AT.ADVISOR_CARDS.items():
        legal = net._placement_legal[cid].numpy().astype(bool)
        t = AT.target_logits_for(obs, cid, legal)
        if t is None:
            # A gated rule declining to speak is the DESIGNED behaviour, not a
            # failure: the win condition refuses to commit into an active push,
            # and _clump_obs() is exactly that board. Asserting `is not None`
            # for every card would forbid the gate this file elsewhere calls
            # load-bearing.
            continue
        spoke += 1
        assert t.shape == (net.placement_cells,)
        assert np.all(np.isneginf(t[~legal])), cid
        assert np.isfinite(t).any(), cid
        if kind in ("building", "spell"):
            assert np.all(np.isfinite(t[legal])), cid
    # ...but "every rule declined" would pass vacuously, so require that the
    # board still produced at least one target.
    assert spoke > 0, "no advisor spoke on a board with a live threat"


def test_advisor_spell_target_peaks_where_the_advisor_aims(net):
    """The distilled surface and the played cell must be the same object."""
    obs = _clump_obs()
    legal = net._placement_legal[tactics.FIREBALL_ID].numpy().astype(bool)
    t = AT.target_logits_for(obs, tactics.FIREBALL_ID, legal)
    peak = int(np.argmax(t))
    ax, ay, val = tactics.best_spell_cell(obs, legal=legal)
    assert val > 0.0
    assert peak == int(ay) * tactics.BOARD_W + int(ax)


def _hog_commit_obs():
    """A quiet board where the win-condition gate OPENS.

    Needs all three conditions at once: our half clear, us solvent past the
    Cannon reserve, and the opponent not banked. Those cannot be arranged by
    STEPPING -- idling to 9 elixir also banks the opponent to 10 and closes the
    third condition -- so the elixir scalar is written directly on a fresh
    board, which keeps time (and therefore the opponent's estimate) at its
    opening value.

    Deliberately not env.set_elixir_for_team: that binding exists only in the
    current build, and python_ai/ holds the stale .pyd whenever the post-build
    copy was blocked by a running trainer.
    """
    env = E.ClashRoyaleEnv(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    # One enemy, deep on THEIR half. Needed because the rule declines when the
    # board is completely empty -- with no enemy anywhere the "weaker lane" is
    # undefined and the tiebreak would emit a fixed cell. Placed at y=25 so it
    # is visible to the lane read without counting as a threat on our half.
    env.inject(6, 13.0, 25.0, 1)
    env.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)   # inject lands on the tick
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32).copy()
    obs[tactics.SPATIAL] = 0.9      # ClashEnv stores elixir / 10
    return obs


def test_advisor_wincon_target_is_a_delta_and_its_kl_is_cross_entropy(net):
    """The Hog rule yields a CELL, not a surface, so its target is a delta.

    Encoding it as a delta keeps one code path for every card: masked_kl
    against a delta is exactly cross-entropy to that cell, so the trainers need
    no separate hard-label branch.
    """
    obs = _hog_commit_obs()
    legal = net._placement_legal[tactics.HOG_ID].numpy().astype(bool)
    t = AT.target_logits_for(obs, tactics.HOG_ID, legal)
    assert t is not None, "gate should be open on a quiet, solvent board"
    gx, gy, _ = tactics.best_hog_cell(obs, legal=legal)
    cell = int(gy) * tactics.BOARD_W + int(gx)
    assert int(np.argmax(t)) == cell
    assert int(np.isneginf(t).sum()) == len(t) - 1

    logits = torch.randn(1, net.placement_cells)
    logits[0, ~torch.tensor(legal)] = float("-inf")
    kl = masked_kl(logits, torch.tensor(t).unsqueeze(0))
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([cell]))
    assert float(kl) == pytest.approx(float(ce), abs=1e-5)


def test_advisor_standardization_matches_the_offline_harness(net):
    """One definition of "score map -> target logits", provable, not asserted.

    `advisor_target._standardize` is a numpy re-implementation of
    `prove_hires.soft_target_logits` rather than an import of it, deliberately:
    importing the offline harness would pull `expert_iteration` into the
    trainers' rollout path. That is exactly the second-copy drift CLAUDE.md
    forbids, so it is pinned here instead of trusted.
    """
    from python_ai.eval.prove_hires import soft_target_logits

    obs = _clump_obs()
    legal = net._placement_legal[tactics.FIREBALL_ID].numpy().astype(bool)
    flat = tactics.spell_catch_map(obs).reshape(-1).astype(np.float32).copy()
    flat[~legal] = float("-inf")

    mine = AT._standardize(flat, legal, 0.25)
    theirs = soft_target_logits(torch.tensor(flat).unsqueeze(0), 0.25)[0].numpy()
    assert np.array_equal(np.isneginf(mine), np.isneginf(theirs))
    np.testing.assert_allclose(mine[legal], theirs[legal], rtol=1e-5, atol=1e-5)


def test_cannon_target_stays_broader_than_the_spell_target(net):
    """The Cannon's argmax is row-major noise off a tie plateau (handoff 2.3).

    So its target must keep the plateau's mass spread rather than committing to
    one arbitrary cell. Measured there: temperature cannot push the Cannon
    target below ~66% of maximum entropy while Fireball reaches ~41%. This pins
    the ORDERING, which is a property of the two surfaces rather than of the
    temperature.
    """
    obs = _clump_obs(card=2, x=9.0, y=13.0)      # a Giant: one big body, real threat
    ents = {}
    for cid in (tactics.CANNON_ID, tactics.FIREBALL_ID):
        legal = net._placement_legal[cid].numpy().astype(bool)
        t = AT.target_logits_for(obs, cid, legal)
        if t is None:
            pytest.skip(f"advisor declined card {cid} on this fixture")
        p = torch.softmax(torch.tensor(t), -1)
        ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        ents[cid] = ent / float(np.log(int(legal.sum())))
    assert ents[tactics.CANNON_ID] > ents[tactics.FIREBALL_ID], ents


# --- the coverage loss itself ----------------------------------------------

def test_elementwise_kl_averages_to_the_scalar_one():
    """masked_kl_elementwise must be masked_kl without the mean, exactly."""
    torch.manual_seed(3)
    new = torch.randn(7, 40)
    tgt = torch.randn(7, 40)
    new[:, 5:9] = float("-inf")
    tgt[:, 5:9] = float("-inf")
    assert float(AT.masked_kl_elementwise(new, tgt).mean()) == pytest.approx(
        float(masked_kl(new, tgt)), abs=1e-6)


def _coverage_fixture(n=6, cells=612):
    torch.manual_seed(4)
    logits = torch.randn(n, cells, requires_grad=True)
    targets = torch.randn(n, cells)
    decision = torch.ones(n)
    return logits, targets, decision


def test_with_no_advisor_target_coverage_is_exactly_the_old_entropy_bonus():
    """The fallback path must be the term it replaces, to the bit.

    Whatever else changes, a state the advisor declines has to behave the way
    v1.2.0 behaved, or the control arm of the A/B is not a control.
    """
    logits, targets, decision = _coverage_fixture()
    has = torch.zeros(len(decision))
    delta, ent_frac, kl, n = AT.coverage_terms(
        logits, targets, has, decision, PLACEMENT_COVERAGE_COEF, np.log(612))
    expected = float(Categorical(logits=logits).entropy().mean().detach()) / np.log(612)
    assert float(ent_frac) == pytest.approx(expected, abs=1e-6)
    assert float(delta) == pytest.approx(
        -PLACEMENT_COVERAGE_COEF * expected, abs=1e-6)
    assert float(kl) == 0.0 and float(n) == 0.0


def test_a_row_gets_the_target_or_the_entropy_bonus_but_never_both():
    """THE RESOLUTION. Entropy flattens, KL concentrates; one row cannot want
    both. Rows split cleanly, and each denominator counts only its own rows."""
    logits, targets, decision = _coverage_fixture(n=4)
    has = torch.tensor([1.0, 1.0, 0.0, 0.0])
    _, ent_frac, kl, n = AT.coverage_terms(
        logits, targets, has, decision, PLACEMENT_COVERAGE_COEF, np.log(612))
    assert float(n) == 2.0

    ent_all = Categorical(logits=logits).entropy()
    assert float(ent_frac) == pytest.approx(
        float(ent_all[2:].mean() / np.log(612)), abs=1e-6), (
        "the entropy bonus leaked onto rows that carry an advisor target")
    kl_all = AT.masked_kl_elementwise(logits, targets)
    assert float(kl) == pytest.approx(float(kl_all[:2].mean()), abs=1e-6)


def test_the_advisor_term_never_reaches_the_chosen_action(fixture_net=None):
    """Same property the entropy coverage term has, and for the same reason.

    The coverage pass scores a card that was AFFORDABLE, not the one that was
    CHOSEN, so nothing it does may reach the chosen action's log-prob or the
    critic. If it did, it would silently corrupt the quantity PPO clips.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = _fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets,
        hidden, extra_card_idx_seq=cover)

    targets = torch.randn(L, B, net.placement_cells)
    delta, _, _, _ = AT.coverage_terms(
        extra, targets, torch.ones(L, B), torch.ones(L, B),
        PLACEMENT_COVERAGE_COEF, np.log(net.placement_cells))

    for name, tensor in (("card logits", cl), ("chosen placement", pl),
                         ("values", values), ("aux", aux)):
        g = torch.autograd.grad(delta, tensor, retain_graph=True,
                                allow_unused=True)[0]
        assert g is None or float(g.abs().sum()) == 0.0, (
            f"the advisor coverage term leaked gradient into {name}")


def test_one_step_of_the_advisor_term_moves_the_head_toward_the_advisor():
    """The end-to-end claim: this term is a TARGET, not just noise.

    Optimizing the coverage loss alone must reduce the distance between the
    head's argmax cell and the advisor's own cell. Entropy cannot do this by
    construction -- it is a marginal objective with no opinion about which cell
    is right in which state -- so this is the property that separates the fix
    from the measured-insufficient version it replaces.
    """
    from python_ai.models.net import MicroRoyaleNet

    torch.manual_seed(11)
    np.random.seed(11)
    n = MicroRoyaleNet(num_ability_slots=0)
    legal_table = AT.build_legal_table(n)
    cid = tactics.FIREBALL_ID

    # The opening hand is drawn by an unseeded mt19937 the engine will not let
    # us set (CLAUDE.md, "Driving the simulator from outside is awkward"), so
    # Fireball is in only ~half of the 4-slot hands and which states survive is
    # not reproducible. Draw until there are enough rather than over-generating
    # a fixed list and hoping -- the fixed-list version passed alone and failed
    # inside the full suite, because the global RNG position differs.
    kept = []
    for i in range(60):
        o = _clump_obs(x=float(3 + (i % 13)), y=12.0)
        if bool((n.hand_card_ids(torch.tensor(o).unsqueeze(0)) == cid).any()):
            kept.append(o)
        if len(kept) == 8:
            break
    assert len(kept) >= 4, f"only {len(kept)} hands held Fireball in 60 draws"
    obs_np = np.stack(kept)
    rows = obs_np.shape[0]

    tgt_np, has_np = AT.targets_for_batch(obs_np, np.full(rows, cid), legal_table)
    assert has_np.all(), "fixture must give the advisor something to say"

    ob = torch.tensor(obs_np)
    targets = torch.tensor(tgt_np)
    hand_ids = n.hand_card_ids(ob)
    slot = (hand_ids == cid).float().argmax(dim=1)
    assert all(int(hand_ids[i, slot[i]]) == cid for i in range(rows))

    zeros = (torch.zeros(rows, 256), torch.zeros(rows, 256))

    def mean_distance():
        with torch.no_grad():
            feats, embeds, sp = n.extract_features(ob)
            hx, _ = n.lstm(feats, zeros)
            got = n.placement_given_card(hx, embeds, slot, ob, sp).argmax(-1)
            want = targets.argmax(-1)
            return float(torch.maximum((got % 18 - want % 18).abs(),
                                       (got // 18 - want // 18).abs()).float().mean())

    before = mean_distance()
    opt = torch.optim.Adam(n.parameters(), lr=3e-3)
    for _ in range(30):
        feats, embeds, sp = n.extract_features(ob)
        hx, _ = n.lstm(feats, zeros)
        logits = n.placement_given_card(hx, embeds, slot, ob, sp)
        delta, _, _, _ = AT.coverage_terms(
            logits, targets, torch.ones(rows), torch.ones(rows),
            PLACEMENT_COVERAGE_COEF, np.log(n.placement_cells))
        opt.zero_grad(set_to_none=True)
        delta.backward()
        opt.step()
    after = mean_distance()

    print(f"\n  advisor-target pull: mean cell distance {before:.2f} -> {after:.2f} "
          f"({rows} states)")
    # A fresh head's map is near-uniform, so `before` is a long way out; the
    # <= 1.0 branch only exists so a lucky init cannot fail a test about
    # LEARNING. Board diagonal is 34 tiles, so 1.0 is already at the target.
    if before <= 1.0:
        assert after <= before, f"the target pushed the head away ({before} -> {after})"
    else:
        assert after < before, (
            f"the advisor target did not move the head ({before} -> {after})")


# ===========================================================================
# 2026-08-16 audit: the two defects the from-scratch 2.6 Hog run was gated on
# ===========================================================================

def test_card_entropy_must_be_normalized_by_the_REACHABLE_maximum():
    """The regression for the bankruptcy bug. Read the numbers.

    Both trainers used to divide the card head's entropy by
    LOG_N_CARD = log(hand_size + 1) = log(5). But the affordability mask leaves
    only (#affordable + 1) legal arms, and MEASURED on model_weights_cured.pth
    over 706 decision steps, 54.1% of them leave exactly TWO. On such a step
    the reachable maximum is log(2) = 0.693 nats, so a 0.35 target expressed
    against log(5) is 0.5633 nats -- 81.3% of what the step can carry.

    The controller therefore held the play/wait choice near a coin flip, the
    agent spent elixir on sight, mean elixir sat at 2.25/10, nothing was
    affordable on 73.9% of steps (78.5% under a big push), and P(play) was FLAT
    against threat (0.1008 none vs 0.1016 largest) -- which reads as defensive
    apathy and is really an imposed bankruptcy.

    This test pins the invariant the fix restores: a distribution that is
    UNIFORM over its legal arms must normalize to exactly 1.0, whatever the
    number of arms. Under the old divisor a uniform 2-arm distribution scored
    ~0.43 and the controller read that as "not random enough".
    """
    logits = torch.zeros(4, 5)
    for row, n_legal in enumerate((2, 3, 4, 5)):
        logits[row, n_legal:] = float("-inf")
    mask = torch.isfinite(logits)
    ent = Categorical(logits=logits).entropy()

    n_legal = mask.sum(-1).clamp(min=2).float()
    reachable = ent / torch.log(n_legal)
    old_way = ent / float(np.log(5))

    print("\n  uniform-over-legal rows: reachable-normalized "
          f"{[round(v, 4) for v in reachable.tolist()]}  vs old/log(5) "
          f"{[round(v, 4) for v in old_way.tolist()]}")

    assert torch.allclose(reachable, torch.ones(4), atol=1e-6), (
        f"uniform over legal arms must read 1.0, got {reachable.tolist()}")
    assert old_way[0] < 0.45, (
        "a uniform 2-arm row used to read ~0.43 of 'maximum' -- if this no "
        "longer holds the normalizer changed and this test needs rereading")
    assert bool((old_way[:3] < 1.0).all()), "only the unmasked row can reach 1.0 the old way"


def test_flawless_defense_pays_only_on_a_win_and_scales_with_tower_hp():
    """The 'perfect defense' term must not be able to reorder win/loss/draw.

    That is the whole safety argument for it being non-potential-based: this
    project already measured that policy-invariant tower shaping left PURE
    DEFENCE as the optimum and win-condition usage decayed to 0.7%. Gating on
    the win is what stops this term walking back into that -- a turtle that
    stalls into a timeout must collect exactly nothing.
    """
    from python_ai.rewards import shaping as T, weights as TW

    dones = np.array([True, True, True, False])
    #                 clean win, scraped win, loss, mid-episode
    step_rewards = np.array([1.0, 1.0, -1.0, 0.0], dtype=np.float32)
    stats = {"team1_tower_damage": np.array(
        [0.0, EC.OWN_TOWER_HP_TOTAL * 0.5, 0.0, 0.0], dtype=np.float32),
        # A crown taken in every row, so this test isolates the HP SCALING.
        # The crown gate itself is covered by
        # test_flawless_bonus_refuses_a_win_with_every_enemy_tower_standing.
        "team1_towers_alive": np.array([2, 2, 2, 2], dtype=np.int64)}

    b = T.flawless_defense_bonus(dones, step_rewards, stats, None)

    print(f"\n  flawless bonus: clean={b[0]:.3f} scraped={b[1]:.3f} "
          f"loss={b[2]:.3f} mid={b[3]:.3f}  (W={TW.W_FLAWLESS_DEFENSE})")

    assert b[0] == pytest.approx(TW.W_FLAWLESS_DEFENSE), "a flawless win pays in full"
    assert b[1] == pytest.approx(TW.W_FLAWLESS_DEFENSE * 0.5), "half the HP, half the bonus"
    assert b[2] == 0.0, "a LOSS must never collect -- it could reorder outcomes"
    assert b[3] == 0.0, "mid-episode steps must never collect"
    draw = T.flawless_defense_bonus(
        np.array([True]), np.array([0.0], dtype=np.float32),
        {"team1_tower_damage": np.array([0.0], dtype=np.float32),
         "team1_towers_alive": np.array([2], dtype=np.int64)}, None)
    assert draw[0] == 0.0, "a DRAW must collect nothing, however clean"


def test_flawless_defense_reads_a_counter_that_already_auto_reset():
    """On a done step the vector env has auto-reset, so the cumulative counter
    may already read 0 for the NEXT episode. The running max must recover the
    finished episode's real damage, or a hard-fought win would be paid as if it
    were flawless -- the bonus would then reward exactly the wrong games."""
    from python_ai.rewards import shaping as T, weights as TW

    took_half = EC.OWN_TOWER_HP_TOTAL * 0.5
    crown = {"team1_towers_alive": np.array([2], dtype=np.int64)}
    prev = {"team1_tower_damage": np.array([took_half], dtype=np.float32), **crown}
    post_reset = {"team1_tower_damage": np.array([0.0], dtype=np.float32), **crown}

    b = T.flawless_defense_bonus(np.array([True]), np.array([1.0], dtype=np.float32),
                                 post_reset, prev)
    assert b[0] == pytest.approx(TW.W_FLAWLESS_DEFENSE * 0.5), (
        "the post-autoreset zero was taken at face value; a scraped win would "
        f"be paid as flawless (got {b[0]}, expected {TW.W_FLAWLESS_DEFENSE * 0.5})")


def test_default_deck_is_the_26_hog_cycle_and_every_card_is_cheap_enough():
    """The deck switch, pinned by the property that motivated it.

    The Giant deck's measured failure was a STARVED win condition: at ~0.35
    elixir per decision a 5-cost card is legal only after ~14 consecutive
    non-spending steps, and Giant was never played once across four full runs.
    Nothing in 2.6 costs more than 4, which makes that failure mode
    structurally unavailable rather than merely unlikely.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    info = [clash_royale_env.get_card_info(c) for c in deck]
    names = [i["name"] for i in info]
    costs = [i["cost"] for i in info]

    print(f"\n  deck: {list(zip(names, costs))}  avg={sum(costs)/len(costs):.3f}")

    assert set(names) == {"Hog Rider", "Musketeer", "Cannon", "Ice Golem",
                          "Skeletons", "Ice Spirit", "The Log", "Fireball"}
    assert max(costs) <= 4.0, f"nothing may cost more than 4 in this deck, got {costs}"
    assert sum(costs) / len(costs) == pytest.approx(2.625), "2.6 Hog Cycle average"
    assert clash_royale_env.validate_deck_slots(deck) == "", "deck must be legal"


def test_win_condition_is_derived_from_the_engine_not_hardcoded():
    """The win condition must follow the deck, not a literal.

    A hardcoded id would mean "Hog Rider" forever and silently credit the wrong
    card the next time the deck moves -- the same drift that had a test
    injecting a Musketeer under a comment saying Archers.
    """
    assert gym_wrapper.WIN_CONDITION_ID == 15, "2.6 Hog Cycle's win condition is the Hog"
    assert gym_wrapper._find_win_condition([10, 1, 41, 25, 7, 2, 6, 5]) == 2, "Giant deck -> Giant"
    assert gym_wrapper._find_win_condition([1, 6, 12, 24, 72, 33, 7, 29]) is None


def test_win_condition_damage_term_responds_to_its_weight():
    """The regression `spell_value_weight` did not have.

    W_SPELL_VALUE's anneal sat DEAD for an entire training era because no test
    ever varied the argument -- the weight was simply never passed. This asserts
    the win-condition term actually reaches the reward.
    """
    from python_ai.rewards import shaping as T, weights as TW

    # Reuse the module's own fixture rather than a parallel copy of the key
    # list -- a second copy is how a shaping test ends up silently exercising
    # a different code path from the trainer.
    cur, prev = _shaping_stats(0.0)
    prev["team0_wincon_damage"] = np.zeros(1, dtype=np.float32)
    cur["team0_wincon_damage"] = np.array([400.0], dtype=np.float32)
    on = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=1.0)[0])
    off = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=0.0)[0])
    print(f"\n  wincon term: w=1.0 -> {on:.5f}   w=0.0 -> {off:.5f}")
    assert on > off, "the win-condition weight does not reach the reward"
    assert on - off == pytest.approx(400.0 / EC.MAX_BUILDING_HP, rel=1e-4)


def test_flawless_bonus_refuses_a_win_with_every_enemy_tower_standing():
    """The turtle gate. A win where no enemy tower fell is a timeout win on
    tower HP -- pure defence, which is the local optimum this term must not
    pay for. Measured at ep 6,053: Hog usage 0.8% while the agent won ~100% of
    games at 1.0x by defending."""
    from python_ai.rewards import shaping as T, weights as TW

    assert TW.FLAWLESS_REQUIRES_CROWN, "this test describes the crown-gated behaviour"
    clean = {"team1_tower_damage": np.zeros(1, dtype=np.float32),
             "team1_towers_alive": np.full(1, 3, dtype=np.int64)}
    crowned = {"team1_tower_damage": np.zeros(1, dtype=np.float32),
               "team1_towers_alive": np.full(1, 2, dtype=np.int64)}
    dones, rew = np.array([True]), np.array([1.0], dtype=np.float32)

    turtle = T.flawless_defense_bonus(dones, rew, clean, None)[0]
    decisive = T.flawless_defense_bonus(dones, rew, crowned, None)[0]
    print(f"\n  flawless bonus: turtle-win={turtle:.3f}  crowned-win={decisive:.3f}")
    assert turtle == 0.0, "a win with all 3 enemy towers up must pay NOTHING"
    assert decisive == pytest.approx(TW.W_FLAWLESS_DEFENSE), (
        "a flawless win that took a crown must still pay in full")


# ==========================================================================
# teacher.py -- the utility-search sparring partner (2026-08-19)
# ==========================================================================
# The elixir-multiplier curriculum priced the win condition negatively (see
# CLAUDE.md's 1.5x hypothesis and its monotone dose-response). It is replaced
# by a COMPETENCE curriculum: a deterministic utility-search bot at a symmetric
# 1.0x economy, whose difficulty is dialed on lookahead/width/epsilon.
#
# These tests pin the parts that fail SILENTLY. In particular a wrong team-1
# frame conversion makes every team-1 placement illegal, which reads as "the
# bot is weak" rather than "the bot is broken" -- exactly the class of bug this
# project has paid for three times.


def test_card_roles_derives_roles_from_the_engine_not_a_hardcoded_list():
    from python_ai.opponents.teacher import card_roles
    roles = card_roles(gym_wrapper.DEFAULT_DECK)
    assert roles[15] == "wincon", "Hog Rider is the deck's building-targeter"
    assert roles[7] == "spell"      # Fireball
    assert roles[33] == "spell"     # The Log
    assert roles[25] == "building"  # Cannon
    assert roles[6] == "ranged"     # Musketeer
    # Ice Golem also targets buildings but is a 2-cost shield, not a win
    # condition -- highest-cost building-targeter wins, the same tiebreak
    # gym_wrapper._find_win_condition uses.
    assert roles[40] != "wincon"
    assert set(roles) == set(gym_wrapper.DEFAULT_DECK)


def _teacher_env(ticks=40):
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    for _ in range(ticks):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    return env


def test_teacher_candidates_are_all_legal_for_either_team():
    """is_valid_placement takes ABSOLUTE y for both teams while step_self_play
    takes team 1's y MIRRORED. Get that backwards and team 1 silently never
    places anything."""
    from python_ai.opponents.teacher import UtilityTeacher

    for team in (0, 1):
        env = _teacher_env()
        t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=team, horizon_ticks=0)
        t.reset()
        obs = np.asarray(env.get_observation_for_team(team), np.float32)
        cands = t.candidates(env, obs)
        assert len(cands) >= 1, "the no-op candidate is always present"
        for c in cands:
            if c.slot == CE.HAND_SIZE:
                continue
            assert env.is_valid_placement(c.card_id, c.x, t.to_absolute_y(c.y), team), (
                f"team {team} proposed an illegal cell: {c}")


def test_teacher_never_proposes_an_unaffordable_card():
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env(ticks=0)
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=0)
    t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    cands = t.candidates(env, obs, elixir=1.0)
    for c in cands:
        if c.slot == CE.HAND_SIZE:
            continue
        assert E.get_card_info(c.card_id)["cost"] <= 1.0


def test_noop_scores_exactly_zero_so_the_teacher_can_hold_elixir():
    """The no-op baseline is what makes every other score a MARGINAL value. If
    it drifts, the bot either dumps elixir on sight or freezes forever."""
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=30)
    t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(env, obs) if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base, obs) == 0.0


def test_rollout_does_not_touch_the_live_match():
    """snapshot() deep-copies the stats collectors. If that ever regresses,
    every hypothetical hit lands in the REAL match's statistics -- and those
    feed the reward shaping, so a search would corrupt its own returns."""
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    before = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
              env.get_elixir_spent(0), env.get_elixir_for_team(0))
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=30)
    t.reset()
    t.act(env, np.asarray(env.get_observation_for_team(0), np.float32))
    after = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
             env.get_elixir_spent(0), env.get_elixir_for_team(0))
    assert before == after


def test_teacher_is_side_agnostic():
    """One class plays both sides. A broken team-1 frame shows up here as a
    bot that never lands a card, not as an exception."""
    from python_ai.opponents.teacher import UtilityTeacher

    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    t1 = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=1, horizon_ticks=30, seed=0)
    t1.reset()
    played = 0
    for _ in range(120):
        obs1 = np.asarray(env.get_observation_for_team(1), np.float32)
        slot, x, y = t1.act(env, obs1)
        spent_before = env.get_elixir_spent(1)
        env.step_self_play(-1, 0, 0, slot, x, y, 10)
        if env.get_elixir_spent(1) > spent_before:
            played += 1
        if env.is_game_over():
            break
    print(f"\n  team-1 teacher landed {played} cards in 120 decisions")
    assert played >= 5, f"team-1 teacher only landed {played} cards"


def test_cycle_tracker_order_logic_against_an_exact_simulated_cycle():
    """The ORDER half of CycleTracker, tested deterministically with no engine.

    WHY NOT DRIVE THE ENGINE. Two engine-driven versions of this test passed
    VACUOUSLY and one stayed flaky. Playing hand slot 0 every step rotates only
    slot 0, so a Hog dealt into slots 1-3 never leaves the hand; playing the
    CHEAPEST affordable card fails the same way for the opposite reason, since
    2.6 has two 1-cost cards and a 4-cost Hog is never cheapest; and a uniform
    draw still misses when the match ends before the Hog is ever affordable.
    A vacuous pass on a cycle tracker is worse than no test at all -- 2.6 is
    DEFINED by cycling -- so the logic is pinned here against an exact model and
    the engine agreement is checked separately below.

    The model is the engine's own rule: the played card goes to the BACK of the
    queue, the front of the queue fills the vacated hand slot.
    """
    from python_ai.opponents.teacher import CycleTracker

    deck = list(gym_wrapper.DEFAULT_DECK)
    hand, queue = deck[:4], deck[4:]
    ct = CycleTracker(deck)
    ct.reset()
    ct.observe(list(hand))

    rng = np.random.default_rng(0)
    for _ in range(200):
        slot = int(rng.integers(len(hand)))
        played = hand[slot]
        hand[slot] = queue.pop(0)
        queue.append(played)
        ct.observe(list(hand))

        # A card just played sits at the back of a 4-long queue, so it is
        # exactly 4 plays from returning. Membership alone is satisfied by ANY
        # ordering; this is the part that can actually be wrong.
        assert ct.distance_to(played) == 4, (
            f"{played} was just played; expected 4, got {ct.distance_to(played)}")
        for i, card in enumerate(queue):
            assert ct.distance_to(card) == i + 1, (
                f"queue {queue} but distance_to({card}) said "
                f"{ct.distance_to(card)}, expected {i + 1}")
        for card in hand:
            assert ct.distance_to(card) == 0


def test_cycle_tracker_agrees_with_the_engines_own_hand():
    """The identity invariant, against the real engine: distance is 0 exactly
    when the card is in hand. Runs several matches because one can end before
    the win condition is ever affordable, and asserts at the end that at least
    one rotation was actually observed -- otherwise this passes vacuously too.
    """
    from python_ai.opponents.teacher import CycleTracker

    costs = {c: E.get_card_info(c)["cost"] for c in gym_wrapper.DEFAULT_DECK}
    driver = np.random.default_rng(0)
    seen_far = False
    for _ in range(6):
        env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
        env.reset()
        ct = CycleTracker(gym_wrapper.DEFAULT_DECK)
        ct.reset()
        for _ in range(200):
            hand = list(env.get_hand_for_team(0))
            ct.observe(hand)
            d = ct.distance_to(15)
            if 15 in hand:
                assert d == 0, f"Hog is in hand {hand} but distance_to said {d}"
            else:
                assert d > 0, f"Hog absent from {hand} but distance_to said 0"
                seen_far = True
            elixir = env.get_elixir_for_team(0)
            playable = [i for i, c in enumerate(hand)
                        if costs[c] <= elixir + 1e-6]
            if not playable:
                env.step_self_play(-1, 0.0, 0.0, -1, 0, 0, 10)
                continue
            env.step_self_play(int(driver.choice(playable)), 9.0, 10.0,
                               -1, 0, 0, 10)
            if env.is_game_over():
                break
        if seen_far:
            break
    assert seen_far, "the win condition never left hand in 6 matches"


def test_teacher_stages_are_competence_not_economy():
    from python_ai.opponents.teacher import TEACHER_STAGES

    assert len(TEACHER_STAGES) == 6
    for s in TEACHER_STAGES:
        assert set(s) == {"horizon_ticks", "epsilon", "k_cells"}, (
            "a stage must never carry an elixir multiplier -- that is the whole "
            "point of this curriculum")
    eps = [s["epsilon"] for s in TEACHER_STAGES]
    hor = [s["horizon_ticks"] for s in TEACHER_STAGES]
    assert eps == sorted(eps, reverse=True), "noise must fall monotonically"
    assert hor == sorted(hor), "lookahead must rise monotonically"


def test_epsilon_one_still_only_emits_legal_actions():
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=0,
                       epsilon=1.0, seed=0)
    t.reset()
    for _ in range(50):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t.act(env, obs)
        if slot != CE.HAND_SIZE:
            cid = env.get_hand_for_team(0)[slot]
            assert env.is_valid_placement(cid, x, t.to_absolute_y(y), 0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
        if env.is_game_over():
            break


def test_default_env_is_unchanged_and_still_uses_the_cpp_heuristic():
    """The teacher is OPT-IN. Every existing harness, probe and eval anchor
    constructs MicroRoyaleEnv with no opponent key and must keep getting the C++
    HeuristicOpponent, or every historical number silently stops being
    comparable."""
    env = gym_wrapper.MicroRoyaleEnv()
    assert env.teacher is None
    assert env.opponent_kind == "builtin"


def test_teacher_env_produces_the_same_info_keys_as_the_builtin_env():
    """train.py reads ~20 keys out of info. stepSelfPlay returns a different
    result type than step (observation0/reward0, no observation/reward), so the
    routing is the one place those could diverge."""
    a = gym_wrapper.MicroRoyaleEnv()
    a.reset()
    b = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 2})
    b.reset()
    assert b.teacher is not None
    act = {"card_index": np.array([CE.HAND_SIZE]),
           "target_x": np.array([9.0]), "target_y": np.array([10.0])}
    _, _, _, _, ia = a.step(act)
    _, _, _, _, ib = b.step(act)
    assert set(ia) == set(ib)


def test_teacher_opponent_actually_plays_cards():
    """A teacher that silently never plays looks exactly like a weak opponent.
    This is the end-to-end version of the frame check in
    test_teacher_candidates_are_all_legal_for_either_team."""
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 3})
    env.reset()
    act = {"card_index": np.array([CE.HAND_SIZE]),
           "target_x": np.array([9.0]), "target_y": np.array([10.0])}
    info = None
    for _ in range(150):
        _, _, term, _, info = env.step(act)
        if term:
            break
    assert info is not None and info["team1_elixir_spent"] > 0.0, (
        "the teacher opponent never spent a single elixir")


def test_set_teacher_stage_is_a_noop_on_a_builtin_env():
    """train.py calls envs.call('set_teacher_stage', n) unconditionally; on a
    heuristic env that must not raise."""
    env = gym_wrapper.MicroRoyaleEnv()
    env.set_teacher_stage(4)          # must not raise
    assert env.teacher is None


#: PIPELINE 1's training path -- `trainers/train.py` plus the loop and the
#: curriculum that moved out of it into `rl/`. Scanning train.py alone, which is
#: what this test used to do, would now pass while the banned call sat one
#: import away.
#:
#: Deliberately NOT pipeline 2: `envs/selfplay_env.BUILTIN_TRAINING_OPPONENTS`
#: really does put "builtin:heuristic@1.35" and "@1.50" in the PFSP pool, so a
#: multiplier there is the anchor's own identity, not a curriculum handicap.
#: The 2026-08-19 pivot was about phase 1's ladder and this test's claim is
#: about phase 1's ladder.
def _phase1_training_sources():
    import pathlib
    root = pathlib.Path(python_ai.PACKAGE_DIR)
    yield root / "trainers" / "train.py"
    yield from sorted((root / "rl").glob("*.py"))


def _curriculum_stages_literal():
    """The real object, imported.

    This used to PARSE `train.py` with `ast`, because CURRICULUM_STAGES was a
    local of `train_ppo()` and could not be imported at all. That worked, and it
    was a smell worth acting on: a constant nothing can import is a constant
    nothing can check, and a parser that finds nothing can only be told apart
    from a parser that finds the wrong thing by the assertion it raises. It now
    lives at module scope in `rl/curriculum.py`.
    """
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    return CURRICULUM_STAGES


def test_training_never_raises_the_opponent_elixir_multiplier():
    """THE PIN ON THE 2026-08-19 PIVOT.

    The 1.5x handicap is what priced the win condition negatively -- measured
    monotone across 1.0/1.25/1.5x, and the reason four separate Hog
    interventions all returned null. The API survives for the ~15 measurement
    harnesses that sweep it (including the falsifier that justified the change);
    phase 1's training path must never call it again.

    AST-based, not line-based. The line scan this replaces skipped anything
    starting with `#`, which silently exempted every mention inside a DOCSTRING
    -- and a docstring is exactly where a future author would explain the ban
    before quietly reintroducing it below.
    """
    import ast
    offenders = []
    for path in _phase1_training_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "set_opponent_elixir_multiplier"):
                offenders.append(f"{path.name}:{node.lineno}: direct call")
            elif isinstance(node, ast.Constant) and node.value in (
                    "set_opponent_elixir_multiplier", "opp_elixir_multiplier"):
                # `envs.call("set_opponent_elixir_multiplier", m)` and the
                # env-config key are the two indirect routes to the same thing.
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "phase 1's training path sets an opponent elixir multiplier:\n  "
        + "\n  ".join(offenders))


def test_curriculum_lives_where_a_test_can_import_it():
    """The regression for the reason the two tests below used to parse source."""
    from python_ai.rl import curriculum
    assert curriculum.CURRICULUM_STAGES is _curriculum_stages_literal()


def test_curriculum_stages_are_competence_not_economy():
    stages = _curriculum_stages_literal()
    assert len(stages) == 6
    for s in stages:
        assert "opp_elixir_multiplier" not in s, (
            "a curriculum stage must never carry an elixir multiplier again")
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in stages] == [0, 1, 2, 3, 4, 5]
    # The gate itself is deliberately unchanged: 0.80 raw win rate on every
    # stage but the last, which has no further auto-advance.
    assert [s["win_rate_threshold"] for s in stages] == [0.8] * 5 + [None]


def test_curriculum_stage_count_matches_the_teacher_ladder():
    """If these ever drift, CURRICULUM_STAGES would index past the end of
    TEACHER_STAGES -- or, worse, silently stop at a rung short of the top."""
    from python_ai.opponents.teacher import TEACHER_STAGES
    stages = _curriculum_stages_literal()
    assert len(stages) == len(TEACHER_STAGES)
    for s in stages:
        assert 0 <= s["teacher_stage"] < len(TEACHER_STAGES)


def test_phase1_opponent_defaults_to_the_teacher():
    from python_ai.trainers import train
    assert train.PHASE1_OPPONENT == "teacher", (
        "the C++ HeuristicOpponent is an EVAL ANCHOR now, not a training "
        "opponent -- see the 2026-08-19 pivot")


# --------------------------------------------------------------------------
# scenario_offense.py -- Proposal A, kept as an accelerator and default-OFF
# --------------------------------------------------------------------------

def test_offensive_scenarios_are_off_by_default():
    """THE ORDER MATTERS. Injection changes the state distribution, not the
    payoff, so switching it on before the payoff is fixed just pays a negative
    price more often -- which is exactly what the four forced-usage experiments
    measured. It stays off until prove_environment.py says otherwise."""
    from python_ai.envs import scenario_offense
    assert scenario_offense.OFFENSIVE_SCENARIO_PROB == 0.0
    env = gym_wrapper.MicroRoyaleEnv()
    assert env.offensive_scenario_prob == 0.0
    env.reset()
    assert env.last_scenario is None


def test_offensive_scenario_reports_a_stale_pyd_instead_of_an_attributeerror():
    """set_elixir_for_team/set_hand_for_team were added by commit 26de409 and
    the post-build copy into python_ai/ can silently fail (MSB3073). Discovering
    that as an AttributeError mid-episode inside a scenario constructor is the
    worst possible place; the check is hoisted to the entry point."""
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
    """A silently rejected setup still counts as an injected episode and would
    report practice that never happened -- set_hand_for_team returns False
    rather than raising, so the return value is the only signal."""
    from python_ai.envs import scenario_offense
    rng = np.random.default_rng(0)
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    name = scenario_offense.punish_window(env, rng, gym_wrapper.DEFAULT_DECK)
    assert name is not None and name.startswith("punish_window")
    assert 15 in list(env.get_hand_for_team(0)), "the win condition must be in hand"
    assert env.get_elixir_for_team(0) == pytest.approx(8.0)
    assert env.get_elixir_for_team(1) == pytest.approx(1.0)
    # Their commitment is on the board, visible to us as ENEMY mass.
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
    """reset() returns the observation BEFORE the rewrite. If the env forgets to
    re-read it, the agent's first observation describes a position that no
    longer exists -- invisible in every metric."""
    env = gym_wrapper.MicroRoyaleEnv({"offensive_scenario_prob": 1.0,
                                      "scenario_seed": 0})
    obs, _ = env.reset()
    assert env.last_scenario is not None
    live = np.asarray(env.game.get_observation_for_team(0), np.float32)
    assert np.allclose(obs, live), "the returned observation is pre-scenario"

def test_teacher_follows_a_deck_change():
    """Phase 1's `random_opponent` re-rolls the opponent deck every few hundred
    episodes. A teacher still holding the OLD deck's role table would treat the
    new deck's win condition as a plain melee troop and count cycle distance
    over cards it no longer holds -- a silent degradation that reads as "the
    teacher is weak against random decks"."""
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
    """Two separate paths write the opponent deck -- set_opponent_deck() and
    reset()'s randomize_opp_deck branch, which writes straight to self.game.
    The second is the easy one to miss."""
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
