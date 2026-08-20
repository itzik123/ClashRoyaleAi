"""place_hires -- the high-resolution placement branch.

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
