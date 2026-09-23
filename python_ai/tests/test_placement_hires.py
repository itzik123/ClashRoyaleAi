"""place_hires, the high-resolution placement branch."""
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


# --- place_hires ---
# `cnn_trunk` pools twice, so the placement head reads a 9x5 map of a 34x18
# board and the card context enters as a spatially uniform vector.
# `place_hires` adds a parallel path from the trunk's pre-pool 16x34x18
# activation, conditioned on the same (hx, card) context and added to the
# coarse logits as a residual.
#
# Its final conv is zero-initialized, so at init it computes the identical
# function: an existing checkpoint loads bit-identically and keeps its trained
# placement head. The zero conv still receives gradient, so the branch starts
# learning immediately.

def _net(seed=0):
    torch.manual_seed(seed)
    return MicroRoyaleNet(num_ability_slots=0)


def _obs_batch(n=8, seed=0):
    """`n` genuinely different real observations."""
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    rng = np.random.default_rng(seed)
    rows = [np.asarray(env.get_observation_for_team(0), dtype=np.float32)]
    while len(rows) < n:
        # The action does not matter; distinct boards do, since one state can
        # be fit by a constant.
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


# --- safety: the change must not disturb what already works ---

def test_trunk_split_is_bit_identical_to_the_sequential():
    """Running cnn_trunk in two halves must equal running it whole, or every
    checkpoint's features shift.
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
    """At init the branch contributes exactly zero: its final conv's weight and
    bias are zeroed.
    """
    net = _net()
    obs = _obs_batch(4)
    logits, hx, embeds, spatial = _place(net, obs)

    # hx, the cycle skip, then the chosen card's embedding: the order
    # `placement_given_card` builds it in.
    ctx = net.place_ctx(torch.cat(
        (hx, net.cycle_features(obs, detached=True),
         embeds[torch.arange(4), 0]), dim=-1))
    coarse_map = net.place_up(spatial + ctx.view(-1, 32, 1, 1))
    coarse = coarse_map[:, 0, :net.placement_rows, :net.board_width].reshape(4, -1)
    coarse = coarse.masked_fill(
        ~net.placement_mask(obs, torch.zeros(4, dtype=torch.long)), float("-inf"))
    assert torch.equal(logits, coarse)


def test_old_checkpoint_keeps_the_placement_head():
    """An existing checkpoint warm-starts everything it carried; only the new
    branch is fresh.
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
    # ...and the branch is still an exact no-op.
    assert torch.equal(fresh.place_hires[-1].weight,
                       torch.zeros_like(fresh.place_hires[-1].weight))


def test_recomputed_hires_equals_the_passed_one():
    """The map passed in and the map rebuilt from `obs` must agree, or rollout and
    update compute different logits and the ratio breaks silently.
    """
    net = _net()
    for p in net.place_hires[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)       # wake the branch up
    obs = _obs_batch(4)
    passed, _, _, _ = _place(net, obs, hires=net.hires_features(obs))
    rebuilt, _, _, _ = _place(net, obs, hires=None)
    assert torch.equal(passed, rebuilt)


def test_refuses_to_guess_when_it_cannot_build_the_branch():
    """No silent fallback to the coarse-only head, which would be a different
    function than the rollout used.
    """
    net = _net()
    obs = _obs_batch(2)
    feats, embeds, spatial = net.extract_features(obs)
    hx = torch.zeros(2, 256)
    with pytest.raises(ValueError, match="hires"):
        net.placement_given_card(hx, embeds, torch.zeros(2, dtype=torch.long),
                                 None, spatial)


def test_forward_sequence_matches_the_single_step_path():
    """The batched update path matches the rollout path, coverage pass included.
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


# --- resolution ---

def _fit_exact_cells(net, obs, targets, steps=400, train_hires=True, lr=3e-3):
    """Fit placement logits to one exact cell per state; returns the argmax match.
    Only the placement pathway trains, so this measures what the head can
    express given fixed features.
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
    """One board per column, each with a single enemy troop in that column. Random
    states can differ only in a scalar the placement head does not read, which
    makes them unfittable at any resolution.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    rows = []
    for x in columns:
        env = CE(deck, deck, 3600)
        env.reset()
        # Archers by id, not by deck position: the fixture must not move when
        # the deck does.
        env.inject(1, float(x), row, 1)                 # Archers, team 1
        env.step(4, 0.0, 0.0, 1)        # no-op tick, so the unit is on the board
        rows.append(np.asarray(env.get_observation_for_team(0), dtype=np.float32))
    return torch.tensor(np.stack(rows))


def test_both_heads_can_resolve_a_single_column_at_small_scale():
    """Both heads resolve a single column: nearest-upsample followed by 3x3 convs
    mixes neighbouring pooled cells, so sub-block position is recoverable. The
    branch's value is measured at realistic scale in prove_hires.py; this pins
    only that it does not make the task harder.
    """
    columns = list(range(2, 16))
    obs = _enemy_column_states(columns)
    # Answer in the same column, row 12 for every state, so only the column has
    # to be read.
    targets = torch.tensor([12 * 18 + x for x in columns], dtype=torch.long)

    control = _fit_exact_cells(_net(seed=7), obs, targets, train_hires=False)
    treatment = _fit_exact_cells(_net(seed=7), obs, targets, train_hires=True)

    print(f"\n  same-column argmax match ({len(columns)} states) -- "
          f"coarse only: {control:.3f}   +hires: {treatment:.3f}")
    assert treatment >= 0.99, f"the branch should fit exactly, got {treatment}"
    assert treatment >= control, "the branch must not cost resolution"


def test_the_pooled_map_covers_the_whole_board():
    """`ceil_mode=True` on both MaxPools is load-bearing, and floor-pooling fails
    silently.

    What changes is the map's size, so that is what is pinned, derived from the
    board rather than hardcoded. Testing that row 33 still reaches the trunk
    does not work: the convolutions spread it into earlier rows before a
    floor-pool truncates.
    """
    import math

    import torch

    from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS
    from python_ai.models.net import MicroRoyaleNet

    net = MicroRoyaleNet(num_ability_slots=0)
    out = net.cnn_trunk(torch.zeros(1, N_CHANNELS, BOARD_H, BOARD_W))

    def pooled(size):        # two 2x2 pools, ceil at each
        return math.ceil(math.ceil(size / 2) / 2)

    def floored(size):
        return (size // 2) // 2

    assert out.shape[2] == pooled(BOARD_H), (
        f"pooled height {out.shape[2]} != ceil-pooled {pooled(BOARD_H)}; "
        f"floor-pooling would give {floored(BOARD_H)} and silently shrink the "
        "board the network can see")
    assert out.shape[3] == pooled(BOARD_W)
    # The two must differ, or this asserts nothing.
    assert pooled(BOARD_H) != floored(BOARD_H)


def test_every_board_row_reaches_the_trunk():
    """No row is wholly invisible to the CNN. Weaker than the ceil_mode guard (it
    passes with ceil_mode off); it catches a row disconnected by a stride, crop
    or layout change.
    """
    import torch

    from python_ai.engine_constants import BOARD_H, BOARD_W, N_CHANNELS
    from python_ai.models.net import MicroRoyaleNet

    net = MicroRoyaleNet(num_ability_slots=0)
    spatial = torch.zeros(1, N_CHANNELS, BOARD_H, BOARD_W)
    base = net.cnn_trunk(spatial)

    dead = []
    for row in range(BOARD_H):
        lit = spatial.clone()
        lit[0, 0, row, :] = 1.0
        if torch.equal(net.cnn_trunk(lit), base):
            dead.append(row)
    assert not dead, f"rows invisible to the CNN trunk: {dead}"
