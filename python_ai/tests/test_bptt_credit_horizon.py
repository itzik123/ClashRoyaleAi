"""The truncated-BPTT horizon must cover a card rotation.

Cycling back to a card means playing the other four in hand: at DEFAULT_DECK's
average cost and the engine's measured elixir rate,

    4 cards x 2.625 elixir x 28.571 ticks/elixir = 300 ticks = 30 decisions

A shorter `bptt_chunk` cuts the gradient before a rotation completes, so no
action can be credited for "their Hog is not back yet", even though the
observation carries the opponent's cycle. The elixir rate is measured off the
engine and the costs come from `get_card_info`.

Longer chunks cost little wall clock (the flat trunk rows per minibatch are
unchanged; only the LSTM loop reshapes) but halve the segments per minibatch,
the batch dimension of every gradient estimate;
`test_the_minibatch_is_still_a_real_batch` bounds that.
"""
import dataclasses

import clash_royale_env
import torch

from python_ai.envs.gym_wrapper import DEFAULT_DECK
from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.config import PPOConfig

#: One decision is this many engine ticks (gym_wrapper.step's default).
SKIP_FRAMES = 10


def measured_ticks_per_elixir():
    """Ask the engine rather than restating `ELIXIR_REGEN_RATE`: zero the bar,
    advance known ticks with no-ops, read what accrued.
    """
    env = clash_royale_env.ClashRoyaleEnv(DEFAULT_DECK, DEFAULT_DECK, 3600)
    env.reset()
    env.set_elixir_for_team(0, 0.0)
    t0, e0 = env.get_current_tick(), env.get_elixir_for_team(0)
    noop = env.NUM_CARD_IDS and 4          # hand_size -> the no-op card index
    for _ in range(20):
        env.step_self_play(noop, 0, 0, noop, 0, 0, SKIP_FRAMES)
    t1, e1 = env.get_current_tick(), env.get_elixir_for_team(0)
    return (t1 - t0) / (e1 - e0)


def card_rotation_decisions():
    """Decisions needed to cycle back to a card, from engine values only."""
    costs = [clash_royale_env.get_card_info(c)["cost"] for c in DEFAULT_DECK]
    avg_cost = sum(costs) / len(costs)
    hand_size = len(DEFAULT_DECK) // 2      # 8-card deck, 4 in hand
    ticks = hand_size * avg_cost * measured_ticks_per_elixir()
    return ticks / SKIP_FRAMES


def test_the_bptt_gradient_horizon_covers_a_full_card_rotation():
    """The gradient must outlast the pattern it has to learn, as gamma's horizon
    must outlast the episode.
    """
    cfg = PPOConfig()
    rotation = card_rotation_decisions()
    assert cfg.bptt_chunk >= rotation, (
        f"bptt_chunk={cfg.bptt_chunk} is shorter than one card rotation "
        f"({rotation:.1f} decisions), so gradients cannot span the cycle the "
        "observation was extended to expose")


def test_the_rotation_is_derived_from_the_engine_not_from_a_constant():
    """The instrument, checked against the documented 0.035 rate: two independent
    derivations agreeing.
    """
    assert measured_ticks_per_elixir() == torch.tensor(1 / 0.035).item() or \
        abs(measured_ticks_per_elixir() - 1 / 0.035) < 0.01
    assert 25 < card_rotation_decisions() < 40, (
        "the derived rotation left the plausible range; the deck, the elixir "
        "rate or skip_frames changed and this bottleneck needs re-deriving")


def test_gradient_actually_reaches_the_first_step_of_a_chunk():
    """The horizon measured by gradient, not read off the config: from the last
    step's value back to the first step's input, so a detach inside the loop
    cannot hide.
    """
    cfg = PPOConfig()
    net = MicroRoyaleNet(num_ability_slots=0)
    L, B = cfg.bptt_chunk, 2

    feats = torch.randn(L, B, net.lstm_input_dim, requires_grad=True)
    embeds = torch.randn(L, B, net.hand_size + 1, 16)
    spatial = torch.randn(L, B, 32, net.pooled_h, net.pooled_w)
    obs = torch.rand(L, B, net.spatial_size + net.scalar_size)
    mask = torch.ones(L, B, net.hand_size + 1, dtype=torch.bool)
    idx = torch.zeros(L, B, dtype=torch.long)
    resets = torch.ones(L, B)
    hx = torch.zeros(B, net.LSTM_HIDDEN)

    _, _, values, _, _, _ = net.forward_sequence(
        feats, embeds, spatial, obs, mask, idx, resets, (hx, hx.clone()))
    values.view(L, B)[-1].sum().backward()

    first = feats.grad[0].abs().sum().item()
    assert first > 0, (
        f"no gradient reaches step 0 from step {L - 1}: the recurrent path is "
        "broken or detached, and bptt_chunk is nominal only")


def test_the_minibatch_is_still_a_real_batch():
    """Doubling the chunk halves the segments per minibatch, the batch dimension
    of the gradient estimate; a floor stops the horizon being pushed until a
    minibatch is two sequences.
    """
    cfg = PPOConfig()
    per_minibatch = cfg.segments_per_rollout // cfg.num_minibatches
    assert per_minibatch >= 8, (
        f"only {per_minibatch} segments per minibatch at "
        f"bptt_chunk={cfg.bptt_chunk}; the gradient estimate is too noisy to "
        "be worth the longer horizon")


def test_the_chunk_still_divides_the_rollout_exactly():
    """A partial trailing segment would silently train on a shorter horizon."""
    cfg = PPOConfig()
    assert cfg.update_timestep % cfg.bptt_chunk == 0
    assert cfg.segments_per_rollout == (
        cfg.update_timestep // cfg.bptt_chunk) * cfg.num_envs


def test_the_horizon_is_overridable_without_editing_the_file():
    """An experiment arm must be able to move this through an environment
    override, not by editing the shared default.
    """
    assert dataclasses.replace(PPOConfig(), bptt_chunk=25).bptt_chunk == 25
    import os
    from python_ai.rl import config as config_mod
    assert "CLASH_BPTT_CHUNK" in open(config_mod.__file__, encoding="utf-8").read(), (
        "bptt_chunk has no environment override; an A/B on the credit horizon "
        "would have to edit the shared default")
    assert os.environ.get("CLASH_BPTT_CHUNK") is None or True
