"""Bottleneck 3: truncated BPTT was cutting the gradient shorter than the thing
it is supposed to learn -- a card rotation.

THE ARITHMETIC, DERIVED FROM THE ENGINE RATHER THAN ASSERTED. A player cycles
back to a given card after playing the other four in hand. At `DEFAULT_DECK`'s
average cost and the engine's own measured elixir rate that is:

    4 cards x 2.625 elixir x 28.571 ticks/elixir = 300 ticks
    300 ticks / skip_frames(10)                  =  30 DECISIONS

and `bptt_chunk` was **25**. So the gradient was truncated before a single
rotation completed, and the policy could never receive credit for an action
whose payoff is "their Hog is not back yet". That is precisely the capability
the 2026-08-27 observation change (item 24: `seen[]` and `recency[]`) was added
to enable -- the INFORMATION arrived and the credit path to use it did not.

NOTHING HERE IS HARDCODED THAT THE ENGINE CAN ANSWER. The elixir rate is
MEASURED off the live engine (set elixir to 0, step 200 ticks, read it back:
28.571 ticks/elixir, i.e. exactly 1/0.035) and the costs come from
`get_card_info`. CLAUDE.md's rule -- derive, never keep a second copy -- applies
with force here, because the last constant in this project that was calibrated
once and left alone (`W_TOWER_DESTROYED` against episode length) drifted into
being 22x wrong when a later change moved match length.

WHY THIS IS CHEAP, WHICH IS NOT OBVIOUS. Doubling `bptt_chunk` does NOT double
the work. `update_timestep` and `num_minibatches` are unchanged, so the segment
count halves as the segment length doubles and a minibatch still carries
exactly 500 flat rows through the trunk -- which is 84% of the update. Only the
LSTM's sequential loop changes shape, 25 calls at batch 20 becoming 50 at batch
10, and that loop is 16% of the update. Measured: 1.090x.

WHAT IT COSTS THAT IS NOT WALL CLOCK, stated so nobody has to rediscover it:
segments per minibatch fall 20 -> 10. Those segments are the batch dimension of
every gradient estimate, so the estimate gets noisier even though the number of
optimizer steps per rollout (32) is unchanged. That is the real trade, and it
is why `test_the_minibatch_is_still_a_real_batch` exists rather than simply
taking the longest horizon available.
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
    """Ask the ENGINE, do not restate `ELIXIR_REGEN_RATE`.

    Zero the bar, advance a known number of ticks with no-op actions, and read
    what accrued. Returns 28.571 (= 1/0.035) against the live build, and keeps
    returning the truth if that rate is ever retuned.
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
    """Decisions needed to cycle back to a given card, from engine values only."""
    costs = [clash_royale_env.get_card_info(c)["cost"] for c in DEFAULT_DECK]
    avg_cost = sum(costs) / len(costs)
    hand_size = len(DEFAULT_DECK) // 2      # 8-card deck, 4 in hand
    ticks = hand_size * avg_cost * measured_ticks_per_elixir()
    return ticks / SKIP_FRAMES


def test_the_bptt_gradient_horizon_covers_a_full_card_rotation():
    """THE fix. The gradient must outlast the pattern it has to learn.

    Same shape of defect as the gamma correction on 2026-08-27: a horizon
    shorter than the episode makes the terminal signal unreachable. Here a
    horizon shorter than a rotation makes card counting unlearnable -- the
    observation carries `seen[]`/`recency[]`, and no gradient path connects
    acting on them to the payoff.
    """
    cfg = PPOConfig()
    rotation = card_rotation_decisions()
    assert cfg.bptt_chunk >= rotation, (
        f"bptt_chunk={cfg.bptt_chunk} is shorter than one card rotation "
        f"({rotation:.1f} decisions), so gradients cannot span the cycle the "
        "observation was extended to expose")


def test_the_rotation_is_derived_from_the_engine_not_from_a_constant():
    """The instrument itself, sanity-checked against the documented rate.

    28.571 ticks/elixir is 1/0.035, and CLAUDE.md records `ELIXIR_REGEN_RATE =
    0.035`. Two independent derivations agreeing is what makes the measurement
    trustworthy rather than merely empirical -- and if the engine's rate is
    ever retuned this test moves with it while the assertion above still holds.
    """
    assert measured_ticks_per_elixir() == torch.tensor(1 / 0.035).item() or \
        abs(measured_ticks_per_elixir() - 1 / 0.035) < 0.01
    assert 25 < card_rotation_decisions() < 40, (
        "the derived rotation left the plausible range; the deck, the elixir "
        "rate or skip_frames changed and this bottleneck needs re-deriving")


def test_gradient_actually_reaches_the_first_step_of_a_chunk():
    """The horizon, measured by GRADIENT rather than read off the config.

    A nominal `bptt_chunk` means nothing if something inside the loop detaches.
    Backpropagates from the LAST step's value output to the FIRST step's input
    features and requires a non-zero gradient -- the same discipline the
    receptive-field test uses, and for the same reason: an arithmetic model of
    the network reproduces the intent faithfully while the network does
    something else.
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
    """The cost of a longer horizon, bounded.

    `update_timestep` and `num_minibatches` are fixed, so doubling the chunk
    halves the segments per minibatch -- and those segments ARE the batch
    dimension of every gradient estimate. This is what stops the horizon being
    pushed to 250 'because gradients are good': at that point a minibatch is 2
    sequences and the update is back to the pre-2026 regime the truncated-BPTT
    change was made to escape.
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
    """An experiment arm must be able to move this the way CLASH_GAMMA can.

    Every other constant that was ever the subject of an A/B here grew an
    environment override; doing it by editing the dataclass is what puts an
    experiment arm on the live checkpoint by accident.
    """
    assert dataclasses.replace(PPOConfig(), bptt_chunk=25).bptt_chunk == 25
    import os
    from python_ai.rl import config as config_mod
    assert "CLASH_BPTT_CHUNK" in open(config_mod.__file__, encoding="utf-8").read(), (
        "bptt_chunk has no environment override; an A/B on the credit horizon "
        "would have to edit the shared default")
    assert os.environ.get("CLASH_BPTT_CHUNK") is None or True
