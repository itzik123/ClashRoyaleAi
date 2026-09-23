"""The deck-coverage penalty: a floor under P(play card | card in hand).

The entropy controller measures per-decision slot entropy, which is blind to
the marginal over the deck: a policy that mostly no-ops and spreads its plays
over five cards meets its target indefinitely. `DECK_COVERAGE_FLOOR` sits in
the gap between the live cards and the dead ones, so the term is exactly zero
for a healthy deck: a hinge, not a target, refusing to let a card reach zero
without pushing toward uniform use.
"""
import copy

import torch

from python_ai.rl.deck_coverage import deck_coverage_penalty


def _logits_with(probs):
    """Card logits whose softmax is `probs` (one row, arms = len(probs))."""
    return torch.log(torch.tensor([probs], dtype=torch.float32))


def test_a_healthy_deck_costs_exactly_zero():
    """Free when every card clears the floor."""
    logits = _logits_with([0.10, 0.10, 0.10, 0.10, 0.60])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, _, _ = deck_coverage_penalty(logits, hand, decision, floor=0.02)
    assert pen.item() == 0.0


def test_a_starved_card_costs_its_log_shortfall():
    """The shortfall is measured in log space."""
    import math
    logits = _logits_with([0.005, 0.30, 0.30, 0.30, 0.095])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, min_p, n = deck_coverage_penalty(logits, hand, decision, floor=0.02)

    assert n == 4
    assert abs(min_p - 0.005) < 1e-6
    # Mean over the 4 cards of relu(log(floor) - log(p)): only card 10 is
    # short.
    assert abs(pen.item() - math.log(0.02 / 0.005) / 4) < 1e-5


def test_the_push_does_not_weaken_as_the_card_gets_deader():
    """The push must not weaken as the card gets deader.

    A hinge on the probability itself carries a softmax p*(1-p) factor and
    vanishes as p -> 0, weakest exactly where the card is deadest. A log
    hinge's gradient is (1 - p), ~1 for any dead card. This pins the shape.
    """
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    grads = []
    for p_dead in (0.0011, 0.005, 0.015):
        others = (1.0 - p_dead) / 3.0
        raw = torch.log(torch.tensor(
            [[p_dead, others, others, others, 1e-9]], dtype=torch.float32)
        ).requires_grad_(True)
        pen, _, _ = deck_coverage_penalty(raw, hand, decision, floor=0.02)
        pen.backward()
        grads.append(-raw.grad[0, 0].item())

    assert all(g > 0 for g in grads), "every starved card must be pushed up"
    # The deadest card is pushed at least as hard as the nearly recovered one.
    assert grads[0] / grads[-1] > 0.9, (
        f"push weakens as the card dies: {grads[0]:.5f} at p=0.0011 vs "
        f"{grads[-1]:.5f} at p=0.015")


def test_it_pushes_the_starved_card_UP_and_leaves_healthy_cards_alone():
    """Gradient descent raises the starved card and not the healthy ones."""
    raw = torch.tensor([[-6.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, _, _ = deck_coverage_penalty(raw, hand, decision, floor=0.02)
    pen.backward()

    # Descending the loss means stepping along -grad.
    starved_step = -raw.grad[0, 0].item()
    healthy_step = -raw.grad[0, 1].item()
    assert starved_step > 0.0, "the starved card's logit must be pushed up"
    assert healthy_step <= 0.0, "a healthy card must not also be pushed up"


def test_the_no_op_arm_is_never_floored():
    """The no-op is not a deck card; flooring it would force the agent to play.
    """
    logits = _logits_with([0.33, 0.33, 0.33, 0.005, 0.005])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    _, _, n = deck_coverage_penalty(logits, hand, decision, floor=0.02)
    assert n == 4, "exactly the four hand slots, never the no-op arm"


def test_padded_rows_are_excluded():
    """Rows outside the decision mask must not dilute or create a shortfall."""
    logits = torch.log(torch.tensor([
        [0.005, 0.30, 0.30, 0.30, 0.095],   # real, card 10 starved
        [0.005, 0.30, 0.30, 0.30, 0.095],   # padding, must be ignored
    ], dtype=torch.float32))
    hand = torch.tensor([[10, 20, 30, 40], [10, 20, 30, 40]])

    both = deck_coverage_penalty(logits, hand, torch.ones(2), floor=0.02)[0]
    one = deck_coverage_penalty(logits, hand, torch.tensor([1.0, 0.0]), floor=0.02)[0]
    torch.testing.assert_close(both, one)


def test_empty_hand_slots_are_skipped_without_nan():
    """-1 marks an empty slot (net.hand_card_ids' convention)."""
    logits = _logits_with([0.25, 0.25, 0.25, 0.20, 0.05])
    hand = torch.tensor([[10, 20, -1, -1]])
    decision = torch.ones(1)

    pen, _, n = deck_coverage_penalty(logits, hand, decision, floor=0.02)
    assert n == 2
    assert torch.isfinite(pen).all()


def test_no_decision_rows_yields_a_finite_zero():
    """A minibatch of pure padding contributes a finite zero, not NaN."""
    logits = _logits_with([0.20, 0.20, 0.20, 0.20, 0.20])
    hand = torch.tensor([[10, 20, 30, 40]])

    pen, min_p, n = deck_coverage_penalty(logits, hand, torch.zeros(1), floor=0.02)
    assert n == 0
    assert pen.item() == 0.0
    assert torch.isfinite(pen).all()
    assert min_p == 0.0


# --- integration: the term reaches the PPO update and is reported ---
import numpy as np
import pytest
import torch.optim as optim

import clash_royale_env
from python_ai.envs.gym_wrapper import DEFAULT_DECK
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN
from python_ai.rl.buffer import CORE_FIELDS, RolloutBuffer
from python_ai.rl.config import PPOConfig
from python_ai.rl.ppo import PPOUpdater

TINY = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                 num_minibatches=1, ppo_epochs=1)


@pytest.fixture(scope="module")
def rollout():
    """A tiny but real rollout, built as in test_rl_ppo.py."""
    torch.manual_seed(11)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    envs = [clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
            for _ in range(TINY.num_envs)]
    for e in envs:
        e.reset()
        # A real threat on our half: the term is threat-gated, so on an empty
        # board it measures nothing and reports NaN. An enemy Hog clears
        # tactics.DECK_COVERAGE_THREAT_HP comfortably.
        e.inject_enemy(15, 14.5, 14.0)
        e.step(clash_royale_env.ClashRoyaleEnv.HAND_SIZE, 0.0, 0.0, 1)
    buf = RolloutBuffer(CORE_FIELDS)
    hx = torch.zeros(TINY.num_envs, LSTM_HIDDEN)
    cx = torch.zeros(TINY.num_envs, LSTM_HIDDEN)
    for _ in range(TINY.update_timestep):
        obs = torch.tensor(np.asarray(
            [e.get_observation_for_team(0) for e in envs], dtype=np.float32))
        mask = net.affordability_mask(obs)
        with torch.no_grad():
            feats, embeds, spatial = net.extract_features(obs)
            logits, _, _, value, (hx2, cx2) = net.step_lstm_and_card(
                feats, (hx, cx), mask)
            card = torch.distributions.Categorical(logits=logits).sample()
            pl = net.placement_given_card(hx2, embeds, card, obs, spatial)
            cell = torch.distributions.Categorical(logits=pl).sample()
            lp = (torch.distributions.Categorical(logits=logits).log_prob(card)
                  + torch.distributions.Categorical(logits=pl).log_prob(cell))
        buf.add(obs=obs, card_actions=card, placement_actions=cell,
                decision=(mask.sum(dim=1) > 1).float(),
                hx_in=hx, cx_in=cx, logprobs=lp, values=value.squeeze(-1),
                rewards=torch.zeros(TINY.num_envs),
                masks=torch.ones(TINY.num_envs),
                valid=torch.ones(TINY.num_envs),
                aux_opp_played=torch.full((TINY.num_envs,), -1, dtype=torch.long),
                coverage_slot=torch.zeros(TINY.num_envs, dtype=torch.long))
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(int(card[i]), 0.0, 0.0, 10)
    return net, buf.stack()


def _run(net, batch, **kwargs):
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    updater = PPOUpdater(net, optimizer, TINY)
    T, N = batch["rewards"].shape
    defaults = dict(ent_coef_card=0.05, ent_coef_placement=0.06,
                    coverage_coef=0.02)
    defaults.update(kwargs)
    return updater.update(batch, torch.randn(T, N), torch.zeros(T, N),
                          vf_clip_range=0.2, **defaults)


def test_the_update_reports_deck_coverage(rollout):
    """Without a reported number a dead card is invisible. Needs a threatened
    board (see the fixture).
    """
    net, batch = rollout
    stats = _run(copy.deepcopy(net), batch)

    assert np.isfinite(stats.deck_coverage)
    assert stats.deck_coverage >= 0.0
    assert 0.0 <= stats.deck_min_card_prob <= 1.0


def test_the_penalty_never_reaches_the_ppo_ratio(rollout):
    """A regulariser must not touch `new_logprobs`, or the ratio no longer
    compares the policies that produced the data.
    """
    net, batch = rollout
    off = _run(copy.deepcopy(net), batch, deck_coverage_coef=0.0)
    on = _run(copy.deepcopy(net), batch, deck_coverage_coef=50.0)

    assert on.clip_frac == pytest.approx(off.clip_frac, abs=1e-9)
