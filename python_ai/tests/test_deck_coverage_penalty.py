"""The deck-coverage penalty: a floor under P(play card | card in hand).

WHY THE ENTROPY CONTROLLER CANNOT DO THIS JOB
---------------------------------------------
`EntropyController` measures entropy over the four hand SLOTS plus no-op, per
decision, normalised by the reachable arms. Across the 2026-08-28 phase-1 run it
held `target_card = 0.35` essentially perfectly -- 0.339 to 0.350 over 24,000
episodes, never approaching its 0.5 ceiling -- while the policy used five of its
eight cards. A policy that no-ops on ~81% of steps and spreads the remainder
over five cards sits at that target indefinitely, because per-decision slot
entropy is blind to the marginal distribution over the DECK.

Measured on the final checkpoint (ep 32,484), P(play card | card in hand):

    Skeletons   0.3429      Hog Rider   0.0704      The Log    0.0091
    Ice Spirit  0.3247      Musketeer   0.0662      Cannon     0.0053
    Ice Golem   0.1868                              Fireball   0.0011

The five live cards and the three dead ones are separated by a factor of ~7,
with nothing in between. `DECK_COVERAGE_FLOOR = 0.02` sits in that gap: it is
3.3x below the weakest live card and 2.2x above the strongest dead one, so the
term is EXACTLY ZERO for a healthy deck and only ever acts on a frozen head.
That is why the floor is a hinge and not a target -- it must not push a policy
toward uniform card use, only refuse to let a card reach zero.
"""
import copy

import torch

from python_ai.rl.deck_coverage import deck_coverage_penalty


def _logits_with(probs):
    """Card logits whose softmax is `probs` (one row, arms = len(probs))."""
    return torch.log(torch.tensor([probs], dtype=torch.float32))


def test_a_healthy_deck_costs_exactly_zero():
    """The term must be free when every card clears the floor."""
    logits = _logits_with([0.10, 0.10, 0.10, 0.10, 0.60])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, _, _ = deck_coverage_penalty(logits, hand, decision, floor=0.02)
    assert pen.item() == 0.0


def test_a_starved_card_costs_its_log_shortfall():
    """The shortfall is measured in LOG space -- see the module docstring."""
    import math
    logits = _logits_with([0.005, 0.30, 0.30, 0.30, 0.095])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, min_p, n = deck_coverage_penalty(logits, hand, decision, floor=0.02)

    assert n == 4
    assert abs(min_p - 0.005) < 1e-6
    # mean over the 4 cards of relu(log(floor) - log(p)): only card 10 is short.
    assert abs(pen.item() - math.log(0.02 / 0.005) / 4) < 1e-5


def test_the_push_does_not_weaken_as_the_card_gets_deader():
    """THE REGRESSION THIS FORM EXISTS FOR.

    The first implementation hinged on the probability directly, so the
    gradient carried a softmax `p*(1-p)` factor and went to ZERO as p did. At
    Fireball's measured p = 0.0011 that gradient was 0.00027 -- 17x weaker than
    at p = 0.019, i.e. weakest exactly where the card was deadest.

    Measured live on the ep-32,484 checkpoint over three paired 18-minute arms,
    that produced a clean null: MinCardProb moved -0.0007 at coef 0, +0.0018 at
    coef 2 and +0.0004 at coef 8. Non-monotone in the coefficient and inside the
    run-to-run noise -- 4x the coefficient did not help, because no coefficient
    can fix a term whose push shrinks as the problem worsens.

    A log hinge's gradient is (1 - p), which is ~1 for any dead card. This test
    pins the SHAPE: the push must not fall off as p -> 0.
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
    # The deadest card must be pushed AT LEAST as hard as the nearly-recovered
    # one. Under the old linear hinge this ratio was 0.059 (17x weaker).
    assert grads[0] / grads[-1] > 0.9, (
        f"push weakens as the card dies: {grads[0]:.5f} at p=0.0011 vs "
        f"{grads[-1]:.5f} at p=0.015")


def test_it_pushes_the_starved_card_UP_and_leaves_healthy_cards_alone():
    """The property that matters: gradient descent on this raises the floor.

    A penalty that merely had the right value could still be flat, or could
    push every card toward uniform. This pins the direction per card.
    """
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
    """No-op is not a deck card. Flooring it would force the agent to play.

    The hand tensor has one entry per card slot; the final logit column is the
    no-op arm and must be excluded from the coverage set entirely.
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
    """-1 marks an empty slot (net.hand_card_ids' own convention)."""
    logits = _logits_with([0.25, 0.25, 0.25, 0.20, 0.05])
    hand = torch.tensor([[10, 20, -1, -1]])
    decision = torch.ones(1)

    pen, _, n = deck_coverage_penalty(logits, hand, decision, floor=0.02)
    assert n == 2
    assert torch.isfinite(pen).all()


def test_no_decision_rows_yields_a_finite_zero():
    """A minibatch of pure padding must contribute nothing, not NaN.

    `rl/ppo.py` already carries a NaN-from-fully-masked-rows regression; this
    term must not reintroduce one on the degenerate batches
    test_rl_ppo_degenerate_batches.py covers.
    """
    logits = _logits_with([0.20, 0.20, 0.20, 0.20, 0.20])
    hand = torch.tensor([[10, 20, 30, 40]])

    pen, min_p, n = deck_coverage_penalty(logits, hand, torch.zeros(1), floor=0.02)
    assert n == 0
    assert pen.item() == 0.0
    assert torch.isfinite(pen).all()
    assert min_p == 0.0


# --------------------------------------------------------------------------
# Integration: the term must reach the PPO update and be reported.
# --------------------------------------------------------------------------
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
    """A tiny but REAL rollout -- same construction as test_rl_ppo.py's."""
    torch.manual_seed(11)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    envs = [clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
            for _ in range(TINY.num_envs)]
    for e in envs:
        e.reset()
        # A REAL THREAT on our half. The term is threat-gated, so on the empty
        # board this fixture used to build it correctly measures nothing and
        # reports NaN -- which is the gate working, not a failure. An enemy Hog
        # (3151 HP) clears tactics.DECK_COVERAGE_THREAT_HP = 400 comfortably.
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
    """Without a reported number, a dead card is invisible again.

    Requires a THREATENED board: the term is gated, so on a quiet one it
    measures nothing by design and reports NaN. See the fixture.
    """
    net, batch = rollout
    stats = _run(copy.deepcopy(net), batch)

    assert np.isfinite(stats.deck_coverage)
    assert stats.deck_coverage >= 0.0
    assert 0.0 <= stats.deck_min_card_prob <= 1.0


def test_the_penalty_never_reaches_the_ppo_ratio(rollout):
    """Same rule the advisor coverage term lives under.

    The term is a REGULARIZER. If it touched `new_logprobs` the importance
    ratio would no longer be the ratio of the policies that produced the data,
    and the update would silently stop being PPO.
    """
    net, batch = rollout
    off = _run(copy.deepcopy(net), batch, deck_coverage_coef=0.0)
    on = _run(copy.deepcopy(net), batch, deck_coverage_coef=50.0)

    assert on.clip_frac == pytest.approx(off.clip_frac, abs=1e-9)
