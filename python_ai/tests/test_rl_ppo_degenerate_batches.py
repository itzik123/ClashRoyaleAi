"""What the update reports when a minibatch carries no trainable decision.

A chunk where nothing was affordable is common, and there the actor and entropy
terms are correctly zero. The report must be NaN, not 0.0: a finite 0.0
"measured entropy" passes the controller's non-finite guard and is read as a
total collapse, driving the coefficient up. The loss must not change: the
critic and the aux head still train on those states.
"""
import numpy as np
import pytest
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.buffer import CORE_FIELDS, RolloutBuffer
from python_ai.rl.config import EntropyConfig, PPOConfig
from python_ai.rl.entropy import EntropyController
from python_ai.rl.ppo import PPOUpdater

TINY = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                 num_minibatches=1, ppo_epochs=1)


def _rollout(decision_value):
    """A real rollout whose `decision` column is forced to `decision_value`."""
    import clash_royale_env
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN

    torch.manual_seed(5)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    envs = [clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
            for _ in range(TINY.num_envs)]
    for e in envs:
        e.reset()

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
                decision=torch.full((TINY.num_envs,), float(decision_value)),
                hx_in=hx, cx_in=cx, logprobs=lp, values=value.squeeze(-1),
                rewards=torch.randn(TINY.num_envs) * 0.1,
                masks=torch.ones(TINY.num_envs),
                valid=torch.ones(TINY.num_envs),
                aux_opp_played=torch.full((TINY.num_envs,), 15,
                                          dtype=torch.long),
                coverage_slot=torch.zeros(TINY.num_envs, dtype=torch.long))
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(0, 0, 0)
    return net, buf.stack()


def _run(net, batch):
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    updater = PPOUpdater(net, opt, TINY)
    adv = torch.randn_like(batch["rewards"]) * 0.1
    returns = adv + batch["values"]
    return updater.update(batch, adv, returns, 0.2,
                          ent_coef_card=0.01, ent_coef_placement=0.01,
                          coverage_coef=0.0)


def test_a_no_decision_update_reports_entropy_as_UNDEFINED_not_zero():
    """0.0 and "no data" are different diagnoses."""
    net, batch = _rollout(0.0)
    stats = _run(net, batch)
    assert np.isnan(stats.ent_card), (
        f"ent_card read {stats.ent_card} on a batch with no decision rows; "
        "a finite 0.0 here is indistinguishable from a collapsed policy")
    assert np.isnan(stats.ent_placement)


def test_the_controller_HOLDS_instead_of_chasing_the_collapse():
    """The consequence the report exists to prevent."""
    net, batch = _rollout(0.0)
    stats = _run(net, batch)

    ctrl = EntropyController(EntropyConfig())
    before_card = ctrl.coef_card
    before_place = ctrl.coef_placement
    frozen_before = ctrl.frozen_updates

    ctrl.update(stats.ent_card, stats.ent_placement, 0)

    assert ctrl.coef_card == before_card
    assert ctrl.coef_placement == before_place
    assert ctrl.frozen_updates == frozen_before + 2, (
        "the controller should have recorded a freeze for BOTH heads")


def test_a_zero_would_have_driven_the_coefficient_UP():
    """Negative control: a 0.0 really would raise the coefficient."""
    ctrl = EntropyController(EntropyConfig())
    before = ctrl.coef_card
    ctrl.update(0.0, 0.0, 0)
    assert ctrl.coef_card > before, (
        "a measured entropy of 0.0 should raise the coefficient -- if it does "
        "not, this test no longer demonstrates the hazard")


def test_the_LOSS_is_untouched_by_the_reporting_change():
    """A no-decision chunk still has real states: the critic and aux head keep
    learning, so the step is taken and not dropped as non-finite.
    """
    net, batch = _rollout(0.0)
    before = [p.detach().clone() for p in net.parameters()]
    stats = _run(net, batch)

    assert stats.nonfinite_skips == 0, (
        "the no-decision batch was dropped by the containment guard; the NaN "
        "escaped the diagnostics into the loss")
    assert np.isfinite(stats.critic_loss) and stats.critic_loss > 0.0
    assert np.isfinite(stats.aux_ce)

    after = list(net.parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after)), (
        "no parameter moved: the critic/aux gradient was lost with the actor's")


def test_a_normal_batch_still_reports_finite_entropy():
    """Control: a normal batch still reports finite entropy."""
    net, batch = _rollout(1.0)
    stats = _run(net, batch)
    assert np.isfinite(stats.ent_card)
    assert np.isfinite(stats.ent_placement)
    assert 0.0 <= stats.ent_card <= 1.0
    assert 0.0 <= stats.ent_placement <= 1.0


def test_a_MIXED_update_averages_only_the_informative_minibatches():
    """With mixed chunks, the reported mean comes from the informative ones only.
    """
    cfg = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                    num_minibatches=2, ppo_epochs=1)
    import clash_royale_env
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN

    torch.manual_seed(7)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    envs = [clash_royale_env.ClashRoyaleEnv(deck, deck, 3600) for _ in range(2)]
    for e in envs:
        e.reset()

    buf = RolloutBuffer(CORE_FIELDS)
    hx = torch.zeros(2, LSTM_HIDDEN)
    cx = torch.zeros(2, LSTM_HIDDEN)
    for t in range(cfg.update_timestep):
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
        # First half of the timesteps decide, second half are forced.
        buf.add(obs=obs, card_actions=card, placement_actions=cell,
                decision=torch.full((2,), 1.0 if t < 2 else 0.0),
                hx_in=hx, cx_in=cx, logprobs=lp, values=value.squeeze(-1),
                rewards=torch.zeros(2), masks=torch.ones(2),
                valid=torch.ones(2),
                aux_opp_played=torch.full((2,), -1, dtype=torch.long),
                coverage_slot=torch.zeros(2, dtype=torch.long))
        hx, cx = hx2, cx2
        for e in envs:
            e.step(0, 0, 0)

    batch = buf.stack()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    adv = torch.zeros_like(batch["rewards"])
    stats = PPOUpdater(net, opt, cfg).update(
        batch, adv, adv + batch["values"], 0.2,
        ent_coef_card=0.01, ent_coef_placement=0.01, coverage_coef=0.0)

    assert np.isfinite(stats.ent_card)
    assert stats.ent_card > 0.05, (
        f"ent_card={stats.ent_card} -- an empty minibatch appears to have been "
        "averaged in as a 0.0")
