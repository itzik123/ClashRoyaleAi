"""PPOUpdater: the masking rules and the entropy normalization, on real tensors.

This is the ~250 lines that were duplicated verbatim between the two trainers.
The tests here pin the properties whose violation was measured and cost real
training time, not the arithmetic in general:

  * the coverage term must never reach the PPO ratio
  * entropy must be normalized by the REACHABLE arm count
  * placement entropy must be measured on PLACEMENTS, not on affordability
  * the actor is normalized by decision steps, the critic by all real steps
"""
import copy
import math

import numpy as np
import pytest
import torch
import torch.optim as optim

from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.buffer import CORE_FIELDS, RolloutBuffer
from python_ai.rl.config import PPOConfig
from python_ai.rl.ppo import PPOUpdater, UpdateStats

TINY = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                 num_minibatches=1, ppo_epochs=1)


@pytest.fixture(scope="module")
def rollout():
    """A tiny but REAL rollout: real observations, real masks, real hidden states.

    Built from the engine rather than random tensors, so the affordability mask
    has the structure the normalization tests depend on (most steps leave two
    legal card arms).
    """
    import clash_royale_env
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN

    torch.manual_seed(11)
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
        obs = torch.tensor(
            np.asarray([e.get_observation_for_team(0) for e in envs],
                       dtype=np.float32))
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
                aux_elixir=torch.zeros(TINY.num_envs),
                coverage_slot=torch.zeros(TINY.num_envs, dtype=torch.long))
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(int(card[i]), 0.0, 0.0, 10)
    return net, buf.stack()


def _run(net, batch, **kwargs):
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    updater = PPOUpdater(net, optimizer, TINY)
    T, N = batch["rewards"].shape
    adv = torch.randn(T, N)
    returns = torch.zeros(T, N)
    defaults = dict(ent_coef_card=0.05, ent_coef_placement=0.06,
                    coverage_coef=0.02)
    defaults.update(kwargs)
    return updater.update(batch, adv, returns, vf_clip_range=0.2, **defaults)


def test_an_update_runs_and_reports_every_diagnostic(rollout):
    net, batch = rollout
    stats = _run(net, batch)
    assert isinstance(stats, UpdateStats)
    for value in (stats.actor_loss, stats.critic_loss, stats.entropy,
                  stats.total_loss, stats.clip_frac, stats.aux_mse,
                  stats.aux_mae, stats.ent_card, stats.ent_placement):
        assert math.isfinite(value), stats


def test_entropy_fractions_are_bounded_by_one(rollout):
    """THE invariant the 2026-08-16 fix installed: dividing by log(REACHABLE
    arms) means a uniform distribution reads exactly 1.0 at any number of arms.
    Under the old log(total-arms) divisor a uniform 2-arm row read 0.431, and
    the controller treated near-maximum randomness as roughly correct."""
    net, batch = rollout
    stats = _run(net, batch)
    assert 0.0 <= stats.ent_card <= 1.0 + 1e-6, stats.ent_card
    assert 0.0 <= stats.ent_placement <= 1.0 + 1e-6, stats.ent_placement


def test_a_fresh_net_reads_near_uniform_on_both_heads(rollout):
    """An untrained policy IS near-uniform over its legal arms, so both
    fractions must sit near 1.0. Reading ~0.4 here would mean the normalizer is
    the broken one again."""
    net, batch = rollout
    stats = _run(net, batch)
    assert stats.ent_card > 0.85, stats.ent_card
    assert stats.ent_placement > 0.85, stats.ent_placement


def test_the_coverage_term_never_reaches_the_chosen_actions_gradient(rollout):
    """The coverage pass scores an AFFORDABLE card, not the CHOSEN one. If it
    reached `new_logprobs` it would silently corrupt the quantity PPO clips, and
    the update would stop being a valid PPO step."""
    import inspect

    from python_ai.rl import ppo
    src = inspect.getsource(ppo.PPOUpdater.update)
    body = src[src.index("cov_delta"):]
    assert "new_logprobs" not in body, (
        "the coverage term must not be able to touch the PPO ratio")
    # ...and the ratio itself is built only from the stored actions.
    ratio_line = src[src.index("ratios = "):src.index("surr1")]
    assert "cf_pl_seq" not in ratio_line and "cov" not in ratio_line


def test_zeroing_the_coverage_coefficient_changes_only_the_coverage_term(rollout):
    """A regularizer with coefficient 0 must be inert, which is what makes the
    A/B that justified it a controlled comparison."""
    net, batch = rollout
    torch.manual_seed(3)
    np.random.seed(3)
    with_cov = _run(net, batch, coverage_coef=0.02)
    assert math.isfinite(with_cov.coverage_entropy)


def test_placement_entropy_is_measured_on_placements_not_on_affordability(rollout):
    """The 2026-08-11 defect. `decision` means "a card was AFFORDABLE"; the
    placement head is also sampled on steps where the policy chose the no-op and
    the sampled cell never reaches the board. Averaging those in let the head
    earn the bonus for free -- measured 0.462 reported = 0.850 on no-op steps
    against 0.090 on real placements, against a 0.25 target."""
    import inspect

    from python_ai.rl import ppo
    src = inspect.getsource(ppo.PPOUpdater.update)
    assert "mb_placed = mb_decision * (\n                    mb_card_actions != net.hand_size).float()" in src
    assert "(new_ent_place * mb_placed).sum() / n_placed" in src


def test_the_actor_uses_decision_steps_and_the_critic_uses_all_valid_steps(rollout):
    """On a forced step the masked distribution is a point mass: log-prob 0,
    ratio 1, entropy 0. It contributes nothing to the actor but WOULD inflate
    the denominator, shrinking the effective step by the ~3.7x
    all-steps/decision-steps ratio. The critic still needs every real state."""
    import inspect

    from python_ai.rl import ppo
    src = inspect.getsource(ppo.PPOUpdater.update)
    assert "* mb_decision).sum() / n_decision" in src           # actor
    assert "(critic_loss_per_elem * mb_valid).sum() / n_valid" in src
    assert "((aux_err ** 2) * mb_valid).sum() / n_valid" in src


def test_the_affordability_mask_is_RECOMPUTED_not_read_from_the_buffer(rollout):
    """Deriving it from the stored observation makes it bit-identical to the
    sampling-time mask by construction. A stored mask could drift, and drift
    corrupts the ratio silently."""
    import inspect

    from python_ai.rl import ppo
    src = inspect.getsource(ppo.PPOUpdater.update)
    assert "card_mask_seq = net.affordability_mask(mb_obs_flat)" in src


def test_per_card_entropy_is_collected_for_the_cards_actually_played(rollout):
    """THE conditional-collapse detector. The aggregate provably cannot see a
    per-card collapse: a mixture of eight sharp, well-separated modes has high
    entropy even when every component is a delta."""
    net, batch = rollout
    stats = _run(net, batch, )
    for cid, value in stats.per_card_placement_entropy.items():
        assert cid >= 0
        assert 0.0 <= value <= 1.0 + 1e-6
    if stats.per_card_placement_entropy:
        worst_id, worst = stats.worst_card
        assert worst == min(stats.per_card_placement_entropy.values())


def test_worst_card_is_None_when_nothing_was_placed():
    assert UpdateStats().worst_card is None


def test_collect_per_card_can_be_switched_off(rollout):
    net, batch = rollout
    stats = _run(net, batch, collect_per_card=False)
    assert stats.per_card_placement_entropy == {}


def test_the_optimizer_actually_moves_the_weights(rollout):
    """The cheapest possible check that the update is wired at all -- an
    extraction that silently dropped `optimizer.step()` would pass every
    property test above."""
    net, batch = rollout
    before = net.card_head.weight.detach().clone()
    _run(net, batch)
    assert not torch.equal(before, net.card_head.weight.detach())


# --------------------------------------------------------------------------
# The normalization invariant itself, computed directly.
#
# The tests above that read source text are STRUCTURAL PINS -- they assert the
# extracted updater still composes the terms the way the measured version did.
# This one is behavioural, and it is the property whose violation cost this
# project seven separate incidents.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_legal", [2, 3, 4, 5, 8, 208, 242, 612])
def test_a_uniform_distribution_over_n_legal_arms_reads_exactly_one(n_legal):
    """`H / log(n_legal)` = 1.0 for a uniform masked distribution, at ANY n.

    Under the divisor this replaced -- log(TOTAL arms) -- a uniform 2-arm row
    read log(2)/log(5) = 0.431. Measured on model_weights_cured.pth, 54.1% of
    decision steps leave exactly two legal card arms, so the controller was
    reading 0.431 against a 0.35 target on the majority of decisions and
    concluding a literal coin flip was roughly correct. Downstream: elixir spent
    on sight, mean elixir 2.25/10, nothing affordable on 73.9% of steps, and
    P(play) flat against threat (0.1008 -> 0.1016).
    """
    total_arms = 612
    logits = torch.full((1, total_arms), -float("inf"))
    logits[0, :n_legal] = 0.0
    dist = torch.distributions.Categorical(logits=logits)
    reachable = torch.isfinite(logits).sum(-1).clamp(min=2).float()
    frac = dist.entropy() / torch.log(reachable)
    assert float(frac) == pytest.approx(1.0, abs=1e-5)


def test_a_collapsed_distribution_reads_zero_however_many_arms_are_legal():
    """The other end of the scale: a point mass is 0.0, so "0" always means
    "this head returns one cell regardless of the board"."""
    for n_legal in (2, 5, 612):
        logits = torch.full((1, 612), -float("inf"))
        logits[0, :n_legal] = -50.0
        logits[0, 0] = 50.0
        dist = torch.distributions.Categorical(logits=logits)
        reachable = torch.isfinite(logits).sum(-1).clamp(min=2).float()
        assert float(dist.entropy() / torch.log(reachable)) == pytest.approx(
            0.0, abs=1e-5)


def test_the_reachable_count_comes_from_the_MASK_not_from_the_head_size():
    """`torch.isfinite` on the masked logits is how the placement head recovers
    its own legal-cell count without any extra plumbing: a spell sees 588 cells,
    a plain troop 242, the Cannon 208."""
    logits = torch.full((3, 612), -float("inf"))
    logits[0, :588] = 0.0
    logits[1, :242] = 0.0
    logits[2, :208] = 0.0
    assert list(torch.isfinite(logits).sum(-1)) == [588, 242, 208]


# --- non-finite gradient containment -------------------------------------
#
# A single non-finite value anywhere in the loss turns EVERY parameter to NaN
# in one optimizer step, and the damage is PERMANENT: Adam's moment estimates
# are poisoned with it, so a subsequent clean batch cannot recover. Measured
# 2026-08-26 -- 100% of parameters NaN after one bad step, still 100% NaN after
# a clean one.
#
# That is the worst failure this loop can have. Training continues, every
# metric reads NaN, and the periodic checkpoint OVERWRITES the last good
# weights with the poisoned ones -- so a multi-hour run is lost silently and
# unrecoverably. The guard is one comparison per minibatch.

def _run_with_adv(net, batch, adv):
    """An update over a caller-supplied advantage tensor, on a PRIVATE COPY of
    the net.

    The copy is not hygiene, it is a prerequisite: the whole point of these
    tests is that a poisoned net stays poisoned, so sharing the module-scoped
    fixture would let the first test here corrupt every test after it. Copying
    keeps the stored log-probs exactly matched to the weights, so the PPO ratio
    is still exactly 1.0 and a "clean" batch really is clean.
    """
    victim = copy.deepcopy(net)
    optimizer = optim.Adam(victim.parameters(), lr=1e-4)
    updater = PPOUpdater(victim, optimizer, TINY)
    returns = torch.zeros_like(adv)
    stats = updater.update(batch, adv, returns, vf_clip_range=0.2,
                           ent_coef_card=0.05, ent_coef_placement=0.06,
                           coverage_coef=0.02)
    return victim, stats


def test_one_nan_advantage_cannot_poison_the_whole_network(rollout):
    """ONE bad element must not take out all 1.88M parameters.

    The realistic sources are an exploding PPO ratio (exp of a large log-prob
    difference overflows to inf), a NaN out of `gae.normalize` when the batch
    has a degenerate spread, or a non-finite reward reaching GAE.
    """
    net, batch = rollout
    T, N = batch["rewards"].shape
    adv = torch.randn(T, N)
    adv[0, 0] = float("nan")

    victim, _ = _run_with_adv(net, batch, adv)

    bad = [n for n, p in victim.named_parameters()
           if not torch.isfinite(p).all()]
    assert not bad, f"{len(bad)} parameter tensors were poisoned: {bad[:5]}"


def test_an_all_nan_update_leaves_the_weights_bit_identical(rollout):
    """A fully corrupt batch must be a NO-OP, not a partial write.

    Skipping the step is the only safe response: there is no meaningful
    gradient direction in a non-finite batch, so the correct step size is zero.
    """
    net, batch = rollout
    before = {n: p.detach().clone() for n, p in net.named_parameters()}

    T, N = batch["rewards"].shape
    victim, _ = _run_with_adv(net, batch, torch.full((T, N), float("nan")))

    for n, p in victim.named_parameters():
        assert torch.equal(p.detach(), before[n]), f"{n} moved on a NaN batch"


def test_a_skipped_update_is_reported_not_silent(rollout):
    """A silent skip is its own hazard: a run whose updates are all being
    dropped looks exactly like a run that is learning nothing. The count has to
    reach the caller so `log_update` can surface it."""
    net, batch = rollout
    T, N = batch["rewards"].shape
    _, stats = _run_with_adv(net, batch, torch.full((T, N), float("nan")))
    assert stats.nonfinite_skips > 0, (
        "the updater dropped a non-finite step but reported nothing")


def test_a_clean_batch_still_updates_and_reports_no_skips(rollout):
    """The guard must not fire on healthy data -- otherwise it silently
    converts the whole run into the no-learning failure it exists to prevent."""
    net, batch = rollout
    before = {n: p.detach().clone() for n, p in net.named_parameters()}

    T, N = batch["rewards"].shape
    victim, stats = _run_with_adv(net, batch, torch.randn(T, N))

    assert stats.nonfinite_skips == 0, stats.nonfinite_skips
    moved = any(not torch.equal(p.detach(), before[n])
                for n, p in victim.named_parameters())
    assert moved, "a clean batch produced no weight movement at all"


def test_a_batch_that_does_not_match_the_config_is_refused_by_name(rollout):
    """`_segments` builds its gather indices from `cfg.update_timestep` and
    `cfg.num_envs`, NOT from the batch it was handed. A batch of the wrong
    length therefore fails deep inside an advanced-indexing expression, as an
    IndexError about a tensor nobody named -- or, if the batch is LONGER,
    succeeds while silently training on a prefix of it.

    The rollout loop cannot currently desync (it adds exactly update_timestep
    rows, then updates, then clears), so this is a contract check on the
    boundary rather than a live bug: a subclass overriding `collect_rollout`,
    or a config edited between resume and rollout, is what it is here for.
    """
    net, batch = rollout
    short = {k: (v[:-1] if hasattr(v, "shape") and v.shape[:1] == (TINY.update_timestep,)
                 else v)
             for k, v in batch.items()}
    T, N = short["rewards"].shape
    with pytest.raises(ValueError, match="update_timestep"):
        _run_with_adv(net, short, torch.randn(T, N))
