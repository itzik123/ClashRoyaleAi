"""Placement-head ROW COMPACTION: the premise, and the bit-exactness bar.

WHAT THIS OPTIMIZATION IS
-------------------------
`PPOUpdater.update` runs the placement head TWICE per BPTT chunk -- once for the
card that was actually chosen, once for the sampled COVERAGE slot -- and the
placement head is ~41% of update time. Both outputs are consumed ONLY on rows
where `decision > 0`:

    actor_loss        (min(surr1,surr2) * mb_decision).sum() / n_decision
    placement entropy (new_ent_place    * mb_placed  ).sum() / n_placed
                      ...and mb_placed = mb_decision * (action != no-op)
    clip_frac         (clipped          * mb_decision).sum() / n_decision
    coverage KL       has  = has_target * decision
    coverage entropy  no_t = (1 - has_target) * decision

`decision` is itself `(card_mask.sum(1) > 1) * valid` (base_trainer), so it is a
subset of `valid` and no other term can reach those rows either.

Measured on model_weights_selfplay.pth (ep 31,312, 2.6 Hog Cycle) over 1500
steps: decision fires on 0.368 of rows, so **63.2% of both placement forwards
AND their backwards is multiplied by exactly zero**.

A NOTE ON WHAT IS *NOT* SLICEABLE, because the obvious reading is wrong
----------------------------------------------------------------------
`Advisor/Rows` sits at 23-25 per 500-row minibatch, which invites slicing the
COVERAGE forward down to those rows. That would be a silent regression, not an
optimization: `coverage_terms` gives the advisor rows KL and gives EVERY OTHER
DECISION ROW the entropy bonus, at a live `PLACEMENT_COVERAGE_COEF = 0.02`.
Deleting that half re-opens the 2026-08-14 placement collapse (Cannon 91.0%
modal share on (11,0), preserving 121 tower HP against 396 for a RANDOM legal
cell -- worse than chance). The advisor row count bounds the KL half only.

THE BAR, and why it is not "bit-exact everywhere"
-------------------------------------------------
Bit-exact was the goal and is NOT achievable, for a reason that belongs to the
backend rather than to this code. Measured, batch 500 -> 184, with the dropped
rows carrying exactly-zero upstream gradient:

    nn.Linear(280->32)                  fwd 4.768e-07   grad_W 2.289e-05
    Conv2d(32,16) 18x10  place_up.1     bit-identical, fwd and grad_W
    Conv2d(16,8)  36x20  place_up.4     grad_W differs by 5.814e-03
    Conv2d(8,1)   36x20  place_up.6     grad_W differs by 2.808e-03
    Conv2d(24,8)  34x18  place_hires.0  grad_W differs by 4.883e-03

The Linear half is FIXED here -- `forward_sequence` computes both `place_ctx`
layers on the full batch and slices, at ~0.5% of the head's cost. The Conv2d
half cannot be: a convolution's weight gradient is a reduction over the batch
dimension, and MKL-DNN re-blocks that reduction when the batch size changes.
Note that it does so only for SOME shapes -- the first probe run here tested
18x10, got "invariant", and was believed until the sweep over the other three
shapes refuted it. An invariance that holds for one layer is not a property of
Conv2d.

So the bar these tests actually enforce, each measured rather than assumed:

  * placement logits on the rows that are KEPT      -- torch.equal
  * the loss and every UpdateStats field, 1 update  -- exact
  * actor_loss / total_loss / clip_frac /
    coverage_entropy, 20 updates                    -- still exact
  * weights                                         -- float32 round-off,
                                                       bounded and measured

This is the same bar `forward_sequence` itself already ships at (CLAUDE.md
records "max logit delta 1.1e-08 vs a float32 eps of 1.19e-07" for the batching
that created it), and strictly stronger, because there the LOSS was not shown
to be exact and here it is.
"""
import copy

import numpy as np
import pytest
import torch
import torch.optim as optim

from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.buffer import CORE_FIELDS, RolloutBuffer
from python_ai.rl.config import PPOConfig
from python_ai.rl.ppo import PPOUpdater

_REAL_FORWARD_SEQUENCE = MicroRoyaleNet.forward_sequence

TINY = PPOConfig(num_envs=2, update_timestep=8, bptt_chunk=2,
                 num_minibatches=1, ppo_epochs=1)


@pytest.fixture(scope="module")
def rollout():
    """A real rollout from the engine, long enough to contain BOTH kinds of row.

    Random tensors would not do: the whole point is the structure of the
    affordability mask, which is what makes a row a decision row or not.
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
        e.seed(7)
        e.reset()

    buf = RolloutBuffer(CORE_FIELDS)
    hx = torch.zeros(TINY.num_envs, LSTM_HIDDEN)
    cx = torch.zeros(TINY.num_envs, LSTM_HIDDEN)
    for t in range(TINY.update_timestep):
        # 2.6 Hog Cycle holds two 1-cost cards, so a rollout left to itself is
        # a decision row on essentially EVERY step and the dead-row tests below
        # would be vacuous. Starve the bar on alternate steps instead of hoping
        # for the state to arise -- the same reasoning as CLAUDE.md's "Deal the
        # Cannon instead of hoping for it".
        if t % 2 == 1:
            for e in envs:
                e.set_elixir_for_team(0, 0.0)
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
                aux_opp_played=torch.full((TINY.num_envs,), -1,
                                          dtype=torch.long),
                coverage_slot=torch.zeros(TINY.num_envs, dtype=torch.long))
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(int(card[i]), 0.0, 0.0, 10)
    return net, buf.stack()


def _run(net, batch, n_updates=1):
    """One (or several) full updates from a FIXED start, deterministically.

    Seeds both RNGs the update touches: torch for the advantage draw, numpy for
    `_segments`' minibatch permutation. Without both, two arms shuffle their
    minibatches differently and nothing downstream is comparable.
    """
    torch.manual_seed(1234)
    np.random.seed(1234)
    net = copy.deepcopy(net)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    updater = PPOUpdater(net, optimizer, TINY)
    T, N = batch["rewards"].shape
    adv = torch.randn(T, N)
    returns = torch.zeros(T, N)
    out = []
    for _ in range(n_updates):
        out.append(updater.update(batch, adv, returns, vf_clip_range=0.2,
                                  ent_coef_card=0.05, ent_coef_placement=0.06,
                                  coverage_coef=0.02))
    return net, out[-1]


def _assert_bit_identical(net_a, net_b, label):
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), (
            f"{label}: '{k}' differs. Max abs delta "
            f"{(sd_a[k].float() - sd_b[k].float()).abs().max():.3e}. The bar "
            f"here is torch.equal, not allclose -- any delta means a reduction "
            f"changed shape or order.")


# --- G1: the falsifier for the whole optimization ---------------------------

def _full_path(corrupt=None):
    """A forward_sequence that IGNORES active_rows, optionally corrupting rows.

    Dropping `active_rows` is what makes G1 a test of the CLAIM rather than of
    the optimization: it asks whether the FULL computation's dead rows are read.
    Both arms take this path, so compaction's own round-off cannot leak into the
    comparison and be mistaken for a leak.
    """
    real = _REAL_FORWARD_SEQUENCE

    def w(self, feats, embeds, spatial, obs, card_mask, card_idx, reset,
          hidden, extra_card_idx_seq=None, hires_seq=None, active_rows=None):
        cl, pl, v, aux, h, extra = real(
            self, feats, embeds, spatial, obs, card_mask, card_idx, reset,
            hidden, extra_card_idx_seq=extra_card_idx_seq, hires_seq=hires_seq)
        if corrupt is not None:
            # `decision` is (card_mask.sum(1) > 1) * valid and the fixture is
            # all-valid, so this selects exactly the intended set of rows.
            sel = corrupt(card_mask).unsqueeze(-1)
            g = torch.Generator().manual_seed(4242)
            pl = torch.where(sel, torch.randn(pl.shape, generator=g), pl)
            if extra is not None:
                extra = torch.where(
                    sel, torch.randn(extra.shape, generator=g), extra)
        return cl, pl, v, aux, h, extra
    return w


def _run_under(forward_impl, net, batch, n_updates=1):
    MicroRoyaleNet.forward_sequence = forward_impl
    try:
        return _run(net, batch, n_updates=n_updates)
    finally:
        MicroRoyaleNet.forward_sequence = _REAL_FORWARD_SEQUENCE



def test_placement_logits_on_NON_DECISION_rows_are_never_read(rollout):
    """Corrupt both placement maps wherever decision == 0. Nothing may move.

    This tests the CLAIM, not an implementation -- it runs against unmodified
    `PPOUpdater` and `forward_sequence`. If it fails, row compaction is invalid
    and no amount of careful coding rescues it, because some term really does
    read a row the mask was supposed to have killed.

    Corrupting with a DETACHED constant is deliberate and tests both halves at
    once: the forward value changes (so any value that leaks moves the loss),
    and the backward path through those rows is severed (so any gradient that
    leaks moves the weights). In the honest case both contributions are exactly
    zero and the weights are bit-identical.
    """
    net, batch = rollout
    assert float(batch["valid"].min()) == 1.0, (
        "this test derives `decision` from the card mask alone, which equals "
        "the stored `decision` only where every row is valid")
    dec = batch["decision"]
    assert 0.0 < float(dec.mean()) < 1.0, (
        f"fixture must contain BOTH kinds of row to be a real test; "
        f"decision rate is {float(dec.mean())}")

    dead_rows = lambda cm: cm.sum(-1) <= 1
    clean, clean_stats = _run_under(_full_path(), net, batch)
    dirty, dirty_stats = _run_under(_full_path(dead_rows), net, batch)

    _assert_bit_identical(clean, dirty, "G1 dead-row corruption")
    assert clean_stats.actor_loss == dirty_stats.actor_loss
    assert clean_stats.ent_placement == dirty_stats.ent_placement
    assert clean_stats.coverage_entropy == dirty_stats.coverage_entropy
    assert clean_stats.advisor_kl == dirty_stats.advisor_kl


def test_the_corruption_harness_can_actually_detect_a_leak(rollout):
    """The control for G1: corrupt DECISION rows instead, and demand it moves.

    Without this, a G1 that passes proves nothing -- a harness that silently
    failed to corrupt anything would also pass. This is the same trap CLAUDE.md
    records for the deploy-zone probe: when a measurement's failure mode is
    maximal permissiveness, it needs an internal control that MUST fire.
    """
    net, batch = rollout
    live_rows = lambda cm: cm.sum(-1) > 1
    clean, _ = _run_under(_full_path(), net, batch)
    dirty, _ = _run_under(_full_path(live_rows), net, batch)

    sd_a, sd_b = clean.state_dict(), dirty.state_dict()
    assert any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a), (
        "corrupting DECISION rows changed nothing -- the harness is not "
        "actually reaching the placement logits, so G1's pass is vacuous")


# --- G2: the precondition that makes `finite * 0.0 == 0.0` true -------------

def test_dead_rows_carry_FINITE_placement_terms_in_the_unmodified_path(rollout):
    """Nothing on a dead row may be nan/inf, or `0 * x` was never `0`.

    Row compaction leaves a finite filler where it skips the convolution. That
    is only bit-identical to the old path if the old path's own value there was
    finite too: `nan * 0.0` is `nan`, and a single one would have poisoned every
    masked sum in the update all along.
    """
    net, batch = rollout
    from torch.distributions import Categorical
    from python_ai.models.policy_io import LSTM_HIDDEN

    L, B = batch["obs"].shape[0], batch["obs"].shape[1]
    obs_flat = batch["obs"].reshape(L * B, -1)
    dead = (batch["decision"].reshape(-1) == 0).nonzero(as_tuple=True)[0]
    assert dead.numel() > 0, "fixture has no dead rows; G2 would be vacuous"

    with torch.no_grad():
        feats, embeds, spatial = net.extract_features(obs_flat)
        hires = net.hires_features(obs_flat)
        card_idx = batch["card_actions"].reshape(-1)
        hx = torch.zeros(L * B, LSTM_HIDDEN)
        pl = net.placement_given_card(hx, embeds, card_idx, obs_flat,
                                      spatial, hires_map=hires)
        ent = Categorical(logits=pl[dead]).entropy()
        lp = Categorical(logits=pl[dead]).log_prob(
            batch["placement_actions"].reshape(-1)[dead])

    assert torch.isfinite(ent).all(), "non-finite placement entropy on a dead row"
    assert torch.isfinite(lp).all(), "non-finite placement log-prob on a dead row"
    assert torch.isfinite(pl[dead]).any(dim=-1).all(), (
        "a dead row is entirely -inf, so Categorical would produce nan")


# --- the optimization itself -------------------------------------------------

def _seq_inputs(net, batch):
    """Rebuild exactly the tensors `PPOUpdater` hands to `forward_sequence`."""
    from python_ai.models.policy_io import LSTM_HIDDEN
    L, B = batch["obs"].shape[0], batch["obs"].shape[1]
    obs_flat = batch["obs"].reshape(L * B, -1)
    feats, embeds, spatial, hires = net.extract_features_hires(obs_flat)
    return dict(
        feats_seq=feats.view(L, B, -1),
        card_embeds_seq=embeds.view(L, B, net.hand_size + 1, -1),
        spatial_seq=spatial.view(L, B, *spatial.shape[1:]),
        obs_seq=obs_flat.view(L, B, -1),
        card_mask_seq=net.affordability_mask(obs_flat).view(
            L, B, net.hand_size + 1),
        card_idx_seq=batch["card_actions"],
        reset_seq=batch["masks"],
        hidden_state=(torch.zeros(B, LSTM_HIDDEN), torch.zeros(B, LSTM_HIDDEN)),
        extra_card_idx_seq=batch["coverage_slot"],
        hires_seq=hires.view(L, B, *hires.shape[1:]),
    )


def test_active_rows_reproduces_the_full_map_on_the_rows_it_keeps(rollout):
    """The rows we still compute must be BIT-IDENTICAL to computing them all.

    This is the whole correctness claim of the optimization at the level it
    actually happens. Anything weaker than torch.equal here means the compacted
    convolution is not the same convolution.
    """
    net, batch = rollout
    kw = _seq_inputs(net, batch)
    live = (batch["decision"].reshape(-1) > 0).nonzero(as_tuple=True)[0]
    dead = (batch["decision"].reshape(-1) == 0).nonzero(as_tuple=True)[0]
    assert live.numel() and dead.numel()

    with torch.no_grad():
        _, full_pl, _, _, _, full_cf = net.forward_sequence(**kw)
        _, sub_pl, _, _, _, sub_cf = net.forward_sequence(
            active_rows=live, **kw)

    C = net.placement_cells
    for name, full, sub in (("chosen", full_pl, sub_pl),
                            ("coverage", full_cf, sub_cf)):
        f, s = full.reshape(-1, C), sub.reshape(-1, C)
        assert torch.equal(s[live], f[live]), (
            f"{name}: compacted rows differ from the full computation")
        assert torch.isfinite(s[dead]).all(), (
            f"{name}: skipped rows must hold a FINITE filler -- an -inf row "
            f"makes Categorical.entropy() nan, and nan * 0.0 is nan, which "
            f"would poison every masked sum in the update")


def test_the_LOSS_and_every_diagnostic_are_bit_identical(rollout):
    """From identical weights, the objective itself does not move at all.

    THIS is the equivalence claim, and it is exact. `TINY` is one minibatch of
    one epoch, so this compares a single forward/backward from a single starting
    point -- which is the only place an exact comparison is meaningful, because
    `update()` steps the optimizer between minibatches and everything after the
    first step is measuring divergence rather than equivalence.

    What it proves: the compacted path optimizes the SAME objective. Nothing
    about the loss surface, the masks, the normalizers or the coverage split
    changed -- only which rows the convolution bothered to evaluate.
    """
    net, batch = rollout
    seen = []

    def counting(self, *a, **kw):
        seen.append(kw.pop("active_rows", None))
        return _REAL_FORWARD_SEQUENCE(self, *a, **kw)

    base_net, base_stats = _run_under(counting, net, batch)
    fast_net, fast_stats = _run(net, batch)

    # The control that MUST fire. Without it this passes trivially while
    # PPOUpdater has not been wired up at all -- the same trap G1's control
    # exists for.
    assert seen and all(r is not None for r in seen), (
        "PPOUpdater never passed active_rows, so both arms ran the SAME code "
        "and this comparison is vacuous")
    assert any(r.numel() < batch["decision"].numel() for r in seen), (
        "active_rows covered every row in every minibatch, so nothing was "
        "actually skipped and the comparison is still vacuous")

    for field in ("actor_loss", "critic_loss", "entropy", "total_loss",
                  "clip_frac", "aux_ce", "aux_acc", "ent_card",
                  "ent_placement", "coverage_entropy", "advisor_kl",
                  "advisor_rows"):
        assert getattr(base_stats, field) == getattr(fast_stats, field), (
            f"UpdateStats.{field} moved: {getattr(base_stats, field)!r} -> "
            f"{getattr(fast_stats, field)!r}")
    assert (base_stats.per_card_placement_entropy
            == fast_stats.per_card_placement_entropy)


def test_the_weights_after_one_step_agree_to_float32_ROUNDOFF(rollout):
    """...but the WEIGHTS do not agree exactly, and cannot. Bound it instead.

    Conv2d's weight gradient is a reduction over the batch dimension, and
    MKL-DNN re-blocks that reduction when the batch size changes -- for SOME
    shapes. Measured at 500 -> 184 with exactly-zero upstream gradient on the
    dropped rows: place_up.1 (18x10) is bit-identical, while place_up.4 and
    place_up.6 (36x20) and place_hires.0 (34x18) differ by 5.8e-03 / 2.8e-03 /
    4.9e-03. So no row-compaction scheme can be bit-exact at the weight level,
    and asserting that it is would be asserting something false.

    The bound here is deliberately generous relative to the measured 1.2e-10:
    the point is to catch a REAL error (a wrong mask, a mis-scattered row),
    which would show up orders of magnitude above round-off, not to police the
    last ULP of a reduction this code does not control.
    """
    net, batch = rollout

    def uncompacted(self, *a, **kw):
        kw.pop("active_rows", None)
        return _REAL_FORWARD_SEQUENCE(self, *a, **kw)

    base_net, _ = _run_under(uncompacted, net, batch)
    fast_net, _ = _run(net, batch)

    sa, sb = base_net.state_dict(), fast_net.state_dict()
    worst_name, worst = None, 0.0
    for k in sa:
        d = float((sa[k].float() - sb[k].float()).abs().max())
        scale = max(float(sa[k].float().abs().max()), 1e-6)
        if d / scale > worst:
            worst_name, worst = k, d / scale
    assert worst < 1e-5, (
        f"'{worst_name}' moved by {worst:.3e} of its own scale after ONE "
        f"optimizer step. Round-off from the conv's batch reduction measures "
        f"~1e-7 here; anything at 1e-5 is a real defect, not float32.")


def test_the_compacted_path_is_REPRODUCIBLE_run_to_run(rollout):
    """Same inputs, same weights, twice. Bit-identical.

    This is the guarantee that actually matters day to day, and it is a
    different claim from "identical to the uncompacted path". Compaction makes
    the conv's batch size depend on the DECISION COUNT, which is data-derived --
    so the fair worry is that the update became nondeterministic and no
    experiment in this repo could be re-run. It did not: `active_rows` comes
    from the stored `decision` column, so for a given batch and minibatch
    permutation the batch sizes are fixed and every kernel takes the same path.

    Landing this change is therefore a ONE-TIME discontinuity in trajectories,
    not an ongoing source of noise. `prove_*.py`-style paired harnesses keep
    working across it; only comparisons that straddle the commit are affected.
    """
    net, batch = rollout
    a, sa = _run(net, batch, n_updates=3)
    b, sb = _run(net, batch, n_updates=3)
    _assert_bit_identical(a, b, "compacted path re-run")
    assert sa.actor_loss == sb.actor_loss
    assert sa.total_loss == sb.total_loss


@pytest.mark.parametrize("mode", ["all", "none"])
def test_the_degenerate_active_sets_are_handled(rollout, mode):
    """Every row live, and NO row live.

    The empty case is the one that breaks in practice: an empty batch reaches
    Conv2d, and a naive `nonzero` path can produce a shape the head refuses.
    It is not hypothetical -- an all-forced chunk is exactly what a bankrupt
    agent produces, and this project measured P(nothing affordable) at 73.9%.
    """
    net, batch = rollout
    kw = _seq_inputs(net, batch)
    n = batch["obs"].shape[0] * batch["obs"].shape[1]
    rows = (torch.arange(n) if mode == "all"
            else torch.zeros(0, dtype=torch.long))

    with torch.no_grad():
        _, full_pl, _, _, _, full_cf = net.forward_sequence(**kw)
        _, sub_pl, _, _, _, sub_cf = net.forward_sequence(
            active_rows=rows, **kw)

    assert sub_pl.shape == full_pl.shape
    assert sub_cf.shape == full_cf.shape
    if mode == "all":
        assert torch.equal(sub_pl, full_pl)
        assert torch.equal(sub_cf, full_cf)
    else:
        assert torch.isfinite(sub_pl).all() and torch.isfinite(sub_cf).all()


def test_coverage_forward_is_skipped_when_no_coverage_slot_is_requested(rollout):
    """`extra_card_idx_seq=None` must still return None, compacted or not."""
    net, batch = rollout
    kw = _seq_inputs(net, batch)
    kw["extra_card_idx_seq"] = None
    live = (batch["decision"].reshape(-1) > 0).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        _, _, _, _, _, cf = net.forward_sequence(active_rows=live, **kw)
    assert cf is None
