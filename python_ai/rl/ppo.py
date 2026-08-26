"""The recurrent PPO update: truncated BPTT, per-head entropy, coverage, aux.

This is the ~250 lines that were duplicated verbatim between `train.py` and
`train_selfplay.py`. The arithmetic here is byte-for-byte what both loops did;
the extraction adds nothing to the objective and removes nothing from it. What
it does add is a place to TEST it -- `tests/test_rl_ppo.py` pins the masking
rules, the entropy normalization and the fact that the coverage term never
touches the PPO ratio.

THE THREE MASKS, and each one is there because getting it wrong was measured:

  `valid`     0 on phantom post-autoreset steps, where the sampled action was
              never executed. Excluded from every loss term.
  `decision`  1 where >=2 card arms were legal. The ACTOR and the ENTROPY terms
              are normalized by this, not by `valid`: on a forced step the
              masked distribution is a point mass (log-prob 0, ratio 1,
              entropy 0), so it contributes nothing but WOULD inflate the
              denominator and shrink the actor's effective step by ~3.7x.
              The CRITIC still uses `valid` -- the value function must be
              learned on every real state, choice or not.
  `placed`    `decision` AND the chosen arm was not the no-op. The placement
              entropy term uses this. Averaging in the no-op steps let the head
              earn the bonus for free on steps whose sampled cell never reached
              the board: measured 0.462 reported = 0.850 no-op vs 0.090 real.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from python_ai.advisors import advisor_target
from python_ai.rl.optim_step import clip_and_step


def _mean(xs):
    """Mean of a possibly-EMPTY list of minibatch stats.

    Empty means every minibatch in the update was dropped by the non-finite
    guard. NaN is the honest report for that -- returning 0.0 would render a
    numerically broken update as a healthy-looking flat line, which is the
    class of "safe guaranteed zero" this project has been bitten by repeatedly.
    `np.mean([])` would also warn, and the suite is kept warning-clean.
    """
    return float(np.mean(xs)) if xs else float("nan")


@dataclass
class UpdateStats:
    """Everything one PPO update produced, for the caller to log.

    Returned rather than written to a SummaryWriter here on purpose: the two
    pipelines log to different series names and the exploiter logs to none, so
    the updater must not own a writer.
    """
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    clip_frac: float = 0.0
    aux_mse: float = 0.0
    aux_mae: float = 0.0
    ent_card: float = 0.0
    ent_placement: float = 0.0
    #: Placement entropy on the no-op arm. Kept purely as the contrast that
    #: makes the 2026-08-11 fix legible: if this and `ent_placement` ever
    #: converge, the no-op arm stopped being a free ride.
    ent_placement_noop: float = float("nan")
    coverage_entropy: float = 0.0
    #: Read `advisor_kl` and `advisor_rows` as a PAIR, never KL alone. KL falls
    #: both when the head learns the advisor's surface and when the advisor
    #: simply stops speaking, and those are opposite situations.
    advisor_kl: float = 0.0
    advisor_rows: float = 0.0
    #: card id -> H(placement | card) as a fraction of that card's own
    #: reachable maximum. THE conditional-collapse detector: the aggregate
    #: provably cannot see a per-card collapse, because a mixture of eight
    #: sharp, well-separated modes has high entropy even when every component
    #: is a delta. On 2026-08-11 the aggregate read a healthy 0.462 while
    #: Cannon sat at 0.017 with 96.4% of its mass on one cell.
    per_card_placement_entropy: dict = field(default_factory=dict)
    #: Minibatches whose gradient was non-finite and whose optimizer step was
    #: therefore DROPPED. Must be 0 on a healthy run. Anything above 0 means
    #: real gradient was discarded -- see the containment guard in `update`.
    #: A sustained nonzero count is a genuine numerical fault (an exploding
    #: ratio, a bad reward), not something the guard has "handled": the guard
    #: only stops it from becoming permanent.
    nonfinite_skips: int = 0

    @property
    def worst_card(self):
        """(card_id, entropy) of the most collapsed card, or None.

        Read this NEXT TO modal share, not instead of it: the most-played card
        legitimately has the lowest entropy (Mini PEKKA measured 0.086 while
        being the healthiest card in the deck). A good head is SHARP but MOVES
        ITS MODE with the board; a broken one returns one cell regardless.
        """
        if not self.per_card_placement_entropy:
            return None
        cid = min(self.per_card_placement_entropy,
                  key=self.per_card_placement_entropy.get)
        return cid, self.per_card_placement_entropy[cid]


class PPOUpdater:
    """One PPO update over a stacked rollout. Owns no state but the net."""

    def __init__(self, net, optimizer, cfg):
        self.net = net
        self.optimizer = optimizer
        self.cfg = cfg
        #: log(number of placement cells). Only the ADVISOR term needs it --
        #: both entropy heads are normalized per step by their own REACHABLE
        #: arm count, which is the 2026-08-16 fix.
        self.log_n_placement = float(np.log(net.placement_cells))

    def _segments(self, device):
        """Every (chunk-start, env) pair is one independent training segment."""
        cfg = self.cfg
        chunk_starts = np.arange(0, cfg.update_timestep, cfg.bptt_chunk)
        segments = np.array([(cs, e) for cs in chunk_starts
                             for e in range(cfg.num_envs)], dtype=np.int64)
        offsets = torch.arange(cfg.bptt_chunk, dtype=torch.long,
                               device=device).unsqueeze(1)
        return segments, offsets

    def update(self, batch, advantages_norm, returns, vf_clip_range,
               ent_coef_card, ent_coef_placement, coverage_coef,
               collect_per_card=True):
        """Run `ppo_epochs` passes over the rollout and return an UpdateStats.

        `batch` is `RolloutBuffer.stack()`; `advantages_norm` and `returns` come
        from `rl.gae`. `vf_clip_range` is scaled to this batch's own return
        spread -- see `PPOConfig.vf_clip_std_frac`.
        """
        cfg, net = self.cfg, self.net
        L, B_total = cfg.bptt_chunk, cfg.num_envs
        device = batch["obs"].device

        # The gather indices below are built from cfg, not from `batch`, so a
        # batch of the wrong shape fails deep inside an advanced-indexing
        # expression as an IndexError naming no tensor -- or, if it is LONGER
        # than cfg says, succeeds while silently training on a prefix. Name it
        # here instead. The live rollout cannot desync (it adds exactly
        # update_timestep rows, then updates, then clears); this guards the
        # boundary against a subclass overriding `collect_rollout` and against
        # a config edited between a resume and the next rollout.
        got_t, got_n = batch["rewards"].shape[:2]
        if (got_t, got_n) != (cfg.update_timestep, cfg.num_envs):
            raise ValueError(
                f"rollout batch is {got_t}x{got_n} but the config says "
                f"update_timestep={cfg.update_timestep} x "
                f"num_envs={cfg.num_envs}; the update indexes by the CONFIG, "
                "so these must agree")
        segments, chunk_offsets = self._segments(device)
        n_segments = len(segments)
        seg_mb_size = max(1, n_segments // cfg.num_minibatches)

        obs_seq = batch["obs"]
        card_actions_seq = batch["card_actions"]
        placement_actions_seq = batch["placement_actions"]
        masks_seq = batch["masks"]
        valid_seq = batch["valid"]
        decision_seq = batch["decision"]
        values_seq = batch["values"]
        old_logprobs_seq = batch["logprobs"]
        aux_elixir_seq = batch["aux_elixir"]
        hx_in_seq, cx_in_seq = batch["hx_in"], batch["cx_in"]
        coverage_slot_seq = batch["coverage_slot"]
        coverage_target_seq = batch.get("coverage_target")
        coverage_has_seq = batch.get("coverage_has")
        if coverage_has_seq is None:
            coverage_has_seq = torch.zeros_like(coverage_slot_seq,
                                                dtype=torch.float32)

        nonfinite_skips = 0
        actor_losses, critic_losses, entropy_bonuses = [], [], []
        total_losses, clip_fracs = [], []
        aux_losses, aux_maes = [], []
        coverage_ents, coverage_kls, coverage_hits = [], [], []
        ent_card_log, ent_place_log, ent_place_noop_log = [], [], []
        percard_place_ent = defaultdict(list)

        for epoch in range(cfg.ppo_epochs):
            seg_perm = np.random.permutation(n_segments)
            for mb_start in range(0, n_segments, seg_mb_size):
                mb_seg = segments[seg_perm[mb_start:mb_start + seg_mb_size]]
                if len(mb_seg) == 0:
                    continue
                t0 = torch.as_tensor(mb_seg[:, 0], dtype=torch.long, device=device)
                ev = torch.as_tensor(mb_seg[:, 1], dtype=torch.long, device=device)
                B = t0.shape[0]

                # (L, B) gather indices: row l selects timestep t0+l per segment.
                tt = t0.unsqueeze(0) + chunk_offsets
                ee = ev.unsqueeze(0).expand(L, B)

                # Batch the (non-recurrent) CNN + scalar feature extraction over
                # the whole chunk at once, then loop only the cheap LSTMCell.
                mb_obs_flat = obs_seq[tt, ee].reshape(L * B, -1)
                # ...and the trunk's PRE-pool activation with it, for the
                # high-resolution placement branch. Taken from the same call
                # rather than rebuilt inside placement_given_card: identical
                # arithmetic either way, but rebuilding would re-run the trunk's
                # first conv -- the most expensive layer in it -- once per
                # minibatch per epoch.
                (feats_seq, card_embeds_seq, spatial_seq,
                 hires_seq) = net.extract_features_hires(mb_obs_flat)
                feats_seq = feats_seq.view(L, B, -1)
                hires_seq = hires_seq.view(L, B, *hires_seq.shape[1:])
                spatial_seq = spatial_seq.view(L, B, *spatial_seq.shape[1:])
                mb_obs_seq = mb_obs_flat.view(L, B, -1)
                card_embeds_seq = card_embeds_seq.view(
                    L, B, net.hand_size + 1, -1)
                # Recomputed from the SAME stored observations the rollout acted
                # on, so it is bit-identical to the mask applied when the action
                # was sampled. DERIVING it (rather than storing it) is what
                # makes drift impossible -- a mask that drifts silently corrupts
                # the PPO ratio.
                card_mask_seq = net.affordability_mask(mb_obs_flat).view(
                    L, B, net.hand_size + 1)

                mb_card_actions = card_actions_seq[tt, ee]
                mb_place_actions = placement_actions_seq[tt, ee]
                mb_masks = masks_seq[tt, ee]

                # Resume from the hidden state actually recorded at this chunk's
                # first timestep (stored-state truncated BPTT).
                rhx = hx_in_seq[t0, ev]
                rcx = cx_in_seq[t0, ev]
                # ONE batched pass over the whole chunk. Only the LSTM is
                # genuinely recurrent; the card/value/aux/placement heads are
                # pointwise in time. Measured 1.82x on the forward pass, and
                # verified equal to the looped path to within float32 round-off
                # (max abs logit delta 1.1e-08 vs an eps of 1.19e-07).
                #
                # mb_card_actions is the STORED action, not a fresh sample:
                # placement must be conditioned on exactly the card the log-prob
                # is scored against, or the ratio breaks silently.
                cf_idx = coverage_slot_seq[tt, ee]
                # Read UP HERE rather than with the other masks below, because
                # the placement head is now told which rows to bother with.
                # Pure reordering of an index read -- no arithmetic moves.
                mb_decision = decision_seq[tt, ee]
                # EVERY consumer of both placement maps is decision-masked:
                # actor_loss by mb_decision, placement entropy by mb_placed
                # (a subset), clip_frac by mb_decision, and both halves of
                # coverage_terms by decision. So the rows dropped here
                # contribute exactly 0.0 to the loss and exactly 0.0 to the
                # gradient -- pinned by tests/test_rl_ppo_compaction.py, which
                # corrupts those rows in the UNMODIFIED path and demands the
                # resulting weights are bit-identical.
                active_rows = (mb_decision.reshape(-1) > 0).nonzero(
                    as_tuple=True)[0]
                (cl_seq, pl_seq, new_values, new_aux_elixir,
                 _, cf_pl_seq) = net.forward_sequence(
                    feats_seq, card_embeds_seq, spatial_seq, mb_obs_seq,
                    card_mask_seq, mb_card_actions, mb_masks, (rhx, rcx),
                    extra_card_idx_seq=cf_idx, hires_seq=hires_seq,
                    active_rows=active_rows)
                card_dist_t = Categorical(logits=cl_seq)
                place_dist_t = Categorical(logits=pl_seq)
                new_logprobs = (card_dist_t.log_prob(mb_card_actions)
                                + place_dist_t.log_prob(mb_place_actions))

                # --- normalize each head by the entropy it can ACTUALLY reach.
                # Both distributions are already masked to their legal arms, so
                # the most entropy a step can carry is log(n_legal) -- never
                # log(total arms). See rl/entropy.py for the full measurement.
                # clamp(min=2) only guards log(1)=0; single-arm rows carry zero
                # entropy and are excluded by mb_decision regardless.
                n_card_legal = card_mask_seq.sum(-1).clamp(min=2).float()
                n_place_legal = torch.isfinite(pl_seq).sum(-1).clamp(min=2).float()
                new_ent_card = card_dist_t.entropy() / torch.log(n_card_legal)
                new_ent_place = place_dist_t.entropy() / torch.log(n_place_legal)

                mb_adv = advantages_norm[tt, ee]
                mb_ret = returns[tt, ee]
                mb_old_logprobs = old_logprobs_seq[tt, ee]
                mb_old_values = values_seq[tt, ee]
                mb_valid = valid_seq[tt, ee]
                n_valid = mb_valid.sum().clamp(min=1.0)
                n_decision = mb_decision.sum().clamp(min=1.0)

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - cfg.eps_clip,
                                    1 + cfg.eps_clip) * mb_adv

                # Value clipping (PPO2-style): cap how far the critic may move
                # from its rollout-time value in one update. Take the WORSE
                # (larger) of clipped/unclipped so the critic cannot dodge the
                # penalty by jumping back and forth outside the trust region.
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -vf_clip_range, vf_clip_range)
                critic_loss_per_elem = torch.max(
                    F.mse_loss(new_values, mb_ret, reduction="none"),
                    F.mse_loss(value_clipped, mb_ret, reduction="none"))

                actor_loss = -(torch.min(surr1, surr2)
                               * mb_decision).sum() / n_decision
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                ent_card_mean = (new_ent_card * mb_decision).sum() / n_decision

                # Placement entropy is measured and rewarded ONLY on steps that
                # actually PLACED a card -- see the module docstring's `placed`.
                mb_placed = mb_decision * (
                    mb_card_actions != net.hand_size).float()
                n_placed = float(mb_placed.sum())
                if n_placed > 0.0:
                    ent_place_mean = (new_ent_place * mb_placed).sum() / n_placed
                else:
                    # No placement anywhere in the chunk: keep the old
                    # denominator rather than feed the controller a 0, which it
                    # would chase as a total collapse.
                    ent_place_mean = (new_ent_place
                                      * mb_decision).sum() / n_decision

                if epoch == 0 and collect_per_card:
                    self._collect_per_card(
                        percard_place_ent, ent_place_noop_log, new_ent_place,
                        mb_obs_flat, mb_card_actions, mb_placed, mb_decision,
                        L, B)

                # --- placement coverage ---------------------------------
                # Entropy (or, where the advisor has a rule, KL to its surface)
                # for a card that was AFFORDABLE this step, chosen or not. A
                # REGULARIZER, not part of the PPO objective: it never touches
                # new_logprobs, so the ratio is unaffected and the update stays
                # a valid PPO step.
                #
                # Its coefficient is deliberately FIXED rather than tied to the
                # adaptive placement coefficient. The controller lowers that one
                # when REAL placements are sharp enough, which is exactly the
                # condition under which an unplayed card is freezing -- tying
                # the two would switch coverage off precisely when it is needed.
                mb_cov_targets = (coverage_target_seq[tt, ee]
                                  if coverage_target_seq is not None else None)
                cov_delta, cov_ent_frac, cov_kl, cov_n = \
                    advisor_target.coverage_terms(
                        cf_pl_seq, mb_cov_targets, coverage_has_seq[tt, ee],
                        mb_decision, coverage_coef, self.log_n_placement)
                coverage_ents.append(float(cov_ent_frac))
                coverage_kls.append(float(cov_kl))
                coverage_hits.append(float(cov_n))

                # ent_*_mean are ALREADY fractions of each head's reachable
                # maximum (divided per step above), so no second division here.
                entropy_bonus = (ent_coef_card * ent_card_mean
                                 + ent_coef_placement * ent_place_mean)

                # Auxiliary opponent-elixir loss. Masked by mb_valid for the
                # same reason the critic loss is: phantom auto-reset steps carry
                # an observation from the NEXT episode paired with stale
                # bookkeeping, and regressing on those teaches noise.
                aux_err = new_aux_elixir - aux_elixir_seq[tt, ee]
                aux_loss = ((aux_err ** 2) * mb_valid).sum() / n_valid
                aux_mae = (aux_err.abs() * mb_valid).sum() / n_valid

                # cov_delta carries both signs already: the entropy half is a
                # bonus (negative), the advisor KL a penalty (positive).
                loss = (actor_loss + 0.5 * critic_loss - entropy_bonus
                        + cov_delta
                        + cfg.aux_elixir_coef * cfg.aux_elixir_scale * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()

                # THE CONTAINMENT GUARD. A single non-finite element anywhere
                # in this minibatch -- an exp() overflow in the PPO ratio, a
                # NaN out of `gae.normalize` on a degenerate batch, a bad
                # reward reaching GAE -- turns EVERY parameter to NaN in one
                # optimizer step. `clip_grad_norm_` does not stop it: it scales
                # by max_norm/(nan+eps), which is itself nan, so the poison is
                # multiplied THROUGH the clip and into every tensor.
                #
                # Measured 2026-08-26: 32 of 32 parameter tensors NaN after one
                # bad step, and PERMANENTLY so -- Adam's moment estimates carry
                # the NaN forward, so a subsequent clean batch does not recover
                # it. The run then trains on, reports NaN for every metric, and
                # the periodic checkpoint OVERWRITES the last good weights.
                #
                # Dropping the step is the only defensible response: there is
                # no descent direction in a non-finite gradient, so the correct
                # step size is zero. The stats appends are skipped with it, so
                # a corrupt minibatch cannot drag the reported means either.
                if not clip_and_step(self.optimizer, self.net.parameters(),
                                     cfg.max_grad_norm):
                    nonfinite_skips += 1
                    continue

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_bonuses.append((ent_card_mean + ent_place_mean).item())
                ent_card_log.append(ent_card_mean.item())
                ent_place_log.append(ent_place_mean.item())
                total_losses.append(loss.item())
                aux_losses.append(aux_loss.item())
                aux_maes.append(aux_mae.item())
                # Fraction of DECISION samples where the ratio hit the clip
                # range. Forced steps have ratio exactly 1.0 by construction and
                # would dilute this toward 0 however much the policy moved.
                clipped = ((ratios - 1.0).abs() > cfg.eps_clip).float()
                clip_fracs.append(
                    ((clipped * mb_decision).sum() / n_decision).item())

        return UpdateStats(
            actor_loss=_mean(actor_losses),
            critic_loss=_mean(critic_losses),
            entropy=_mean(entropy_bonuses),
            total_loss=_mean(total_losses),
            clip_frac=_mean(clip_fracs),
            aux_mse=_mean(aux_losses),
            aux_mae=_mean(aux_maes),
            ent_card=_mean(ent_card_log),
            ent_placement=_mean(ent_place_log),
            ent_placement_noop=_mean(ent_place_noop_log),
            coverage_entropy=_mean(coverage_ents),
            advisor_kl=_mean(coverage_kls),
            advisor_rows=_mean(coverage_hits),
            nonfinite_skips=nonfinite_skips,
            per_card_placement_entropy={
                cid: float(np.mean(v)) for cid, v in percard_place_ent.items()},
        )

    def _collect_per_card(self, percard, noop_log, new_ent_place, mb_obs_flat,
                          mb_card_actions, mb_placed, mb_decision, L, B):
        """H(placement | card) per card id, plus the no-op arm for contrast.

        Accumulated on epoch 0 only, from tensors the update already computed,
        so it costs no extra simulation and no extra forward pass.

        `new_ent_place` is already divided by log(n_legal) per step, which is
        what makes cards comparable at all: a spell sees 588 legal cells, a
        plain troop 242, the Cannon 208, so a raw nat count is not comparable
        across cards.
        """
        net = self.net
        with torch.no_grad():
            slot_ids = net.hand_card_ids(mb_obs_flat).view(L, B, net.hand_size)
            slot_ids = torch.cat(
                [slot_ids, torch.full((L, B, 1), -1, dtype=slot_ids.dtype,
                                      device=slot_ids.device)], dim=2)
            played_id = slot_ids.gather(
                2, mb_card_actions.unsqueeze(-1)).squeeze(-1)
            sel = mb_placed > 0
            for cid, frac in zip(played_id[sel].tolist(),
                                 new_ent_place[sel].tolist()):
                if cid >= 0:
                    percard[cid].append(frac)
            mb_noop = mb_decision * (mb_card_actions == net.hand_size).float()
            n_noop = float(mb_noop.sum())
            if n_noop > 0.0:
                noop_log.append(float((new_ent_place * mb_noop).sum() / n_noop))
