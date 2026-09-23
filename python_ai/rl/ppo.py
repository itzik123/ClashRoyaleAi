"""The recurrent PPO update: truncated BPTT, per-head entropy, coverage, aux.

Three masks:

  `valid`     0 on phantom post-autoreset steps, whose action never ran.
              Excluded from every loss term.
  `decision`  1 where >= 2 card arms were legal. The actor and entropy terms
              are normalized by this: a forced step is a point mass that
              contributes nothing but would inflate the denominator ~3.7x.
              The critic still uses `valid`.
  `placed`    `decision` and the chosen arm was not the no-op. Placement
              entropy uses this; on no-op steps the sampled cell never
              reaches the board, so it would be a free bonus.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from python_ai.advisors import advisor_target, tactics
from python_ai.rl import deck_coverage
from python_ai.rl.engine_stats import next_card_labels
from python_ai.rl import abilities as ability_mod
from python_ai.rl.optim_step import clip_and_step


def _mean(xs):
    """Mean of a possibly empty list of minibatch stats; NaN when empty.

    Empty means the non-finite guard dropped every minibatch, and 0.0 would
    read as a healthy flat line.
    """
    return float(np.mean(xs)) if xs else float("nan")


@dataclass
class UpdateStats:
    """Everything one PPO update produced, for the caller to log.

    Returned rather than written here: the pipelines log to different series
    and the exploiter logs to none.
    """
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    clip_frac: float = 0.0
    aux_ce: float = 0.0
    aux_acc: float = 0.0
    #: The same next-card loss read off the detached cycle branch rather than
    #: hx: is card identity still recoverable from the 24-dim block alone?
    #: Marginal ~0.22 accuracy, ceiling ~0.55.
    cycle_id_ce: float = 0.0
    cycle_id_acc: float = 0.0
    ent_card: float = 0.0
    ent_placement: float = 0.0
    #: Placement entropy on the no-op arm, as a contrast to `ent_placement`.
    ent_placement_noop: float = float("nan")
    coverage_entropy: float = 0.0
    #: Read `advisor_kl` with `advisor_rows`: KL also falls when the advisor
    #: stops speaking.
    advisor_kl: float = 0.0
    advisor_rows: float = 0.0
    #: Mean over deck cards of relu(floor - P(play card | card in hand)).
    #: Exactly 0.0 on a healthy deck; read with `deck_min_card_prob`. See
    #: rl/deck_coverage.py.
    deck_coverage: float = 0.0
    #: Min over deck cards of P(play card | card in hand).
    deck_min_card_prob: float = 0.0
    #: max |ratio - 1| over the first minibatch of epoch 0, before any weight
    #: has moved. Must be ~0; anything else means rollout and update log-probs
    #: disagree.
    ratio_dev_first: float = 0.0
    #: Mean Champion-ability entropy (fraction of log 2) over rows where
    #: activation was legal; nan without a ready Champion.
    ent_ability: float = float("nan")
    #: card id -> H(placement | card) as a fraction of that card's reachable
    #: maximum. The aggregate cannot see a per-card collapse: a mixture of
    #: sharp, well-separated modes still has high entropy.
    per_card_placement_entropy: dict = field(default_factory=dict)
    #: Minibatches dropped for a non-finite gradient. Must be 0; a sustained
    #: nonzero count is a real numerical fault the guard only contains.
    nonfinite_skips: int = 0

    @property
    def worst_card(self):
        """(card_id, entropy) of the most collapsed card, or None.

        Read next to modal share: the most-played card legitimately has the
        lowest entropy. A good head is sharp but moves its mode with the board.
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
        #: Only the advisor term needs log(placement cells); the entropy heads
        #: are normalized by their reachable arm count.
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
               collect_per_card=True, deck_coverage_coef=None, aux_scale=1.0):
        """Run `ppo_epochs` passes over the rollout and return an UpdateStats.

        `batch` is `RolloutBuffer.stack()`; `advantages_norm` and `returns`
        come from `rl.gae`.
        """
        cfg, net = self.cfg, self.net
        L, B_total = cfg.bptt_chunk, cfg.num_envs
        device = batch["obs"].device

        # The gather indices come from cfg, so a mismatched batch would fail
        # obscurely or silently train on a prefix.
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
        # One backward scan per update: next-card labels need the whole (T, N)
        # block and the episode boundaries.
        aux_label_seq, aux_has_seq = next_card_labels(
            batch["aux_opp_played"], masks_seq, valid_seq)
        hx_in_seq, cx_in_seq = batch["hx_in"], batch["cx_in"]
        coverage_slot_seq = batch["coverage_slot"]
        n_ability = int(getattr(net, "num_ability_slots", 0) or 0)
        ability_actions_seq = batch.get("ability_actions") if n_ability else None
        ability_ready_seq = batch.get("ability_ready") if n_ability else None
        ratio_dev_first = None
        ent_ability_log = []
        coverage_target_seq = batch.get("coverage_target")
        coverage_has_seq = batch.get("coverage_has")
        if coverage_has_seq is None:
            coverage_has_seq = torch.zeros_like(coverage_slot_seq,
                                                dtype=torch.float32)

        nonfinite_skips = 0
        actor_losses, critic_losses, entropy_bonuses = [], [], []
        total_losses, clip_fracs = [], []
        aux_losses, aux_accs = [], []
        cyc_losses, cyc_accs = [], []
        coverage_ents, coverage_kls, coverage_hits = [], [], []
        deck_pens, deck_min_probs = [], []
        # The module default is the shipping value; an argument lets an
        # experiment turn it off.
        deck_coef = (deck_coverage.DECK_COVERAGE_COEF
                     if deck_coverage_coef is None else deck_coverage_coef)
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

                # Batch the non-recurrent feature extraction over the whole
                # chunk, taking the trunk's pre-pool activation for the
                # high-resolution placement branch from the same call.
                mb_obs_flat = obs_seq[tt, ee].reshape(L * B, -1)
                (feats_seq, card_embeds_seq, spatial_seq,
                 hires_seq) = net.extract_features_hires(mb_obs_flat)
                feats_seq = feats_seq.view(L, B, -1)
                hires_seq = hires_seq.view(L, B, *hires_seq.shape[1:])
                spatial_seq = spatial_seq.view(L, B, *spatial_seq.shape[1:])
                mb_obs_seq = mb_obs_flat.view(L, B, -1)
                card_embeds_seq = card_embeds_seq.view(
                    L, B, net.hand_size + 1, -1)
                # Recomputed from the stored observations rather than stored,
                # so it matches the sampling-time mask by construction; a
                # drifting mask would corrupt the ratio.
                card_mask_seq = net.affordability_mask(mb_obs_flat).view(
                    L, B, net.hand_size + 1)

                mb_card_actions = card_actions_seq[tt, ee]
                mb_place_actions = placement_actions_seq[tt, ee]
                mb_masks = masks_seq[tt, ee]

                # Resume from the hidden state recorded at the chunk's first
                # step.
                rhx = hx_in_seq[t0, ev]
                rcx = cx_in_seq[t0, ev]
                # One batched pass over the chunk; only the LSTM is recurrent.
                # Placement is conditioned on the stored card, the one the
                # log-prob is scored against.
                cf_idx = coverage_slot_seq[tt, ee]
                mb_decision = decision_seq[tt, ee]
                # Every consumer of the placement maps is decision-masked, so
                # skipping non-decision rows changes neither loss nor gradient
                # (tests/test_rl_ppo_compaction.py).
                active_rows = (mb_decision.reshape(-1) > 0).nonzero(
                    as_tuple=True)[0]
                fwd = net.forward_sequence(
                    feats_seq, card_embeds_seq, spatial_seq, mb_obs_seq,
                    card_mask_seq, mb_card_actions, mb_masks, (rhx, rcx),
                    extra_card_idx_seq=cf_idx, hires_seq=hires_seq,
                    active_rows=active_rows,
                    # Only passed with abilities, so a Champion-less deck keeps
                    # the old signature.
                    **({"with_ability": True} if ability_actions_seq is not None else {}))
                (cl_seq, pl_seq, new_values, new_aux_logits,
                 _, cf_pl_seq) = fwd[:6]
                card_dist_t = Categorical(logits=cl_seq)
                place_dist_t = Categorical(logits=pl_seq)
                new_logprobs = (card_dist_t.log_prob(mb_card_actions)
                                + place_dist_t.log_prob(mb_place_actions))
                # The Champion ability is part of the joint action, scored
                # under the readiness mask it was sampled with
                # (rl/abilities.py).
                ent_ability_frac = None
                if ability_actions_seq is not None:
                    mb_ready = ability_ready_seq[tt, ee].bool()
                    ab_lp, ab_ent = ability_mod.log_prob_and_entropy(
                        fwd[6], mb_ready, ability_actions_seq[tt, ee])
                    new_logprobs = new_logprobs + ab_lp
                    ent_ability_frac = (ab_ent * mb_ready.float()) / math.log(2.0)

                # Normalize each head by the entropy it can reach,
                # log(n_legal), not log(total arms). See rl/entropy.py.
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
                # Before the clamp: with no decision in the chunk the
                # decision-denominated stats are undefined, not zero.
                has_decision = bool(mb_decision.sum() > 0)
                n_decision = mb_decision.sum().clamp(min=1.0)

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                if ratio_dev_first is None:
                    with torch.no_grad():
                        dev = ((ratios - 1.0).abs() * mb_decision)
                        ratio_dev_first = float(dev.max()) if has_decision else 0.0
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - cfg.eps_clip,
                                    1 + cfg.eps_clip) * mb_adv

                # PPO2 value clipping: take the worse of clipped and unclipped,
                # so the critic cannot dodge the penalty by jumping outside the
                # trust region.
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -vf_clip_range, vf_clip_range)
                critic_loss_per_elem = torch.max(
                    F.mse_loss(new_values, mb_ret, reduction="none"),
                    F.mse_loss(value_clipped, mb_ret, reduction="none"))

                actor_loss = -(torch.min(surr1, surr2)
                               * mb_decision).sum() / n_decision
                critic_loss = (critic_loss_per_elem * mb_valid).sum() / n_valid
                ent_card_mean = (new_ent_card * mb_decision).sum() / n_decision

                # Placement entropy only on steps that placed a card; see the
                # module docstring.
                mb_placed = mb_decision * (
                    mb_card_actions != net.hand_size).float()
                n_placed = float(mb_placed.sum())
                if n_placed > 0.0:
                    ent_place_mean = (new_ent_place * mb_placed).sum() / n_placed
                else:
                    # No placement in the chunk: fall back to the decision
                    # denominator rather than report a 0 the controller would
                    # chase as a collapse.
                    ent_place_mean = (new_ent_place
                                      * mb_decision).sum() / n_decision

                if epoch == 0 and collect_per_card:
                    self._collect_per_card(
                        percard_place_ent, ent_place_noop_log, new_ent_place,
                        mb_obs_flat, mb_card_actions, mb_placed, mb_decision,
                        L, B)

                # Placement coverage: entropy (or KL to the advisor's surface)
                # for every affordable card, chosen or not. A regularizer that
                # never enters the PPO ratio. Its coefficient is fixed because
                # the adaptive placement coefficient drops exactly when an
                # unplayed card is freezing.
                mb_cov_targets = (coverage_target_seq[tt, ee]
                                  if coverage_target_seq is not None else None)
                cov_delta, cov_ent_frac, cov_kl, cov_n = \
                    advisor_target.coverage_terms(
                        cf_pl_seq, mb_cov_targets, coverage_has_seq[tt, ee],
                        mb_decision, coverage_coef, self.log_n_placement)
                coverage_ents.append(float(cov_ent_frac))
                coverage_kls.append(float(cov_kl))
                coverage_hits.append(float(cov_n))

                # Deck coverage: a hinge floor under P(play card | card in
                # hand), per deck card. Also a regularizer outside the ratio (a
                # test scans this region textually, so do not name the log-prob
                # tensor here). Entropy over hand slots cannot replace it: a
                # five-card policy satisfies that while three cards sit at
                # zero. Threat-gated, since a Cannon is worth a lot under
                # attack and nothing on a quiet board; ungated it cost 0.42
                # win-rate points.
                with torch.no_grad():
                    threat = (tactics.threat_level_batch(mb_obs_flat)
                              > tactics.DECK_COVERAGE_THREAT_HP).float()
                deck_pen, deck_min_p, deck_n = deck_coverage.deck_coverage_penalty(
                    cl_seq.reshape(-1, cl_seq.shape[-1]),
                    net.hand_card_ids(mb_obs_flat),
                    mb_decision.reshape(-1),
                    threat=threat)
                if deck_n:
                    deck_pens.append(float(deck_pen.detach()))
                    deck_min_probs.append(deck_min_p)

                # Already fractions of each head's reachable maximum.
                entropy_bonus = (ent_coef_card * ent_card_mean
                                 + ent_coef_placement * ent_place_mean)
                if ent_ability_frac is not None:
                    # Same coefficient and units as the card head: a ready
                    # ability is a 2-way choice.
                    n_ready = ability_ready_seq[tt, ee].float().sum().clamp(min=1.0)
                    ent_ab_mean = ent_ability_frac.sum() / n_ready
                    entropy_bonus = entropy_bonus + ent_coef_card * ent_ab_mean
                    if bool(ability_ready_seq[tt, ee].any()):
                        ent_ability_log.append(float(ent_ab_mean.detach()))

                # Next-opponent-card cross-entropy, over labelled rows only:
                # `aux_has` also drops each episode's tail, which has no next
                # play.
                mb_aux_has = aux_has_seq[tt, ee]
                n_aux = mb_aux_has.sum().clamp_min(1.0)
                aux_ce_all = F.cross_entropy(
                    new_aux_logits.reshape(-1, new_aux_logits.shape[-1]),
                    aux_label_seq[tt, ee].reshape(-1),
                    reduction="none").view_as(mb_aux_has)
                aux_loss = (aux_ce_all * mb_aux_has).sum() / n_aux
                aux_acc = ((new_aux_logits.argmax(-1)
                            == aux_label_seq[tt, ee]).float()
                           * mb_aux_has).sum() / n_aux

                # Cap the aux term's magnitude at the uniform CE, ln(n_cards).
                # The head reads hx without a detach, so a stale head (e.g.
                # after the opponent pool changed) would drag the whole trunk.
                # A detached rescale rather than a clamp, which would zero the
                # gradient exactly when the head must relearn; below the
                # ceiling the factor is 1.
                aux_ceiling = math.log(new_aux_logits.shape[-1])
                aux_for_grad = aux_loss * (
                    aux_ceiling / aux_loss.detach().clamp_min(1e-6)
                ).clamp(max=1.0)

                # Cycle-branch identity loss: same label and mask, read off the
                # detached 24-dim ScalarEncoder branch, so this is the only
                # gradient that branch gets. Recomputed rather than taken from
                # feats_seq, which carries the detached copy.
                if net.cycle_id_head is not None:
                    cyc_feat = net.cycle_features(mb_obs_flat)
                    cyc_logits = net.predict_cycle_card(cyc_feat).view(
                        L, B, -1)
                    cyc_ce_all = F.cross_entropy(
                        cyc_logits.reshape(-1, cyc_logits.shape[-1]),
                        aux_label_seq[tt, ee].reshape(-1),
                        reduction="none").view_as(mb_aux_has)
                    cycle_id_loss = (cyc_ce_all * mb_aux_has).sum() / n_aux
                    cycle_id_acc = ((cyc_logits.argmax(-1)
                                     == aux_label_seq[tt, ee]).float()
                                    * mb_aux_has).sum() / n_aux
                else:
                    cycle_id_loss = torch.zeros((), device=device)
                    cycle_id_acc = torch.zeros((), device=device)

                # The coverage delta is already signed: an entropy bonus
                # (negative) plus an advisor KL penalty (positive).
                loss = (actor_loss + 0.5 * critic_loss - entropy_bonus
                        + cov_delta
                        + deck_coef * deck_pen
                        + (cfg.aux_card_coef * cfg.aux_card_scale * float(aux_scale)
                           * aux_for_grad)
                        + cfg.cycle_id_coef * cycle_id_loss)

                self.optimizer.zero_grad()
                loss.backward()

                # Drop a minibatch with a non-finite gradient. One NaN step
                # poisons every parameter (clip_grad_norm_ multiplies it
                # through) and Adam's moments keep it there, so the next
                # checkpoint would overwrite the good weights.
                if not clip_and_step(self.optimizer, self.net.parameters(),
                                     cfg.max_grad_norm):
                    nonfinite_skips += 1
                    continue

                # Valid-denominated: informative on every surviving minibatch.
                critic_losses.append(critic_loss.item())
                total_losses.append(loss.item())
                aux_losses.append(aux_loss.item())
                aux_accs.append(aux_acc.item())
                cyc_losses.append(cycle_id_loss.item())
                cyc_accs.append(cycle_id_acc.item())

                # Decision-denominated, so undefined, not zero, on a chunk with
                # no choice (common for a spent-down agent). Recording a 0
                # would make EntropyController read a collapse and raise its
                # coefficient; skipping hands the empty case to `_mean([])`,
                # which returns NaN.
                if has_decision:
                    actor_losses.append(actor_loss.item())
                    entropy_bonuses.append(
                        (ent_card_mean + ent_place_mean).item())
                    ent_card_log.append(ent_card_mean.item())
                    ent_place_log.append(ent_place_mean.item())
                    # Clip fraction over decision steps only; forced steps have
                    # ratio exactly 1.
                    clipped = ((ratios - 1.0).abs() > cfg.eps_clip).float()
                    clip_fracs.append(
                        ((clipped * mb_decision).sum() / n_decision).item())

        return UpdateStats(
            actor_loss=_mean(actor_losses),
            critic_loss=_mean(critic_losses),
            entropy=_mean(entropy_bonuses),
            total_loss=_mean(total_losses),
            clip_frac=_mean(clip_fracs),
            aux_ce=_mean(aux_losses),
            aux_acc=_mean(aux_accs),
            cycle_id_ce=_mean(cyc_losses),
            cycle_id_acc=_mean(cyc_accs),
            ent_card=_mean(ent_card_log),
            ent_placement=_mean(ent_place_log),
            ent_placement_noop=_mean(ent_place_noop_log),
            coverage_entropy=_mean(coverage_ents),
            advisor_kl=_mean(coverage_kls),
            advisor_rows=_mean(coverage_hits),
            deck_coverage=_mean(deck_pens),
            deck_min_card_prob=_mean(deck_min_probs),
            nonfinite_skips=nonfinite_skips,
            per_card_placement_entropy={
                cid: float(np.mean(v)) for cid, v in percard_place_ent.items()},
            ratio_dev_first=float(ratio_dev_first or 0.0),
            ent_ability=(float(np.mean(ent_ability_log)) if ent_ability_log
                         else float("nan")),
        )

    def _collect_per_card(self, percard, noop_log, new_ent_place, mb_obs_flat,
                          mb_card_actions, mb_placed, mb_decision, L, B):
        """H(placement | card) per card id, plus the no-op arm for contrast.

        Epoch 0 only, from tensors the update already computed. Entropies are
        per-step fractions of log(n_legal), which is what makes cards with
        different legal areas comparable.
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
