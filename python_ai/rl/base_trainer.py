"""The recurrent-PPO training loop, shared by both pipelines.

The base class owns the loop; a subclass supplies what differs:

    aspect                 pipeline 1              pipeline 2
    -------------------    --------------------    -----------------------
    opponent               UtilityTeacher          PFSP league / scripted
    episode bookkeeping    curriculum + phase      scenarios + strategy ROI
    GAE                    plain                   truncation bootstrap
    draw penalty keyed on  dones                   terminateds only
    entropy config         PHASE1_ENTROPY          PHASE2_ENTROPY
    periodic work          replay, handoff         replay, eval, exploiter

A template method because the loop's order is load-bearing (hidden state
captured before the LSTM advances, the coverage slot drawn on the observation
acted on). Every hook runs before the forward pass or after the transition is
stored, so no override can change what enters the PPO objective.
"""
import os
import time
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from python_ai.advisors import advisor_target
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN, load_state_dict_flexible
from python_ai.rewards.shaping import (
    compute_shaping, flawless_defense_bonus, spell_value_weight,
)
from python_ai.rewards.weights import DRAW_PENALTY
from python_ai.rl import gae as gae_mod
from python_ai.rl import abilities as ability_mod
from python_ai.rl.engine_stats import reseat_prev_stats
from python_ai.rl.buffer import (
    ABILITY_FIELDS, ADVISOR_FIELDS, CORE_FIELDS, TRUNCATION_FIELDS, RolloutBuffer,
)
from python_ai.rl.config import aux_warmup_scale
from python_ai.rl.checkpointing import (
    HISTORICAL_CHECKPOINT_DIR, HISTORICAL_CHECKPOINT_INTERVAL_EPISODES,
    atomic_save, clash_settings, run_path, save_historical_snapshot,
    settings_drift, weights_path,
)
from python_ai.rl.config import PPOConfig
from python_ai.rl.coverage import PLACEMENT_COVERAGE_COEF, placement_coverage_slots
from python_ai.rl.engine_stats import extract_engine_stats, opponent_played_card
from python_ai.rl.entropy import EntropyController
from python_ai.rl.episode_metrics import EpisodeMetrics
from python_ai.rl.ppo import PPOUpdater
from python_ai.rl.seeding import engine_seeds, seed_everything, worker_seeds


@dataclass
class StepContext:
    """Everything a per-step hook could need, gathered once, so a subclass never
    reaches into the loop's locals.
    """
    step: int
    obs_tensor: torch.Tensor
    next_obs: np.ndarray
    card_idx: torch.Tensor
    placement_cell: torch.Tensor
    card_mask: torch.Tensor
    raw_rewards: np.ndarray
    terminateds: np.ndarray
    truncateds: np.ndarray
    dones: np.ndarray
    infos: dict
    stats: dict
    prev_stats: Optional[dict]
    prev_dones: np.ndarray


def _trainee_deck():
    from python_ai.deck import DEFAULT_DECK
    return DEFAULT_DECK


class BaseTrainer:
    """Recurrent PPO with truncated BPTT. Subclass and fill in the opponent."""

    # Live checkpoint and TensorBoard directory. Anchored, not cwd-relative:
    # `setup` deletes `log_dir` on a non-resume start.
    weight_path: str = weights_path("model_weights.pth")
    log_dir: str = run_path("runs/clash_royale")
    #: Embedded in historical snapshot filenames; pipeline 2's age gate reads
    #: it.
    pipeline_name: str = "pipeline1"
    #: Console/replay prefix.
    replay_prefix: str = "replay"
    #: True where a scenario window can truncate an episode without ending the
    #: game; the critic must then bootstrap V(final_obs), never 0.
    uses_truncation_bootstrap: bool = False
    #: Pipeline 2 keys DRAW_PENALTY on `terminateds` alone, so a scenario
    #: window that runs out is not penalized as a draw.
    draw_on_terminated_only: bool = False

    def __init__(self, cfg=None, entropy_cfg=None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy = EntropyController(entropy_cfg or self._default_entropy())
        self.episodes_completed = 0
        self.full_resume = False
        self.net = None
        self.optimizer = None
        self.envs = None
        self.writer = None
        self.updater = None
        self.buffer = None
        self.metrics = None
        self.advisor_legal = {}
        self._hx = self._cx = None
        self._obs = None
        self._prev_stats = None
        self._prev_dones = None
        self._prev2_dones = None

    def _default_entropy(self):
        from python_ai.rl.config import PHASE1_ENTROPY
        return PHASE1_ENTROPY

    def build_envs(self):
        raise NotImplementedError

    def build_replay_env(self):
        """A single non-vectorized env for the periodic demo replay, or None."""
        return None

    def ability_engine_slots(self):
        """Engine slot (deck index 1 or 2) driven by each Champion head, in order;
        see rl/abilities.py.
        """
        from python_ai.deck import DEFAULT_DECK
        from python_ai.rl.abilities import ability_engine_slots
        return ability_engine_slots(DEFAULT_DECK)

    def num_ability_slots(self):
        return len(self.ability_engine_slots())

    def load_checkpoint(self):
        """Restore weights and training state. Returns True on a full resume.

        Anything less restarts `episodes_completed` at 0 and so needs a clean
        log directory, or the new run's scalars interleave with the old run's.
        """
        return False

    def checkpoint_payload(self):
        """Extra keys this pipeline persists beyond model/optimizer/episodes."""
        return {}

    def anneal_episodes_done(self):
        """The clock the placement-entropy anneal runs on. Pipeline 2 subtracts
        its last stall re-boost.
        """
        return self.episodes_completed

    def on_step(self, ctx):
        """After the transition is stored, before the episode-end scan."""

    def on_episode_end(self, i, ctx):
        """One env finished. Own the outcome bookkeeping for this pipeline."""

    def on_after_update(self, stats):
        """Extra logging/periodic work after an update is logged and saved."""

    def should_stop(self):
        return self.episodes_completed >= 1_000_000

    def on_finish(self):
        """Final save/handoff."""

    def setup(self):
        cfg = self.cfg
        # Anchored: replays/*.json feed the placement-phase histogram, and a
        # detector aimed at a directory the run did not write to reports a
        # clean board forever.
        os.makedirs(run_path("replays"), exist_ok=True)
        os.makedirs(HISTORICAL_CHECKPOINT_DIR, exist_ok=True)

        # Before anything stochastic. A no-op when cfg.seed is None; see
        # rl/seeding.py.
        if seed_everything(cfg.seed) is not None:
            print(f"Deterministic run: seed={cfg.seed} "
                  "(torch, numpy, stdlib random, and per-worker env seeds)")

        # Report what the deck turns on and off, and refuse a deck that cannot
        # be trained, before any env is built.
        from python_ai.deck import DEFAULT_DECK
        from python_ai.envs.deck_contract import print_report, validate_deck
        print_report(validate_deck(DEFAULT_DECK, strict=False))
        validate_deck(DEFAULT_DECK, strict=True)

        print(f"Initializing {cfg.num_envs} vectorized environments...")
        self.envs = self.build_envs()
        self.net = MicroRoyaleNet(
            num_ability_slots=self.num_ability_slots()).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        # Champion abilities (rl/abilities.py). A Champion-less deck gets no
        # heads, fields or terms.
        self._ability_slots = self.ability_engine_slots()
        self._ability_ready = torch.zeros(
            (cfg.num_envs, len(self._ability_slots)), dtype=torch.bool)

        # Built before the resume, which refills its outcome window, the
        # curriculum gate's input.
        self.metrics = EpisodeMetrics(cfg.num_envs)

        #: When this lineage (a fresh phase 1) began. Restored on resume and
        #: inherited by phase 2; keeps a previous run's snapshots out of the
        #: PFSP pool.
        self.lineage_started_at = time.time()
        self.full_resume = self.load_checkpoint()
        if not self.full_resume and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        fields = list(CORE_FIELDS)
        if self.uses_truncation_bootstrap:
            fields += list(TRUNCATION_FIELDS)
        if advisor_target.enabled():
            fields += list(ADVISOR_FIELDS)
        if self.num_ability_slots() > 0:
            fields += list(ABILITY_FIELDS)
        self.buffer = RolloutBuffer(fields)
        self.advisor_legal = (advisor_target.build_legal_table(self.net)
                              if advisor_target.enabled() else {})
        self.updater = PPOUpdater(self.net, self.optimizer, cfg)

        # Seeded reset: the only route to the engine's own RNG (the
        # opening-hand shuffle). Gymnasium uses the seed for this reset and
        # continues the stream, so the whole run is reproducible.
        _engine_seeds = engine_seeds(self.cfg.seed, self.cfg.num_envs)
        self._obs, _ = self.envs.reset(
            seed=_engine_seeds if self.cfg.seed is not None else None)
        self._prev_stats = None
        self._prev_dones = np.zeros(cfg.num_envs, dtype=bool)
        #: Dones from two steps ago: envs whose previous step was the phantom
        #: post-autoreset step. See engine_stats.reseat_prev_stats.
        self._prev2_dones = np.zeros(cfg.num_envs, dtype=bool)
        self._hx = torch.zeros(cfg.num_envs, LSTM_HIDDEN).to(self.device)
        self._cx = torch.zeros(cfg.num_envs, LSTM_HIDDEN).to(self.device)
        self._last_save_ep = self.episodes_completed
        self._last_historical_ep = self.episodes_completed
        self._last_replay_ep = self.episodes_completed

    def run(self):
        self.setup()
        print(f"Training started on {self.cfg.num_envs} CPU cores simultaneously!")
        try:
            while not self.should_stop():
                self.collect_rollout()
                stats = self.run_update()
                self.log_update(stats)
                self.buffer.clear()
                self._hx, self._cx = self._hx.detach(), self._cx.detach()
                self.periodic(stats)
        except KeyboardInterrupt:
            # Keep the network as of the last update; the rollout in flight is
            # discarded.
            print(">>> Interrupted -- saving the checkpoint before exiting.")
            self.buffer.clear()
            self.save_checkpoint()
            raise
        self.on_finish()
        if self.writer is not None:
            self.writer.close()

    def collect_rollout(self):
        cfg, net = self.cfg, self.net
        for step in range(cfg.update_timestep):
            obs_tensor = torch.tensor(self._obs, dtype=torch.float32).to(self.device)

            # Derived from obs_tensor, which the buffer stores, so the update
            # recomputes it bit-identically.
            card_mask = net.affordability_mask(obs_tensor)
            cov = self._draw_coverage(obs_tensor, card_mask)

            # Captured before step_lstm_and_card advances (hx, cx): a BPTT
            # chunk starting here resumes from it.
            hx_in, cx_in = self._hx, self._cx

            with torch.no_grad():
                # The card is sampled before placement is conditioned on it.
                # The hi-res map is threaded into placement_given_card so the
                # trunk's first conv runs once per step
                # (tests/test_rollout_no_redundant_conv.py).
                features, card_embeds, spatial_map, hires_map = \
                    net.extract_features_hires(obs_tensor)
                card_logits, ab1_logits, ab2_logits, state_value, (self._hx, self._cx) = \
                    net.step_lstm_and_card(features, (hx_in, cx_in), card_mask)
                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()
                placement_logits = net.placement_given_card(
                    self._hx, card_embeds, card_idx, obs_tensor, spatial_map,
                    hires_map=hires_map)
                placement_dist = Categorical(logits=placement_logits)
                placement_cell = placement_dist.sample()
                total_logprob = (card_dist.log_prob(card_idx)
                                 + placement_dist.log_prob(placement_cell))
                # Champion abilities are part of the joint action, sampled
                # under the readiness reported with this observation.
                ability_ready = self._ability_ready.to(self.device)
                ability_logits = [l for l in (ab1_logits, ab2_logits) if l is not None]
                ability_actions, ability_logprob, _ = ability_mod.sample(
                    ability_logits, ability_ready)
                if ability_logits:
                    total_logprob = total_logprob + ability_logprob

            target_x, target_y = net.cell_to_xy(placement_cell)
            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(cfg.num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(cfg.num_envs, 1),
            }
            # Both keys always present, as AsyncVectorEnv's Dict space
            # requires.
            action.update(ability_mod.action_dict(
                ability_actions, self._ability_slots, cfg.num_envs))

            next_obs, raw_rewards, terminateds, truncateds, infos = \
                self.envs.step(action)
            dones = terminateds | truncateds
            stats = extract_engine_stats(infos, cfg.num_envs)
            # Readiness for the next decision. A missing key (phantom step)
            # reads not-ready.
            next_ability_ready = ability_mod.ready_from_infos(
                infos, cfg.num_envs, self._ability_slots)

            boot = self._truncation_bootstrap(dones, raw_rewards, next_obs,
                                              truncateds)

            # A draw is an episode that ended with a near-zero raw reward.
            draw_source = terminateds if self.draw_on_terminated_only else dones
            is_draw = draw_source & (np.abs(raw_rewards) < 0.5)
            draw_penalty = DRAW_PENALTY * is_draw.astype(np.float32)

            # Shaping uses the config's gamma (the potential terms are
            # policy-invariant only with the GAE gamma) and the annealed spell
            # weight. The first real step of an episode is shaped against its
            # own row, not the phantom step's zero defaults.
            prev_for_shaping = reseat_prev_stats(stats, self._prev_stats,
                                                 self._prev2_dones)
            shaping = compute_shaping(
                stats, prev_for_shaping, gamma=cfg.gamma,
                w_spell=spell_value_weight(self.episodes_completed))
            shaping = shaping * (1.0 - self._prev_dones)
            flawless = flawless_defense_bonus(dones, raw_rewards, stats,
                                              prev_for_shaping)
            shaped_rewards = raw_rewards + shaping - draw_penalty + flawless
            self.metrics.accumulate(shaped_rewards, shaping)

            # Under NEXT_STEP autoreset, an env whose previous step ended the
            # episode does not execute this action: it resets and returns
            # reward 0. Not a real transition, so it is excluded from the loss
            # and treated as a trajectory break for GAE and the LSTM state.
            valid = torch.tensor(1.0 - self._prev_dones,
                                 dtype=torch.float32).to(self.device)
            mask = torch.tensor(1.0 - dones,
                                dtype=torch.float32).to(self.device) * valid

            row = {
                "obs": obs_tensor,
                "card_actions": card_idx,
                "placement_actions": placement_cell,
                # A decision if the card head had a real choice or a Champion
                # ability was ready.
                "decision": ((card_mask.sum(dim=1) > 1)
                             | ability_ready.any(dim=1)).float() * valid,
                "hx_in": hx_in,
                "cx_in": cx_in,
                "logprobs": total_logprob,
                "values": state_value.squeeze(-1),
                "rewards": torch.tensor(shaped_rewards,
                                        dtype=torch.float32).to(self.device),
                "masks": mask,
                "valid": valid,
                # Raw per-step stream (-1 = nothing played), turned into
                # next-card labels inside the update.
                "aux_opp_played": torch.tensor(
                    opponent_played_card(infos, cfg.num_envs)).to(self.device),
                "coverage_slot": cov["slot"],
            }
            if self.uses_truncation_bootstrap:
                row["boot_nonterminal"] = boot["nonterminal"] * valid
                row["trunc_flag"] = boot["flag"]
                row["trunc_boot"] = boot["value"]
            if self._ability_slots:
                row["ability_actions"] = ability_actions
                row["ability_ready"] = ability_ready
            if advisor_target.enabled():
                row["coverage_target"] = cov["target"]
                row["coverage_has"] = cov["has"]
            self.buffer.add(**row)

            # Reset hidden states for environments whose episode just ended.
            mask_col = mask.unsqueeze(1)
            self._hx = self._hx * mask_col
            self._cx = self._cx * mask_col

            ctx = StepContext(
                step=step, obs_tensor=obs_tensor, next_obs=next_obs,
                card_idx=card_idx, placement_cell=placement_cell,
                card_mask=card_mask, raw_rewards=raw_rewards,
                terminateds=terminateds, truncateds=truncateds, dones=dones,
                infos=infos, stats=stats, prev_stats=self._prev_stats,
                prev_dones=self._prev_dones)
            self.on_step(ctx)
            for i, done in enumerate(dones):
                if done:
                    self.episodes_completed += 1
                    self.on_episode_end(i, ctx)

            self._obs = next_obs
            self._prev_stats = stats
            self._ability_ready = next_ability_ready
            self._prev2_dones = self._prev_dones
            self._prev_dones = dones

    def _draw_coverage(self, obs_tensor, card_mask):
        """One affordable hand slot per env, plus the advisor's surface for it.

        Drawn in the rollout because the advisor target is a function of the
        live observation; drawing it in the update would pair one card's logits
        with another card's target.
        """
        net = self.net
        hand_ids = net.hand_card_ids(obs_tensor)
        weights = advisor_target.slot_weights_for(hand_ids)
        slot = placement_coverage_slots(
            card_mask.unsqueeze(0), net.hand_size,
            slot_weights=None if weights is None else weights.unsqueeze(0))[0]
        out = {"slot": slot}
        if advisor_target.enabled():
            slot_ids = torch.cat(
                [hand_ids, torch.full((self.cfg.num_envs, 1), -1,
                                      dtype=torch.long)], dim=1)
            cov_ids = slot_ids.gather(1, slot.view(-1, 1)).squeeze(1)
            tgt, has = advisor_target.targets_for_batch(
                obs_tensor.cpu().numpy(), cov_ids.cpu().numpy(),
                self.advisor_legal)
            out["target"] = torch.from_numpy(tgt)
            out["has"] = torch.from_numpy(has.astype(np.float32))
        return out

    def _truncation_bootstrap(self, dones, raw_rewards, next_obs, truncateds):
        """V(final_obs) for episodes that stopped while the game continued.

        A truncation is a scenario window expiring on a live match;
        bootstrapping 0 there would teach the critic that the world ends. Any
        real ending, including a timeout, is terminal. Read from `truncateds`,
        which selfplay_env sets only for an expiring scenario window, not
        inferred from the reward: an exact-tie timeout pays ~0 and must still
        be terminal.
        """
        n = self.cfg.num_envs
        zero = torch.zeros(n, dtype=torch.float32, device=self.device)
        if not self.uses_truncation_bootstrap:
            return {"nonterminal": zero, "flag": zero, "value": zero}
        needs_boot = np.asarray(truncateds, dtype=bool)
        is_terminal = dones & ~needs_boot
        boot_value = zero
        if needs_boot.any():
            # next_obs at a done step is the episode's true final observation,
            # and (hx, cx) is the state that would process it.
            with torch.no_grad():
                feats, _, _ = self.net.extract_features(
                    torch.tensor(next_obs, dtype=torch.float32).to(self.device))
                _, _, _, v, _ = self.net.step_lstm_and_card(
                    feats, (self._hx, self._cx))
            boot_value = torch.where(
                torch.as_tensor(needs_boot, dtype=torch.bool, device=self.device),
                v.squeeze(-1), zero)
        return {
            "nonterminal": torch.as_tensor(1.0 - is_terminal.astype(np.float32),
                                           dtype=torch.float32, device=self.device),
            "flag": torch.as_tensor(needs_boot.astype(np.float32),
                                    dtype=torch.float32, device=self.device),
            "value": boot_value,
        }

    def run_update(self):
        cfg, net = self.cfg, self.net
        # `drain` releases the per-step lists (~218 MB of observations) before
        # the update.
        batch = self.buffer.drain()

        # Value depends only on hx, so no card or placement is needed.
        with torch.no_grad():
            next_obs_t = torch.tensor(self._obs, dtype=torch.float32).to(self.device)
            feats, _, _ = net.extract_features(next_obs_t)
            _, _, _, next_value, _ = net.step_lstm_and_card(
                feats, (self._hx, self._cx))
            next_value = next_value.squeeze(-1)

        advantages = gae_mod.compute_gae(
            batch["rewards"], batch["values"], batch["masks"], next_value,
            cfg.gamma, cfg.gae_lambda,
            boot_nonterminal=batch.get("boot_nonterminal"),
            trunc_flag=batch.get("trunc_flag"),
            trunc_boot=batch.get("trunc_boot"))
        # Critic targets are the raw returns; only advantages are normalized.
        returns = advantages + batch["values"]
        # Normalize over `decision` rows, the only rows the actor loss reads.
        # Forced steps would shift the mean and std, and PPO's clip is
        # asymmetric in sign(A), so an off-centre advantage changes which
        # samples clip. `decision` already excludes phantom rows.
        adv_norm = gae_mod.normalize(advantages, mask=batch["decision"])
        self._note_placements(batch)

        with torch.no_grad():
            keep = batch["valid"] > 0.5
            r, v = returns[keep], batch["values"][keep]
            self._explained_variance = float(gae_mod.explained_variance(r, v))
            # Scaled to this batch's return spread, never tighter than
            # eps_clip. `safe_std` because std() is NaN on fewer than two rows
            # and clamp propagates NaN.
            vf_clip_range = max(cfg.vf_clip_std_frac * gae_mod.safe_std(r),
                                cfg.eps_clip)
        self._vf_clip_range = vf_clip_range

        return self.updater.update(
            batch, adv_norm, returns, vf_clip_range,
            ent_coef_card=self.entropy.coef_card,
            ent_coef_placement=self.entropy.coef_placement,
            coverage_coef=PLACEMENT_COVERAGE_COEF,
            # The next-card aux loss is ramped in; see
            # PPOConfig.aux_warmup_episodes.
            aux_scale=aux_warmup_scale(self.episodes_completed,
                                       cfg.aux_warmup_episodes))

    def _note_placements(self, batch):
        """Feed this rollout's real placements (card id, cell) to the modal-share
        window; no-op, phantom and forced rows are dropped.
        """
        from python_ai.rl.placement_stats import ModalShareWindow
        if getattr(self, "_modal_share", None) is None:
            self._modal_share = ModalShareWindow()
        with torch.no_grad():
            cards = batch["card_actions"].reshape(-1)
            cells = batch["placement_actions"].reshape(-1)
            placed = (cards < self.net.hand_size) & (batch["decision"].reshape(-1) > 0.5)
            if not bool(placed.any()):
                self._modal_share.add_update([], [])
                return
            obs = batch["obs"].reshape(-1, batch["obs"].shape[-1])[placed]
            hand = self.net.hand_card_ids(obs)
            ids = hand.gather(1, cards[placed].view(-1, 1)).squeeze(1)
            self._modal_share.add_update(ids.tolist(), cells[placed].tolist())

    def log_update(self, stats):
        """TensorBoard series shared by both pipelines, plus the controller step."""
        w, ep = self.writer, self.episodes_completed
        # Modal share per card, the conditional-collapse detector. Above ~0.6
        # on a sharp head, a card goes to one cell regardless of the board.
        shares = getattr(self, "_modal_share", None)
        shares = shares.shares() if shares is not None else {}
        if shares:
            from python_ai.engine_constants import card_name
            for cid, share in shares.items():
                w.add_scalar(f"Placement/ModalShare/{card_name(cid)}", share, ep)
            w.add_scalar("Placement/ModalShare_Max", max(shares.values()), ep)
        w.add_scalar("Loss/Actor", stats.actor_loss, ep)
        w.add_scalar("Loss/Critic", stats.critic_loss, ep)
        w.add_scalar("Loss/Entropy", stats.entropy, ep)
        w.add_scalar("Loss/Total", stats.total_loss, ep)
        w.add_scalar("Loss/Clip_Fraction", stats.clip_frac, ep)
        # Recurrent-PPO self-check; must read ~0.
        w.add_scalar("Loss/Ratio_Dev_First_Minibatch", stats.ratio_dev_first, ep)
        if stats.ent_ability == stats.ent_ability:          # not nan
            w.add_scalar("Policy/Entropy_Ability", stats.ent_ability, ep)
        w.add_scalar("Loss/Critic_Explained_Variance", self._explained_variance, ep)
        w.add_scalar("Loss/Value_Clip_Range", self._vf_clip_range, ep)
        w.add_scalar("Loss/Entropy_Card", stats.ent_card, ep)
        w.add_scalar("Loss/Entropy_Placement", stats.ent_placement, ep)
        w.add_scalar("Policy/Entropy_Card_Frac", stats.ent_card, ep)
        w.add_scalar("Policy/Entropy_Placement_Frac", stats.ent_placement, ep)
        # Read against the no-cycle-knowledge baseline: ~ln(8) = 2.08 CE and
        # 0.125 accuracy.
        w.add_scalar("Aux/NextCard_CE", stats.aux_ce, ep)
        w.add_scalar("Aux/NextCard_Acc", stats.aux_acc, ep)
        # The same label off the detached cycle branch. Fast to move (24 dims,
        # no recurrence); the branch's ceiling is ~0.55 accuracy.
        w.add_scalar("Aux/CycleId_CE", stats.cycle_id_ce, ep)
        w.add_scalar("Aux/CycleId_Acc", stats.cycle_id_acc, ep)
        # Must be 0. Printed too, since the guard lets the run survive and
        # nothing else would say a fault happened.
        w.add_scalar("Loss/NonFinite_Skips", stats.nonfinite_skips, ep)
        if stats.nonfinite_skips:
            print(f"  [WARN] {stats.nonfinite_skips} non-finite minibatch "
                  f"gradient(s) dropped this update. The weights are intact, "
                  f"but something upstream produced NaN/inf -- check the "
                  f"reward stream and the PPO ratio.", flush=True)

        frozen_before = self.entropy.frozen_updates
        target_placement = self.entropy.update(
            stats.ent_card, stats.ent_placement, self.anneal_episodes_done())
        if self.entropy.frozen_updates > frozen_before:
            # A held controller and a satisfied one look identical in the
            # coefficient trace.
            print("  [WARN] entropy controller froze this update: the measured "
                  "entropy was not finite (no minibatch survived). Coefficients "
                  "held.", flush=True)
        w.add_scalar("Entropy/Controller_Frozen_Updates",
                     self.entropy.frozen_updates, ep)
        w.add_scalar("Policy/Entropy_Coef_Card", self.entropy.coef_card, ep)
        w.add_scalar("Policy/Entropy_Coef_Placement",
                     self.entropy.coef_placement, ep)
        # Target next to measured: shows whether the controller is fighting the
        # policy.
        w.add_scalar("Entropy/Placement_Target", target_placement, ep)
        w.add_scalar("Entropy/Placement_Measured", stats.ent_placement, ep)
        # Affordable-but-unchosen cards. Measured falling with Coverage steady
        # is sharpening; both falling is the freeze this term prevents.
        w.add_scalar("Entropy/Placement_Coverage", stats.coverage_entropy, ep)
        if stats.ent_placement_noop == stats.ent_placement_noop:  # not NaN
            w.add_scalar("Entropy/Placement_Measured_NoOp",
                         stats.ent_placement_noop, ep)
        w.add_scalar("Advisor/KL", stats.advisor_kl, ep)
        w.add_scalar("Advisor/Rows", stats.advisor_rows, ep)
        # Watch the min, not the penalty: the penalty is 0.0 both on a healthy
        # deck and with the floor switched off.
        w.add_scalar("Deck/Coverage_Penalty", stats.deck_coverage, ep)
        w.add_scalar("Deck/MinCardProb", stats.deck_min_card_prob, ep)
        w.add_scalar("Advisor/Coef", advisor_target.ADVISOR_COVERAGE_COEF, ep)
        # Shows the spell-value anneal is actually running.
        w.add_scalar("Shaping/SpellValueWeight",
                     spell_value_weight(self.episodes_completed), ep)
        self._log_per_card(stats)

        print(f"  >> Update @ ep {ep} | Actor: {stats.actor_loss:.5f} | "
              f"Critic: {stats.critic_loss:.5f} | Entropy: {stats.entropy:.4f} | "
              f"ClipFrac: {stats.clip_frac:.4f} | "
              f"NextCardCE: {stats.aux_ce:.2f} Acc: {stats.aux_acc:.2f} | "
              f"CycleId: {stats.cycle_id_ce:.2f} Acc: {stats.cycle_id_acc:.2f}")

    def _log_per_card(self, stats):
        """H(placement | card), and the minimum over cards.

        A card near 0 places at a fixed point regardless of the board; read it
        next to modal share, since the most-played card legitimately has the
        lowest entropy.
        """
        from python_ai.engine_constants import card_name
        per_card = stats.per_card_placement_entropy
        if not per_card:
            return
        ep = self.episodes_completed
        for cid, value in per_card.items():
            self.writer.add_scalar(
                f"Entropy/Placement_ByCard/{card_name(cid)}", value, ep)
        worst_id, worst = stats.worst_card
        self.writer.add_scalar("Entropy/Placement_ByCard_Min", worst, ep)
        shown = sorted(per_card.items(), key=lambda kv: kv[1])
        line = ("     H(place|card): "
                + "  ".join(f"{card_name(c)[:9]} {v:.3f}" for c, v in shown))
        if stats.ent_placement_noop == stats.ent_placement_noop:
            line += f"  | no-op arm {stats.ent_placement_noop:.3f}"
        print(line)

    def periodic(self, stats):
        cfg = self.cfg
        if self.episodes_completed - self._last_save_ep >= cfg.save_every_episodes:
            self.save_checkpoint()
            self._last_save_ep = self.episodes_completed
        if (self.episodes_completed - self._last_historical_ep
                >= HISTORICAL_CHECKPOINT_INTERVAL_EPISODES):
            path = save_historical_snapshot(
                self.net, self.episodes_completed, self.pipeline_name)
            self._last_historical_ep = self.episodes_completed
            self.on_historical_snapshot(path)
        self.on_after_update(stats)
        if (self.episodes_completed - self._last_replay_ep
                >= cfg.replay_every_episodes):
            if self.record_replay():
                self._last_replay_ep = self.episodes_completed

    def on_historical_snapshot(self, path):
        """Pipeline 2 refreshes its PFSP pool here so the snapshot enters rotation
        at once.
        """

    def record_replay(self):
        from python_ai.rl.replay import record_greedy_replay
        env = self.build_replay_env()
        if env is None:
            return False
        print(f"Generating replay video for episode {self.episodes_completed}...")
        record_greedy_replay(
            self.net, env, self.device,
            os.path.join(run_path("replays"),
                         f"{self.replay_prefix}_ep"
                         f"{self.episodes_completed}.json"))
        return True

    def save_checkpoint(self, verbose=True):
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_completed": self.episodes_completed,
            "outcome_history": list(self.metrics.outcomes),
            # Which deck produced these weights (CLASH_DECK).
            "deck": list(_trainee_deck()),
            # Every CLASH_* setting in force; see checkpointing.clash_settings.
            "clash_settings": clash_settings(),
            "lineage_started_at": float(getattr(self, "lineage_started_at", 0.0)),
        }
        payload.update(self.entropy.state_dict())
        payload.update(self.checkpoint_payload())
        atomic_save(payload, self.weight_path, keep_previous=True)
        if verbose:
            print(f">>> Checkpoint saved to {self.weight_path} "
                  f"(episode {self.episodes_completed})")

    @staticmethod
    def _report_settings_drift(saved):
        """Warn about CLASH_* settings that differ from the checkpoint's.

        A warning, not an error: a deliberate change is legitimate, but it must
        not be silent.
        """
        if saved is None:
            print(">>> [SETTINGS] checkpoint predates the CLASH_* stamp; the "
                  "settings it was trained under cannot be compared.")
            return
        changed, operational = settings_drift(saved, clash_settings())
        fmt = lambda a: "unset" if a is None else repr(a)  # noqa: E731
        if changed:
            print(">>> WARNING: resuming under DIFFERENT CLASH_* SETTINGS than the "
                  "checkpoint was trained with. Each is read at import, so THIS "
                  "run continues under the new value:")
            for key, a, b in changed:
                print(f">>>     {key}: {fmt(a)} -> {fmt(b)}")
        if operational:
            print(">>> [SETTINGS] operational only (paths / cadence / workers / "
                  "seed): " + ", ".join(f"{k} {fmt(a)} -> {fmt(b)}"
                                        for k, a, b in operational))

    def restore_common(self, checkpoint):
        """Model + optimizer + the state every pipeline keeps.

        The optimizer is restored only on a clean model load: Adam moments for
        a different tensor are worse than none.
        """
        clean = load_state_dict_flexible(
            self.net, checkpoint["model"],
            f"{self.pipeline_name} resume ({self.weight_path})")
        if clean:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Optimizer state NOT restored (architecture mismatch above) "
                  "-- starting the optimizer fresh; network weights were still "
                  "warm-started where shapes matched.")
        saved_deck = checkpoint.get("deck")
        if saved_deck is not None and list(saved_deck) != list(_trainee_deck()):
            print(f">>> WARNING: resuming a checkpoint trained on a DIFFERENT DECK "
                  f"{list(saved_deck)} with CLASH_DECK={list(_trainee_deck())}. The "
                  f"weights load, but card embeddings, placement habits and every "
                  f"win rate belong to the old deck.")
        self._report_settings_drift(checkpoint.get("clash_settings"))
        # Legacy checkpoints lack the stamp; 0.0 disables the lineage filter.
        self.lineage_started_at = float(checkpoint.get("lineage_started_at", 0.0))
        self.episodes_completed = checkpoint["episodes_completed"]
        self.entropy.load_state_dict(checkpoint)
        self.metrics.outcomes.extend(checkpoint.get("outcome_history", []))
        return clean
