"""The recurrent-PPO training loop, once, for both pipelines.

WHAT WAS ACTUALLY DUPLICATED. `train.train_ppo()` and
`train_selfplay.train_selfplay_ppo()` were 1,620 and 1,565 lines respectively,
and the overlap was not incidental -- it was the whole algorithm: the same
hyperparameters, the same seventeen-key stats extraction, the same rollout with
the same three masks, the same GAE, the same 250-line minibatch update, the same
entropy controller, the same replay recorder, the same checkpoint cadence. The
differences fit on one screen:

    aspect                 pipeline 1              pipeline 2
    -------------------    --------------------    -----------------------
    opponent               UtilityTeacher          PFSP league / scripted
    episode bookkeeping    curriculum + phase      scenarios + strategy ROI
    GAE                    plain                   truncation bootstrap
    draw penalty keyed on  dones                   terminateds only
    entropy config         PHASE1_ENTROPY          PHASE2_ENTROPY
    periodic work          replay, handoff         replay, eval, exploiter

So this class owns the loop and the subclass owns the differences. That is a
template method, and the reason it is the right shape here rather than
composition-by-callback is that the loop's ORDER is itself load-bearing in
several places (the phase gate must be evaluated before the stage gate; the
hidden state must be captured before the LSTM advances; the coverage slot must
be drawn on the observation the action is taken from), and a template makes that
order live in exactly one place.

WHAT A SUBCLASS MUST NOT DO. It must not touch the rollout's arithmetic. Every
hook below is called either before the forward pass or after the transition is
already stored, so no override can change what enters the PPO objective. That is
deliberate: an override that could silently alter the ratio is the failure this
refactor exists to make impossible.
"""
import os
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
from python_ai.rl.buffer import (
    ADVISOR_FIELDS, CORE_FIELDS, TRUNCATION_FIELDS, RolloutBuffer,
)
from python_ai.rl.checkpointing import (
    HISTORICAL_CHECKPOINT_DIR, HISTORICAL_CHECKPOINT_INTERVAL_EPISODES,
    atomic_save, run_path, save_historical_snapshot, weights_path,
)
from python_ai.rl.config import PPOConfig
from python_ai.rl.coverage import PLACEMENT_COVERAGE_COEF, placement_coverage_slots
from python_ai.rl.engine_stats import extract_engine_stats, opponent_elixir_target
from python_ai.rl.entropy import EntropyController
from python_ai.rl.episode_metrics import EpisodeMetrics
from python_ai.rl.ppo import PPOUpdater
from python_ai.rl.seeding import seed_everything, worker_seeds


@dataclass
class StepContext:
    """Everything a per-step hook could need, gathered once.

    Passed to `on_step` and `on_episode_end` so a subclass never has to reach
    back into the loop's locals -- and so adding a diagnostic to one pipeline
    cannot accidentally change the other's control flow.
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


class BaseTrainer:
    """Recurrent PPO with truncated BPTT. Subclass and fill in the opponent."""

    # -- what a subclass declares -------------------------------------------
    #: Live checkpoint (optimizer + training state). Resumed from.
    #: Both are ANCHORED, not cwd-relative -- `log_dir` especially, because
    #: `setup_writer` shutil.rmtree's it on a non-resume start.
    weight_path: str = weights_path("model_weights.pth")
    log_dir: str = run_path("runs/clash_royale")
    #: Embedded in historical snapshot filenames; pipeline 2's age gate reads it.
    pipeline_name: str = "pipeline1"
    #: Console/replay prefix.
    replay_prefix: str = "replay"
    #: True where a scenario window can truncate an episode without ending the
    #: game -- then the critic must bootstrap V(final_obs), never 0.
    uses_truncation_bootstrap: bool = False
    #: DRAW_PENALTY punishes a real game that timed out. Pipeline 2 keys it on
    #: `terminateds` alone, so a scenario-window truncation (truncated=True,
    #: reward ~0) is not mistaken for a draw and a SUCCESSFUL defence that ran
    #: out its focused window is not spuriously penalized.
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

    # ======================================================================
    # subclass surface
    # ======================================================================
    def _default_entropy(self):
        from python_ai.rl.config import PHASE1_ENTROPY
        return PHASE1_ENTROPY

    def build_envs(self):
        raise NotImplementedError

    def build_replay_env(self):
        """A single non-vectorized env for the periodic demo replay, or None."""
        return None

    def num_ability_slots(self):
        from python_ai.envs.gym_wrapper import DEFAULT_DECK_ABILITY_SLOTS
        return DEFAULT_DECK_ABILITY_SLOTS

    def load_checkpoint(self):
        """Restore weights + training state. Returns True on a FULL resume.

        A full resume is what gates the TensorBoard log wipe: a
        file-exists-but-incompatible load, or a legacy weights-only checkpoint,
        both still restart `episodes_completed` at 0 and therefore need a clean
        log directory just like a from-scratch run -- otherwise the new run's
        scalars interleave with the old run's at the same episode numbers and
        both graphs become unreadable.
        """
        return False

    def checkpoint_payload(self):
        """Extra keys this pipeline persists beyond model/optimizer/episodes."""
        return {}

    def anneal_episodes_done(self):
        """The clock the placement-entropy anneal runs on. Pipeline 2 subtracts
        its last stall re-boost, which is what makes the re-boost do anything."""
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

    # ======================================================================
    # the loop
    # ======================================================================
    def setup(self):
        cfg = self.cfg
        os.makedirs("replays", exist_ok=True)
        os.makedirs(HISTORICAL_CHECKPOINT_DIR, exist_ok=True)

        # FIRST, before anything stochastic happens. Network initialisation is
        # the single biggest random input to a run -- two A/B arms that start
        # from different weights are not comparable -- and `build_envs` draws
        # its per-worker seeds from this too. A no-op when cfg.seed is None,
        # which is the default; see rl/seeding.py.
        if seed_everything(cfg.seed) is not None:
            print(f"Deterministic run: seed={cfg.seed} "
                  "(torch, numpy, stdlib random, and per-worker env seeds)")

        print(f"Initializing {cfg.num_envs} vectorized environments...")
        self.envs = self.build_envs()
        self.net = MicroRoyaleNet(
            num_ability_slots=self.num_ability_slots()).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        # This loop no longer samples/stores/scores Champion abilities at all --
        # with a Champion-less deck they were pure noise in the PPO ratio and
        # the entropy bonus. Fail LOUDLY rather than silently ignoring a
        # Champion that IS in the deck: the net would build the heads, nothing
        # would ever sample them, and the bot would simply never use its
        # Champion with nothing saying why.
        if self.num_ability_slots() > 0:
            raise NotImplementedError(
                "the deck contains a Champion (num_ability_slots > 0), but this "
                "training loop's ability sampling was removed when the deck had "
                "none. Restore the ability_dist sampling/buffering/log-prob "
                "branches before training this deck.")

        # Built BEFORE the resume: `restore_common` refills the outcome window
        # from the checkpoint, and a metrics object created afterwards would
        # silently discard it -- which is the curriculum gate's entire input.
        self.metrics = EpisodeMetrics(cfg.num_envs)

        self.full_resume = self.load_checkpoint()
        if not self.full_resume and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        fields = list(CORE_FIELDS)
        if self.uses_truncation_bootstrap:
            fields += list(TRUNCATION_FIELDS)
        if advisor_target.enabled():
            fields += list(ADVISOR_FIELDS)
        self.buffer = RolloutBuffer(fields)
        self.advisor_legal = (advisor_target.build_legal_table(self.net)
                              if advisor_target.enabled() else {})
        self.updater = PPOUpdater(self.net, self.optimizer, cfg)

        self._obs, _ = self.envs.reset()
        self._prev_stats = None
        self._prev_dones = np.zeros(cfg.num_envs, dtype=bool)
        self._hx = torch.zeros(cfg.num_envs, LSTM_HIDDEN).to(self.device)
        self._cx = torch.zeros(cfg.num_envs, LSTM_HIDDEN).to(self.device)
        self._last_save_ep = self.episodes_completed
        self._last_historical_ep = self.episodes_completed
        self._last_replay_ep = self.episodes_completed

    def run(self):
        self.setup()
        print(f"Training started on {self.cfg.num_envs} CPU cores simultaneously!")
        while not self.should_stop():
            self.collect_rollout()
            stats = self.run_update()
            self.log_update(stats)
            self.buffer.clear()
            self._hx, self._cx = self._hx.detach(), self._cx.detach()
            self.periodic(stats)
        self.on_finish()
        if self.writer is not None:
            self.writer.close()

    # -- rollout ------------------------------------------------------------
    def collect_rollout(self):
        cfg, net = self.cfg, self.net
        for step in range(cfg.update_timestep):
            obs_tensor = torch.tensor(self._obs, dtype=torch.float32).to(self.device)

            # Affordability mask, derived purely from obs_tensor -- which is
            # exactly what the buffer stores, so the update recomputes a
            # bit-identical mask instead of trusting a stored one.
            card_mask = net.affordability_mask(obs_tensor)
            cov = self._draw_coverage(obs_tensor, card_mask)

            # Captured BEFORE step_lstm_and_card advances (hx, cx) -- this is
            # the state a BPTT chunk starting here must resume from.
            hx_in, cx_in = self._hx, self._cx

            with torch.no_grad():
                # Autoregressive placement: the card must actually be SAMPLED
                # before placement can be conditioned on it, so this cannot be a
                # single net(...) call.
                features, card_embeds, spatial_map = net.extract_features(obs_tensor)
                card_logits, _, _, state_value, (self._hx, self._cx) = \
                    net.step_lstm_and_card(features, (hx_in, cx_in), card_mask)
                card_dist = Categorical(logits=card_logits)
                card_idx = card_dist.sample()
                placement_logits = net.placement_given_card(
                    self._hx, card_embeds, card_idx, obs_tensor, spatial_map)
                placement_dist = Categorical(logits=placement_logits)
                placement_cell = placement_dist.sample()
                total_logprob = (card_dist.log_prob(card_idx)
                                 + placement_dist.log_prob(placement_cell))

            target_x, target_y = net.cell_to_xy(placement_cell)
            action = {
                "card_index": card_idx.cpu().numpy(),
                "target_x": target_x.cpu().numpy().reshape(cfg.num_envs, 1),
                "target_y": target_y.cpu().numpy().reshape(cfg.num_envs, 1),
                # Deck has no Champion -> nothing to activate. Explicit zeros
                # rather than omitted: AsyncVectorEnv's Dict-space iteration
                # requires every action_space key to be present.
                "activate_ability_slot1": np.zeros(cfg.num_envs, dtype=np.int64),
                "activate_ability_slot2": np.zeros(cfg.num_envs, dtype=np.int64),
            }

            next_obs, raw_rewards, terminateds, truncateds, infos = \
                self.envs.step(action)
            dones = terminateds | truncateds
            stats = extract_engine_stats(infos, cfg.num_envs)

            boot = self._truncation_bootstrap(dones, raw_rewards, next_obs)

            # A draw is an episode that ENDED with a near-zero raw reward.
            draw_source = terminateds if self.draw_on_terminated_only else dones
            is_draw = draw_source & (np.abs(raw_rewards) < 0.5)
            draw_penalty = DRAW_PENALTY * is_draw.astype(np.float32)

            # gamma passed explicitly: the tower term is potential-based
            # (gamma*Phi(s') - Phi(s)) and its policy-invariance guarantee only
            # holds if this is the SAME gamma GAE uses below. w_spell likewise
            # -- omitting it silently pinned the Fireball-value term at its
            # START weight forever instead of annealing it to zero.
            shaping = compute_shaping(
                stats, self._prev_stats, gamma=cfg.gamma,
                w_spell=spell_value_weight(self.episodes_completed))
            shaping = shaping * (1.0 - self._prev_dones)
            flawless = flawless_defense_bonus(dones, raw_rewards, stats,
                                              self._prev_stats)
            shaped_rewards = raw_rewards + shaping - draw_penalty + flawless
            self.metrics.accumulate(shaped_rewards, shaping)

            # `valid` marks envs whose PREVIOUS step ended the episode. Under
            # gymnasium's NEXT_STEP autoreset such envs do not execute the
            # sampled action at all -- the worker just resets and returns a
            # fresh obs with reward 0 -- so this step is not a real transition:
            # excluded from the loss, and treated as a trajectory break so GAE
            # cannot bootstrap through it and the LSTM state entering the new
            # episode starts clean instead of carrying the dead board.
            valid = torch.tensor(1.0 - self._prev_dones,
                                 dtype=torch.float32).to(self.device)
            mask = torch.tensor(1.0 - dones,
                                dtype=torch.float32).to(self.device) * valid

            row = {
                "obs": obs_tensor,
                "card_actions": card_idx,
                "placement_actions": placement_cell,
                "decision": (card_mask.sum(dim=1) > 1).float() * valid,
                "hx_in": hx_in,
                "cx_in": cx_in,
                "logprobs": total_logprob,
                "values": state_value.squeeze(-1),
                "rewards": torch.tensor(shaped_rewards,
                                        dtype=torch.float32).to(self.device),
                "masks": mask,
                "valid": valid,
                "aux_elixir": torch.tensor(
                    opponent_elixir_target(infos, cfg.num_envs)).to(self.device),
                "coverage_slot": cov["slot"],
            }
            if self.uses_truncation_bootstrap:
                row["boot_nonterminal"] = boot["nonterminal"] * valid
                row["trunc_flag"] = boot["flag"]
                row["trunc_boot"] = boot["value"]
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
            self._prev_dones = dones

    def _draw_coverage(self, obs_tensor, card_mask):
        """One AFFORDABLE hand slot per env, plus the advisor's surface for it.

        Drawn HERE, in the rollout, and buffered -- not resampled inside the PPO
        epoch loop where it used to live. An advisor target is a function of the
        OBSERVATION, so it can only be computed while that observation is the
        live one, and a fresh draw in the update would pair one card's logits
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

    def _truncation_bootstrap(self, dones, raw_rewards, next_obs):
        """V(final_obs) for episodes that ENDED WITHOUT a king dying.

        TRUE terminal (raw reward +/-1) vs TRUNCATION (a natural max-tick
        timeout OR a scenario-window cutoff, raw reward ~0). Only true terminals
        get value 0 bootstrapped; truncations must bootstrap V(final_obs) or the
        critic learns a biased "the world ends here" value. Derived from the raw
        engine reward's sign, so it needs no extra signal from the wrapper.
        """
        n = self.cfg.num_envs
        zero = torch.zeros(n, dtype=torch.float32, device=self.device)
        if not self.uses_truncation_bootstrap:
            return {"nonterminal": zero, "flag": zero, "value": zero}
        is_terminal = dones & (np.abs(raw_rewards) > 0.5)
        needs_boot = dones & ~is_terminal
        boot_value = zero
        if needs_boot.any():
            # next_obs at a done step is the episode's TRUE final observation
            # (gymnasium next-step autoreset), and (hx, cx) here is the state
            # that would process it -- post this step's forward, pre the
            # done-mask reset -- so this is exactly V(final_obs).
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

    # -- update -------------------------------------------------------------
    def run_update(self):
        cfg, net = self.cfg, self.net
        batch = self.buffer.stack()

        # Bootstrap value for the state right after the last stored step. Value
        # depends only on hx and never needs a card or a placement, so this
        # skips past card_logits entirely.
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
        # Critic targets are the RAW returns; only advantages are normalized.
        returns = advantages + batch["values"]
        # Masked by `valid` for the same reason the two statistics below are:
        # a phantom post-autoreset row trains nothing, so it must not set the
        # mean and std that rescale every row that does.
        adv_norm = gae_mod.normalize(advantages, mask=batch["valid"])

        with torch.no_grad():
            keep = batch["valid"] > 0.5
            r, v = returns[keep], batch["values"][keep]
            self._explained_variance = float(gae_mod.explained_variance(r, v))
            # Scaled to THIS batch's own return spread, never tighter than
            # eps_clip -- see PPOConfig.vf_clip_std_frac.
            vf_clip_range = torch.clamp(cfg.vf_clip_std_frac * r.std(),
                                        min=cfg.eps_clip).item()
        self._vf_clip_range = vf_clip_range

        return self.updater.update(
            batch, adv_norm, returns, vf_clip_range,
            ent_coef_card=self.entropy.coef_card,
            ent_coef_placement=self.entropy.coef_placement,
            coverage_coef=PLACEMENT_COVERAGE_COEF)

    def log_update(self, stats):
        """TensorBoard series shared by both pipelines, plus the controller step."""
        w, ep = self.writer, self.episodes_completed
        w.add_scalar("Loss/Actor", stats.actor_loss, ep)
        w.add_scalar("Loss/Critic", stats.critic_loss, ep)
        w.add_scalar("Loss/Entropy", stats.entropy, ep)
        w.add_scalar("Loss/Total", stats.total_loss, ep)
        w.add_scalar("Loss/Clip_Fraction", stats.clip_frac, ep)
        w.add_scalar("Loss/Critic_Explained_Variance", self._explained_variance, ep)
        w.add_scalar("Loss/Value_Clip_Range", self._vf_clip_range, ep)
        w.add_scalar("Loss/Entropy_Card", stats.ent_card, ep)
        w.add_scalar("Loss/Entropy_Placement", stats.ent_placement, ep)
        w.add_scalar("Policy/Entropy_Card_Frac", stats.ent_card, ep)
        w.add_scalar("Policy/Entropy_Placement_Frac", stats.ent_placement, ep)
        # Reported in ELIXIR UNITS so it is directly interpretable: MAE is "how
        # many elixir off is our estimate of what the opponent is holding". An
        # always-guess-the-mean baseline sits near 1.35; anything meaningfully
        # below that means the recurrent state genuinely learned to count.
        w.add_scalar("Aux/OppElixir_MAE", stats.aux_mae, ep)
        w.add_scalar("Aux/OppElixir_MSE", stats.aux_mse, ep)
        # Non-finite minibatches whose optimizer step was dropped. MUST be 0.
        # Printed as well as logged, because the whole point of the guard is
        # that the run now SURVIVES a numerical fault -- which means nothing
        # else in the console would ever tell you one happened.
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
            # The controller could not read this update and held its
            # coefficients. Reported because a HELD controller and a controller
            # that is simply satisfied look identical in the coefficient trace.
            print("  [WARN] entropy controller froze this update: the measured "
                  "entropy was not finite (no minibatch survived). Coefficients "
                  "held.", flush=True)
        w.add_scalar("Entropy/Controller_Frozen_Updates",
                     self.entropy.frozen_updates, ep)
        w.add_scalar("Policy/Entropy_Coef_Card", self.entropy.coef_card, ep)
        w.add_scalar("Policy/Entropy_Coef_Placement",
                     self.entropy.coef_placement, ep)
        # The annealed target NEXT TO the measured value, so "is the controller
        # fighting the policy" stays answerable at a glance -- that comparison is
        # what diagnosed the fixed-target pathology in the first place.
        w.add_scalar("Entropy/Placement_Target", target_placement, ep)
        w.add_scalar("Entropy/Placement_Measured", stats.ent_placement, ep)
        # Affordable-but-unchosen cards. Read NEXT TO Placement_Measured:
        # Measured falling while Coverage stays up is a policy sharpening on the
        # cards it plays; BOTH falling is the freeze this term exists to
        # prevent, and it is the shape that produced the (11,0) collapse.
        w.add_scalar("Entropy/Placement_Coverage", stats.coverage_entropy, ep)
        if stats.ent_placement_noop == stats.ent_placement_noop:  # not NaN
            w.add_scalar("Entropy/Placement_Measured_NoOp",
                         stats.ent_placement_noop, ep)
        w.add_scalar("Advisor/KL", stats.advisor_kl, ep)
        w.add_scalar("Advisor/Rows", stats.advisor_rows, ep)
        w.add_scalar("Advisor/Coef", advisor_target.ADVISOR_COVERAGE_COEF, ep)
        # Proof that the spell-value anneal actually runs -- it was dead code
        # for a whole training era and no test varied its argument.
        w.add_scalar("Shaping/SpellValueWeight",
                     spell_value_weight(self.episodes_completed), ep)
        self._log_per_card(stats)

        print(f"  >> Update @ ep {ep} | Actor: {stats.actor_loss:.5f} | "
              f"Critic: {stats.critic_loss:.5f} | Entropy: {stats.entropy:.4f} | "
              f"ClipFrac: {stats.clip_frac:.4f} | "
              f"OppElixirMAE: {stats.aux_mae:.2f}")

    def _log_per_card(self, stats):
        """H(placement | card), and the minimum over cards.

        THE conditional-collapse detector. A card near 0 here is placing at a
        fixed point regardless of the board -- but read it next to modal share,
        because the most-played card legitimately has the lowest entropy.
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

    # -- periodic work ------------------------------------------------------
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
        """Pipeline 2 refreshes its PFSP pool here so the new snapshot enters
        rotation immediately instead of waiting for the next refresh."""

    def record_replay(self):
        from python_ai.rl.replay import record_greedy_replay
        env = self.build_replay_env()
        if env is None:
            return False
        print(f"Generating replay video for episode {self.episodes_completed}...")
        record_greedy_replay(
            self.net, env, self.device,
            f"replays/{self.replay_prefix}_ep{self.episodes_completed}.json")
        return True

    def save_checkpoint(self, verbose=True):
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_completed": self.episodes_completed,
            "outcome_history": list(self.metrics.outcomes),
        }
        payload.update(self.entropy.state_dict())
        payload.update(self.checkpoint_payload())
        atomic_save(payload, self.weight_path)
        if verbose:
            print(f">>> Checkpoint saved to {self.weight_path} "
                  f"(episode {self.episodes_completed})")

    def restore_common(self, checkpoint):
        """Model + optimizer + the state every pipeline keeps.

        The optimizer is restored ONLY on a clean model load. A warm-started
        architecture change leaves some parameters freshly initialized, and Adam
        moments recorded for a different tensor are worse than none.
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
        self.episodes_completed = checkpoint["episodes_completed"]
        self.entropy.load_state_dict(checkpoint)
        self.metrics.outcomes.extend(checkpoint.get("outcome_history", []))
        return clean
