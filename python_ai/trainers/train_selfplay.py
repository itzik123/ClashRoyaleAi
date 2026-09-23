"""Pipeline 2: the PFSP league.

    python_ai/venv/Scripts/python.exe python_ai/trainers/train_selfplay.py

Pipeline 1 teaches the bot to beat a scripted opponent -- a necessary bootstrap,
not a ceiling. This one introduces the pressure that opponent cannot: the trainee
plays FROZEN SNAPSHOTS OF ITS OWN PAST SELVES, four scripted bots, and the C++
heuristic, sampled by Prioritized Fictitious Self-Play. Both sides genuinely
decide what to play (`ClashEnv::stepSelfPlay`) instead of team 1 being a
built-in bot.

WHAT THIS FILE STILL OWNS, now that the loop lives in `rl/base_trainer.py`:

  * the league -- pool discovery, the fixed Elo roster, the exploiter burst
  * scenario-aware episode bookkeeping (an injected threat is a handicap and
    must stay out of the headline W/L/D)
  * the live strategy read-out: Cards/Game, Elixir@Play, ROI, Fwd, TwrDmg/1k
  * the two entropy RE-BOOST heuristics

WHY THIS ONE NEEDS TRUNCATION BOOTSTRAPPING and pipeline 1 does not: scenarios
inject a focused window that can end an episode without a king dying. See
`BaseTrainer.uses_truncation_bootstrap`.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# FIRST, and before gymnasium/numpy/torch below -- see the same note in
# trainers/train.py. Importing this package caps OPENBLAS_NUM_THREADS, which
# OpenBLAS reads at load time, so pulling numpy ahead of it silently forfeits
# ~353 MB of private commit per process.
import python_ai  # noqa: E402,F401

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import torch  # noqa: E402
from python_ai.engine_constants import card_name  # noqa: E402
from python_ai.envs import selfplay_env  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.envs.selfplay_env import (  # noqa: E402
    BUILTIN_MIN_WEIGHT, BUILTIN_TRAINING_OPPONENTS, MicroRoyaleSelfPlayEnv,
)
from python_ai.envs.scripted_opponents import SCRIPTED_OPPONENTS  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.rewards.shaping import building_hp_end  # noqa: E402
from python_ai.rl.seeding import worker_seeds
from python_ai.rl.base_trainer import BaseTrainer  # noqa: E402
from python_ai.rl.checkpointing import run_path, weights_path  # noqa: E402
from python_ai.rl.config import PHASE2_ENTROPY, PPOConfig  # noqa: E402
from python_ai.trainers import exploiter as exploiter_mod  # noqa: E402
from python_ai.trainers import league  # noqa: E402
from python_ai.trainers.strategy_metrics import (  # noqa: E402
    ScenarioMetrics, StrategyMetrics,
)

# Same escape hatch as before for mutual passivity: once draws cross this
# share of the current 100-episode window (now aggregated across whatever mix
# of pool opponents each worker happened to sample, not one fixed opponent),
# re-boost exploration -- floored entropy is exactly what locks a passive
# equilibrium in place, since neither side has any remaining chance to
# stumble into a different joint strategy. Reward-shaping fixes (see
# rewards/weights.py's W_ELIXIR_TRADE/DRAW_PENALTY/W_ELIXIR_OVERFLOW comments)
# address WHY passivity looked attractive; this addresses the fact that once
# both sides are already sitting in it, ordinary gradient descent has no
# exploration left to climb back out.
STALL_DRAW_RATE_THRESHOLD = 0.5

# Cooldown so this doesn't re-fire every single episode once the window is
# saturated with draws -- gives each boost a real window to actually take
# effect before deciding whether another one is needed.
STALL_REBOOST_COOLDOWN_EPISODES = 1000

# Escape hatch for the failure mode the draw-rate trigger above CAN'T see:
# draws near zero but the aggregate win rate across the pool still stuck
# losing, entropy floored the whole time with nothing ever refreshing it.
# Narrowed to also require win_rate < 0.5 (not just "below some target"):
# confirmed in practice (this project's own earlier ladder-based run) that
# firing an exploration re-boost while ALREADY decisively winning measurably
# HURTS -- there's no better joint policy for injected noise to find in a
# matchup that's already going well, only sampling noise for the entropy
# bonus to add. Below 0.5 the trainee is genuinely losing more than winning
# against the pool it's currently being sampled against, which IS worth
# reacting to.
ENTROPY_STALE_REBOOST_EPISODES = 1500

def selfplay_paths():
    """(selfplay checkpoint, phase-1 bootstrap, TensorBoard dir) for THIS run.

    FOLLOWS PHASE 1'S REDIRECTION (audit 08, gap 3). These were constants, so a
    phase 1 run under CLASH_WEIGHTS / CLASH_LOGDIR handed off to a child that
    bootstrapped from python_ai/model_weights.pth -- which did not exist, so it
    crashed at startup, or which belonged to another run, so it trained the
    wrong network. The child inherits the environment, so deriving from it is
    enough. Without redirection the three paths are exactly the old defaults.
    """
    w = os.environ.get("CLASH_WEIGHTS")
    if w:
        bootstrap = weights_path(w)
        stem, ext = os.path.splitext(bootstrap)
        weights = stem + "_selfplay" + (ext or ".pth")
    else:
        bootstrap = weights_path("model_weights.pth")
        weights = weights_path("model_weights_selfplay.pth")
    logdir = os.environ.get("CLASH_LOGDIR")
    logdir = run_path(logdir + "_selfplay") if logdir else run_path("runs/clash_royale_selfplay")
    return weights, bootstrap, logdir


# Pipeline #1's final artifact -- read ONCE, only to seed a from-scratch
# pipeline #2 run (bare weights only; pipeline #2 keeps its own separate
# episode count/optimizer state in WEIGHT_PATH from then on, so pipeline #1's
# own checkpoint is never overwritten by this script).
WEIGHT_PATH, BOOTSTRAP_FROM_PATH, SELFPLAY_LOG_DIR = selfplay_paths()


class Phase2Trainer(BaseTrainer):
    """PPO inside the PFSP league."""

    pipeline_name = "pipeline2"
    replay_prefix = "selfplay_replay"
    weight_path = WEIGHT_PATH
    log_dir = SELFPLAY_LOG_DIR
    #: A scenario window can end an episode without a king dying, so the critic
    #: must bootstrap V(final_obs) there rather than learn a terminal 0.
    uses_truncation_bootstrap = True
    #: ...and for the same reason DRAW_PENALTY keys on `terminateds` alone: a
    #: scenario cutoff arrives as truncated=True with reward ~0, and a
    #: SUCCESSFUL defence that simply ran out its focused window must not be
    #: punished as a stalled game.
    draw_on_terminated_only = True

    def __init__(self, cfg=None):
        super().__init__(cfg or PPOConfig(), PHASE2_ENTROPY)
        self.historical_pool = []
        self.reference_roster = []
        # Episode at which the entropy anneal clock last reset. Only the two
        # re-boost triggers move it.
        self.entropy_reboost_episode = 0
        self.last_stall_reboost_episode = 0
        self.last_eval_ep = 0
        # League exploiter state. The SCHEDULE survives a restart; the
        # exploiter's own WEIGHTS deliberately do not -- persisting them would
        # double every checkpoint write for a marginal gain, and a resume that
        # re-seeds from the current main agent is more interpretable anyway,
        # since a re-seeded exploiter starts as a bit-exact copy of its target
        # and that is what makes 0.50 the exact null its win rate is read
        # against.
        self.exploiter_state = None
        self.exploiter_burst_index = 0
        self.last_exploiter_burst_ep = None
        self.strategy = None
        self.scenario_metrics = ScenarioMetrics()

    # -- environment --------------------------------------------------------
    def build_envs(self):
        # One seed per worker, derived from the run seed by SeedSequence so the
        # workers stay statistically INDEPENDENT (they must not inject the same
        # scenario in lockstep) while the run stays reproducible. Unseeded runs
        # get [None]*n, i.e. exactly the previous behaviour.
        seeds = worker_seeds(self.cfg.seed, self.cfg.num_envs)
        return gym.vector.AsyncVectorEnv(
            [selfplay_env.make_env(seed=s) for s in seeds])

    def build_replay_env(self):
        if not self.historical_pool:
            return None
        # Scenarios OFF: a demo replay should show a normal full game.
        env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
        # Newest/strongest known pool entry -- purely for a representative demo,
        # unrelated to PFSP sampling or eval.
        env.set_historical_opponent(self.historical_pool[-1])
        return env

    def setup(self):
        super().setup()
        self.strategy = StrategyMetrics(self.cfg.num_envs, DEFAULT_DECK)
        self._refresh_pool(initial=True)

    # -- pool ---------------------------------------------------------------
    def _pool_broadcast(self):
        """SCRIPTED_OPPONENTS and the builtin anchors are permanent pool members
        for TRAINING sampling only, appended on top of `historical_pool` rather
        than mixed into it -- that variable also feeds the evaluation roster,
        which torch.load()s every entry it is given, and a "scripted:X" tag
        would crash there rather than merely misbehave.
        """
        return (self.historical_pool + SCRIPTED_OPPONENTS
                + BUILTIN_TRAINING_OPPONENTS)

    def _refresh_pool(self, initial=False):
        self.historical_pool = league.discover_historical_checkpoints(
            self.episodes_completed,
            since=getattr(self, "lineage_started_at", None))
        if initial and not self.historical_pool:
            raise RuntimeError(
                f"No historical snapshot is at least MIN_OPPONENT_AGE_EPISODES "
                f"({league.MIN_OPPONENT_AGE_EPISODES}) episodes older than the "
                f"current trainee (episode {self.episodes_completed}) -- "
                "pipeline #1 (trainers/train.py) needs to have run long enough "
                "to have saved at least one eligible snapshot before pipeline "
                "#2 has anything old enough to play against.")
        self.envs.call("refresh_pfsp_pool", self._pool_broadcast())
        if initial:
            print(f"PFSP pool initialized with {len(self.historical_pool)} "
                  f"historical + {len(SCRIPTED_OPPONENTS)} scripted + "
                  f"{len(BUILTIN_TRAINING_OPPONENTS)} builtin-heuristic "
                  f"opponent(s) [{', '.join(BUILTIN_TRAINING_OPPONENTS)} at min "
                  f"weight {BUILTIN_MIN_WEIGHT}].")

    def on_historical_snapshot(self, path):
        # This run's own new snapshots join the SAME shared pool. Refreshing
        # here AND broadcasting keeps both PFSP sampling and the evaluation
        # roster growing as the trainee improves.
        self._refresh_pool()

    # -- resume -------------------------------------------------------------
    def anneal_episodes_done(self):
        # Measured on the anneal clock SINCE THE LAST STALL RE-BOOST, not on the
        # raw counter. Until 2026-08-09 this read the raw counter, which made
        # the re-boost dead code: it printed a message and changed nothing.
        return self.episodes_completed - self.entropy_reboost_episode

    def load_checkpoint(self):
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path, map_location=self.device,
                                    weights_only=False)
            clean = self.restore_common(checkpoint)
            if not clean:
                # Part of the network just got reinitialized (architecture
                # change). Entropy is almost certainly floored this deep into
                # training, so without this the freshly-random part would settle
                # into another near-deterministic "averaged" policy before ever
                # exploring enough to discover it can behave differently.
                self.entropy_reboost_episode = self.episodes_completed
                print(">>> Architecture changed on resume -- forcing a fresh "
                      f"entropy re-boost from episode {self.episodes_completed} "
                      "so the reinitialized part actually gets explored.")
            else:
                # Falls back to the OLD ladder-era field name, then to the
                # episode count, so a pre-PFSP checkpoint resumes sanely instead
                # of KeyError-ing.
                self.entropy_reboost_episode = checkpoint.get(
                    "entropy_reboost_episode",
                    checkpoint.get("stage_start_episode",
                                   self.episodes_completed))
            self.last_stall_reboost_episode = checkpoint.get(
                "last_stall_reboost_episode", self.episodes_completed)
            self.last_eval_ep = checkpoint.get("last_eval_ep",
                                               self.episodes_completed)
            self.reference_roster = checkpoint.get("reference_roster", [])
            # Exploiter schedule. Falls back to episodes_completed rather than
            # None, because None means "never bursted" and, past
            # EXPLOITER_FIRST_BURST_EPISODE, should_run_burst() reads that as
            # "due now" -- so a legacy checkpoint would fire an unscheduled
            # 75-minute burst immediately on every resume.
            self.last_exploiter_burst_ep = checkpoint.get(
                "last_exploiter_burst_episode", self.episodes_completed)
            self.exploiter_burst_index = checkpoint.get(
                "exploiter_burst_index", 0)
            self._seed_pfsp(checkpoint)
            print(f"Resumed pipeline #2 (PFSP) from {self.weight_path}: "
                  f"episode {self.episodes_completed}, reference roster size "
                  f"{len(self.reference_roster)}, ent coef c/p "
                  f"{self.entropy.coef_card:.4f}/"
                  f"{self.entropy.coef_placement:.4f}, {self._burst_note()}")
            return True

        if os.path.exists(BOOTSTRAP_FROM_PATH):
            bootstrap = torch.load(BOOTSTRAP_FROM_PATH,
                                   map_location=self.device, weights_only=False)
            state = (bootstrap["model"]
                     if isinstance(bootstrap, dict) and "model" in bootstrap
                     else bootstrap)
            # Same lineage as the phase 1 that produced these weights -- its
            # snapshots are this run's opponent pool, the previous run's are not.
            if isinstance(bootstrap, dict):
                self.lineage_started_at = float(bootstrap.get("lineage_started_at", 0.0))
            load_state_dict_flexible(
                self.net, state,
                f"pipeline2 bootstrap from pipeline1 ({BOOTSTRAP_FROM_PATH})")
            print(f"Seeded pipeline #2's trainee from pipeline #1's "
                  f"{BOOTSTRAP_FROM_PATH} (bare weights only -- pipeline #2 "
                  "keeps its own separate episode count from here).")
            return False

        raise RuntimeError(
            f"Neither {self.weight_path} nor {BOOTSTRAP_FROM_PATH} exists -- "
            "pipeline #2 needs pipeline #1's finished bot to start from.")

    def _burst_note(self):
        if not exploiter_mod.EXPLOITER_ENABLED:
            return "exploiter disabled (see EXPLOITER_ENABLED)"
        if self.last_exploiter_burst_ep is None:
            return "exploiter enabled, no burst on record -- first burst is due now"
        return ("next exploiter burst at ep "
                f"{self.last_exploiter_burst_ep + exploiter_mod.EXPLOITER_CYCLE_EPISODES}")

    @staticmethod
    def merge_pfsp(per_worker):
        """[(stats, counts), ...] from the workers -> pooled (stats, counts).

        COUNT-WEIGHTED, and a worker that never scored an opponent contributes
        nothing for it: its 0.5 is the prior, not a measurement. Keys nobody has
        played are dropped, so a resume leaves them to the prior as before.
        """
        total, weighted = {}, {}
        for stats, counts in per_worker:
            for key, n in (counts or {}).items():
                if n <= 0 or key not in (stats or {}):
                    continue
                total[key] = total.get(key, 0) + int(n)
                weighted[key] = weighted.get(key, 0.0) + float(stats[key]) * int(n)
        return ({k: weighted[k] / total[k] for k in total}, dict(total))

    def _merged_pfsp(self):
        """Pooled PFSP estimates, or (None, None) if the workers cannot say."""
        if self.envs is None:
            return None, None
        try:
            per_worker = self.envs.call("get_pfsp_stats")
        except Exception:        # noqa: BLE001 -- must never break a checkpoint
            return None, None
        stats, counts = self.merge_pfsp(per_worker)
        return (stats or None), (counts or None)

    def _seed_pfsp(self, checkpoint):
        """Push a resumed checkpoint's pooled estimates into every worker.

        Each worker gets the pooled rate and an EQUAL SHARE of the pooled count,
        so the next merge weights them as the interchangeable estimators they
        are. Absent keys mean an older checkpoint: the prior, as before.
        """
        stats = checkpoint.get("pfsp_stats") or {}
        if not stats or self.envs is None:
            return
        counts = checkpoint.get("pfsp_counts") or {}
        share = {k: max(1, int(n) // max(1, self.cfg.num_envs))
                 for k, n in counts.items()}
        self.envs.call("set_pfsp_stats", stats, share)
        worst = min(stats.items(), key=lambda kv: kv[1])
        print(f">>> Restored PFSP estimates for {len(stats)} opponents "
              f"(hardest: {os.path.basename(str(worst[0]))} {worst[1]:.3f}) -- "
              f"without this every opponent restarts at the 0.5 prior.")

    def checkpoint_payload(self):
        payload = {
            "entropy_reboost_episode": self.entropy_reboost_episode,
            "last_stall_reboost_episode": self.last_stall_reboost_episode,
            "last_eval_ep": self.last_eval_ep,
            "reference_roster": self.reference_roster,
            "last_exploiter_burst_episode": self.last_exploiter_burst_ep,
            "exploiter_burst_index": self.exploiter_burst_index,
        }
        # TODO 00.8: the per-opponent PFSP estimates lived only in the workers
        # and were lost on every resume. See MicroRoyaleSelfPlayEnv.set_pfsp_stats.
        stats, counts = self._merged_pfsp()
        if stats:
            payload["pfsp_stats"] = stats
            payload["pfsp_counts"] = counts or {}
        return payload

    # -- per-step diagnostics ----------------------------------------------
    def on_step(self, ctx):
        """Record which envs ACTUALLY got a card down this step.

        The engine silently refuses illegal/unaffordable plays, so "the policy
        chose a card" is not the same as "a card was played" -- only a rise in
        cumulative elixir_spent proves it. `prev_dones` masks the phantom
        post-autoreset step, where the counter restarts at 0 and the delta is
        meaningless.
        """
        if ctx.prev_stats is None:
            return
        spent_delta = (ctx.stats["team0_elixir_spent"]
                       - ctx.prev_stats["team0_elixir_spent"])
        really_played = (spent_delta > 1e-6) & (~ctx.prev_dones)
        if not really_played.any():
            return
        hand_ids = self.net.hand_card_ids(ctx.obs_tensor).cpu().numpy()
        elixir = self.net.elixir_from_obs(ctx.obs_tensor).cpu().numpy()
        card_idx = ctx.card_idx.cpu().numpy()
        for i in np.nonzero(really_played)[0]:
            slot = int(card_idx[i])
            card_id = (int(hand_ids[i, slot]) if slot < self.net.hand_size
                       else None)
            self.strategy.record_play(
                i, card_id, float(elixir[i]),
                int(ctx.placement_cell[i].item()), self.net.board_width)

    # -- per-episode --------------------------------------------------------
    def on_episode_end(self, i, ctx):
        infos = ctx.infos
        is_scenario = infos.get(
            "is_scenario", np.zeros(self.cfg.num_envs, dtype=np.float32))[i] > 0.5
        if is_scenario:
            # Scenario episodes stay OUT of the matchup histories so the
            # headline W/L/D and the length/building-HP curves remain pure
            # normal-game signals.
            defensive = infos.get(
                "scenario_defensive",
                np.zeros(self.cfg.num_envs, dtype=np.float32))[i] > 0.5
            self.scenario_metrics.record(defensive, ctx.raw_rewards[i])
            self.metrics.reset_env(i)
            self.strategy.reset_env(i)
        else:
            steps = int(self.metrics.ep_steps[i])
            ally_end, enemy_end = building_hp_end(ctx.next_obs[i])
            self.metrics.finish_episode(i, ctx.raw_rewards[i], ally_end,
                                        enemy_end)
            self.strategy.finish_episode(
                i, infos["ep_killed_by_card"][i], infos["ep_spent_by_card"][i],
                infos["team0_tower_damage"][i], steps)

        if self.episodes_completed % 10 == 0 and self.metrics.outcomes:
            self._print_progress()
        self._maybe_reboost_entropy()

    def _maybe_reboost_entropy(self):
        """Exploration re-boosts. OPPONENT SELECTION is PFSP's job now -- this
        reads the GLOBAL outcome window across whatever mix of pool opponents
        the workers happened to sample."""
        outcomes = self.metrics.outcomes
        if len(outcomes) != outcomes.maxlen:
            return
        m = self.metrics.summary()
        since = self.episodes_completed - self.last_stall_reboost_episode
        if (m["draw_rate"] >= STALL_DRAW_RATE_THRESHOLD
                and since >= STALL_REBOOST_COOLDOWN_EPISODES):
            self._reboost(f"High draw rate ({m['draw_rate']:.2f}) across the "
                          "pool -- re-boosting exploration to try to break out "
                          "of a passive equilibrium.")
        elif (m["win_rate"] < 0.5 and since >= ENTROPY_STALE_REBOOST_EPISODES):
            self._reboost(f"Losing trend (win rate {m['win_rate']:.2f}) across "
                          "the pool with no draw-stall trigger in "
                          f"{ENTROPY_STALE_REBOOST_EPISODES}+ episodes -- "
                          "periodically re-boosting exploration anyway.")

    def _reboost(self, message):
        self.entropy_reboost_episode = self.episodes_completed
        self.last_stall_reboost_episode = self.episodes_completed
        print(f">>> {message}")

    # -- read-out -----------------------------------------------------------
    def _print_progress(self):
        m = self.metrics.summary()
        s = self.strategy.summary()
        sc = self.scenario_metrics.summary()
        ep = self.episodes_completed
        w = self.writer
        # ep_len_history is in bot-steps (each == skip_frames ticks), so *10
        # converts to real engine ticks.
        avg_ticks = m["avg_length"] * 10
        worst_name = (card_name(s["roi_worst_card"]).split("_")[0][:9]
                      if s["roi_worst_card"] is not None else "-")
        print(f"Episodes: {ep} | Avg(50): {m['avg_reward']:.2f} | "
              f"W/L/D: {m['win_rate']:.2f}/{m['loss_rate']:.2f}/"
              f"{m['draw_rate']:.2f} | Decisive: {m['decisive_win_rate']:.2f} | "
              f"ScenDef: {sc['defensive_rate']:.2f} | "
              f"ScenOff: {sc['other_rate']:.2f} | AvgTicks: {avg_ticks:.0f} | "
              f"Cards/Game: {s['cards_per_game']:.2f}/8 | "
              f"Plays: {s['plays_per_game']:.1f} | "
              f"Elixir@Play: {s['elixir_at_play']:.2f} | ROI: {s['roi']:.2f} | "
              f"Worst: {worst_name} {s['roi_worst']:.2f} | "
              f"Fwd: {100 * s['forward_rate']:.0f}% | "
              f"TwrDmg/1k: {s['tower_damage_rate']:.0f} | "
              f"Pool: {len(self.historical_pool)} | "
              f"EntCoef c/p: {self.entropy.coef_card:.4f}/"
              f"{self.entropy.coef_placement:.4f}")

        w.add_scalar("Economy/ROI_All_Cards", s["roi"], ep)
        w.add_scalar("Economy/ROI_Worst_Card", s["roi_worst"], ep)
        if s["roi_per_card"] is not None:
            for ci, cid in enumerate(DEFAULT_DECK):
                value = s["roi_per_card"][ci]
                if value == value:      # not NaN -- the card was played
                    w.add_scalar(f"Economy/ROI_ByCard/{card_name(cid)}",
                                 float(value), ep)
        w.add_scalar("Strategy/Forward_Placement_Rate", s["forward_rate"], ep)
        w.add_scalar("Strategy/Tower_Damage_Per_1k_Ticks",
                     s["tower_damage_rate"], ep)
        w.add_scalar("Strategy/Distinct_Cards_Per_Game", s["cards_per_game"], ep)
        w.add_scalar("Strategy/Plays_Per_Game", s["plays_per_game"], ep)
        w.add_scalar("Strategy/Elixir_At_Play", s["elixir_at_play"], ep)
        w.add_scalar("Progress/Episode_Length_Ticks_50", avg_ticks, ep)
        if sc["defensive_n"]:
            w.add_scalar("Scenario/Defense_Success_Rate", sc["defensive_rate"], ep)
            w.add_scalar("Scenario/Sample_Count", sc["defensive_n"], ep)
        if sc["other_n"]:
            w.add_scalar("Scenario/Offensive_No_Loss_Rate", sc["other_rate"], ep)
            w.add_scalar("Scenario/Offensive_Sample_Count", sc["other_n"], ep)
        w.add_scalar("Training/Avg_Reward_50", m["avg_reward"], ep)
        w.add_scalar("Reward/Episode_Shaping_Sum", m["avg_shaping"], ep)
        w.add_scalar("Training/Win_Rate_100", m["win_rate"], ep)
        w.add_scalar("Rates/Loss_100", m["loss_rate"], ep)
        w.add_scalar("Rates/Draw_100", m["draw_rate"], ep)
        w.add_scalar("Rates/Decisive_Win_100", m["decisive_win_rate"], ep)
        w.add_scalar("Training/Decided_Count_100", m["decided"], ep)
        if math.isfinite(m["decisive_win_rate_long"]):
            w.add_scalar("Rates/Decisive_Win_500",
                         m["decisive_win_rate_long"], ep)
        w.add_scalar("Progress/Episode_Length_50", m["avg_length"], ep)
        w.add_scalar("Progress/Enemy_Building_HP_End_50",
                     m["enemy_building_hp_end"], ep)
        w.add_scalar("Progress/Ally_Building_HP_End_50",
                     m["ally_building_hp_end"], ep)
        w.add_scalar("Training/PFSP_Pool_Size", len(self.historical_pool), ep)
        w.add_scalar("Training/Entropy_Coef_Card", self.entropy.coef_card, ep)
        w.add_scalar("Training/Entropy_Coef_Placement",
                     self.entropy.coef_placement, ep)

    # -- periodic league work ----------------------------------------------
    def on_after_update(self, stats):
        self._maybe_run_exploiter_burst()
        self._maybe_evaluate()

    def _maybe_run_exploiter_burst(self):
        """Train a SEPARATE agent whose only job is to beat the main agent as it
        is right now, then drop its snapshot into the same pool the main agent
        samples from. See `trainers/exploiter.py` for why the existing pool
        cannot produce this pressure on its own: every neural opponent in it is
        a past self, so self-play is free to cycle rather than improve.

        The exploiter's snapshot is deliberately eligible IMMEDIATELY -- the age
        gate only matches `_pipeline2_ep<N>` filenames, and an exploiter goes
        stale as the hole is patched, not as it ages.
        """
        if not exploiter_mod.should_run_burst(self.episodes_completed,
                                              self.last_exploiter_burst_ep):
            return
        print(f">>> Starting exploiter burst #{self.exploiter_burst_index} "
              f"at episode {self.episodes_completed}...")
        self.exploiter_state, _stats = exploiter_mod.run_exploiter_burst(
            self.net, self.device, self.episodes_completed,
            self.num_ability_slots(), exploiter_state=self.exploiter_state,
            burst_index=self.exploiter_burst_index, writer=self.writer)
        self.exploiter_burst_index += 1
        self.last_exploiter_burst_ep = self.episodes_completed
        # Refresh so the brand-new snapshot actually enters rotation; without
        # this it would sit unused until the next historical save.
        self._refresh_pool()

    def _maybe_evaluate(self):
        if (self.episodes_completed - self.last_eval_ep
                < league.EVAL_INTERVAL_EPISODES):
            return
        league.update_reference_roster(self.reference_roster,
                                       self.historical_pool)
        if self.reference_roster:
            print(f"Evaluating current policy (greedy) against "
                  f"{len(self.reference_roster)} fixed reference opponent(s), "
                  f"{league.EVAL_GAMES_PER_OPPONENT} games each...")
            agent_elo, per_opponent = league.evaluate_against_roster(
                self.net, self.device, self.reference_roster,
                n_games=league.EVAL_GAMES_PER_OPPONENT)
            self.writer.add_scalar("Eval/Elo", agent_elo,
                                   self.episodes_completed)
            for i, entry in enumerate(self.reference_roster):
                self.writer.add_scalar(
                    f"Eval/WinRate_vs_Anchor{i}",
                    per_opponent[entry["path"]]["score"],
                    self.episodes_completed)
            print(f">>> Eval @ ep {self.episodes_completed}: Elo~{agent_elo:.0f} "
                  f"(vs {len(self.reference_roster)} fixed anchor(s) -- a "
                  "comparable trend against a never-changing roster, not a "
                  "calibrated rating)")
        self.last_eval_ep = self.episodes_completed


def train_selfplay_ppo():
    """Entry point, kept under its historical name."""
    Phase2Trainer().run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_selfplay_ppo()
