"""Pipeline 2: the PFSP league.

    python_ai/venv/Scripts/python.exe python_ai/trainers/train_selfplay.py

The trainee plays frozen snapshots of its own past selves, four scripted bots
and the C++ heuristic, sampled by prioritized fictitious self-play; both sides
decide their own plays (`ClashEnv::stepSelfPlay`).

This file owns the league (pool discovery, the fixed Elo roster, the exploiter
burst), scenario-aware episode bookkeeping, the strategy read-out (Cards/Game,
Elixir@Play, ROI, Fwd, TwrDmg/1k) and the two entropy re-boost heuristics. The
loop is `rl/base_trainer.py`.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# First, before gymnasium/numpy/torch: see the same note in trainers/train.py.
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

# Re-boost exploration when draws exceed this share of the last 100 episodes:
# once both sides sit in a passive equilibrium with floored entropy, gradient
# descent has no exploration left to climb out.
STALL_DRAW_RATE_THRESHOLD = 0.5

# Give each boost time to take effect before another.
STALL_REBOOST_COOLDOWN_EPISODES = 1000

# Re-boost also after this long without one while losing (win rate < 0.5). Not
# while winning: a re-boost then measurably hurt.
ENTROPY_STALE_REBOOST_EPISODES = 1500

def selfplay_paths():
    """(selfplay checkpoint, phase-1 bootstrap, TensorBoard dir) for this run.

    Derived from CLASH_WEIGHTS / CLASH_LOGDIR, which the child inherits from
    phase 1, so a redirected run hands off to its own files. Without
    redirection these are the defaults.
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


# Pipeline 1's checkpoint seeds a from-scratch pipeline 2 (bare weights);
# pipeline 2 keeps its own checkpoint and never overwrites pipeline 1's.
WEIGHT_PATH, BOOTSTRAP_FROM_PATH, SELFPLAY_LOG_DIR = selfplay_paths()


class Phase2Trainer(BaseTrainer):
    """PPO inside the PFSP league."""

    pipeline_name = "pipeline2"
    replay_prefix = "selfplay_replay"
    weight_path = WEIGHT_PATH
    log_dir = SELFPLAY_LOG_DIR
    #: A scenario window can end an episode without a king dying: bootstrap
    #: V(final_obs).
    uses_truncation_bootstrap = True
    #: DRAW_PENALTY on real endings only: a successful defence that ran out its
    #: window must not be punished as a stalled game.
    draw_on_terminated_only = True

    def __init__(self, cfg=None):
        super().__init__(cfg or PPOConfig(), PHASE2_ENTROPY)
        self.historical_pool = []
        self.reference_roster = []
        # Episode at which the entropy anneal clock last reset; only the
        # re-boosts move it.
        self.entropy_reboost_episode = 0
        self.last_stall_reboost_episode = 0
        self.last_eval_ep = 0
        # The exploiter's schedule survives a restart; its weights do not. A
        # re-seed from the current agent starts it as an exact copy, which is
        # what makes 0.50 its null.
        self.exploiter_state = None
        self.exploiter_burst_index = 0
        self.last_exploiter_burst_ep = None
        self.strategy = None
        self.scenario_metrics = ScenarioMetrics()

    # -- environment --------------------------------------------------------
    def build_envs(self):
        # Per-worker seeds from the run seed, independent and reproducible;
        # None when unseeded.
        seeds = worker_seeds(self.cfg.seed, self.cfg.num_envs)
        return gym.vector.AsyncVectorEnv(
            [selfplay_env.make_env(seed=s) for s in seeds])

    def build_replay_env(self):
        if not self.historical_pool:
            return None
        # Scenarios off: a demo replay shows a normal game.
        env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
        # The newest pool entry, for a representative demo.
        env.set_historical_opponent(self.historical_pool[-1])
        return env

    def setup(self):
        super().setup()
        self.strategy = StrategyMetrics(self.cfg.num_envs, DEFAULT_DECK)
        self._refresh_pool(initial=True)

    # -- pool ---------------------------------------------------------------
    def _pool_broadcast(self):
        """The training pool: historical snapshots plus the permanent scripted and
        builtin members, kept out of `historical_pool` because that also feeds
        the evaluation roster, which torch.loads every entry.
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
        # New snapshots join the pool; refresh so PFSP and the evaluation
        # roster see them.
        self._refresh_pool()

    # -- resume -------------------------------------------------------------
    def anneal_episodes_done(self):
        # The anneal clock since the last re-boost, which is what lets a
        # re-boost work.
        return self.episodes_completed - self.entropy_reboost_episode

    def load_checkpoint(self):
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path, map_location=self.device,
                                    weights_only=False)
            clean = self.restore_common(checkpoint)
            if not clean:
                # Part of the net was just reinitialized: re-boost so it
                # actually explores.
                self.entropy_reboost_episode = self.episodes_completed
                print(">>> Architecture changed on resume -- forcing a fresh "
                      f"entropy re-boost from episode {self.episodes_completed} "
                      "so the reinitialized part actually gets explored.")
            else:
                # Falls back to the old ladder-era field, then the episode
                # count, for older checkpoints.
                self.entropy_reboost_episode = checkpoint.get(
                    "entropy_reboost_episode",
                    checkpoint.get("stage_start_episode",
                                   self.episodes_completed))
            self.last_stall_reboost_episode = checkpoint.get(
                "last_stall_reboost_episode", self.episodes_completed)
            self.last_eval_ep = checkpoint.get("last_eval_ep",
                                               self.episodes_completed)
            self.reference_roster = checkpoint.get("reference_roster", [])
            # Falls back to the episode count, not None, which would read as
            # "burst due now" on every resume of a legacy checkpoint.
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
            # Same lineage as the phase 1 that produced these weights: its
            # snapshots are this run's pool.
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

        Count-weighted; a worker that never played an opponent holds only the
        prior and contributes nothing. Unplayed keys are dropped.
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
        """Push a resumed checkpoint's pooled estimates into every worker, each
        with an equal share of the pooled count. Absent keys mean an older
        checkpoint.
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
        # The per-opponent PFSP estimates, so a resume does not reset them.
        stats, counts = self._merged_pfsp()
        if stats:
            payload["pfsp_stats"] = stats
            payload["pfsp_counts"] = counts or {}
        return payload

    # -- per-step diagnostics ----------------------------------------------
    def on_step(self, ctx):
        """Record which envs actually got a card down this step.

        The engine silently refuses illegal or unaffordable plays, so only a
        rise in cumulative elixir_spent proves a play. `prev_dones` masks the
        phantom post-autoreset step.
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
            # Scenario episodes stay out of the matchup histories and the
            # headline W/L/D.
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
        """Exploration re-boosts, read from the global outcome window across the
        opponents the workers sampled.
        """
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
        # Bot-steps to engine ticks (skip_frames = 10).
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
                if value == value:      # not NaN: the card was played
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
        """Train a separate agent to beat the current main agent, then add its
        snapshot to the pool (see trainers/exploiter.py).

        Every other neural opponent is a past self, so self-play can cycle
        rather than improve. The snapshot is eligible at once: the age gate
        only matches `_pipeline2_ep<N>` names, and an exploiter goes stale as
        its hole is patched, not with age.
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
        # Refresh so the new snapshot enters rotation now.
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
    """Entry point."""
    Phase2Trainer().run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_selfplay_ppo()
