"""Pipeline 1: PPO against the UtilityTeacher, then against randomized decks.

    python_ai/venv/Scripts/python.exe python_ai/trainers/train.py

Two phases, then an automatic handoff:

    "mirror"           team 1 is the teacher at a symmetric 1.0x economy,
                       playing decks from the meta pool. Difficulty is the
                       teacher's lookahead (rl.curriculum.CURRICULUM_STAGES).
      | win rate >= PHASE2_ENTRY_WIN_RATE at stage >= PHASE2_MIN_CURRICULUM_STAGE
      v
    "random_opponent"  replays the stage ladder per deck: an overfitting
                       check before the league, not where generalization is
                       learned.
      | episodes in phase >= RANDOM_OPPONENT_EPISODE_BUDGET
      v
    trainers/train_selfplay.py, launched by subprocess (pipeline 2, PFSP league)

The handoff is an episode count within the phase; PHASE2_ENTRY_WIN_RATE gates
only mirror -> random_opponent.
"""
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# First, before gymnasium/numpy/torch: importing the package caps
# OPENBLAS_NUM_THREADS, which OpenBLAS reads when it loads.
# tests/test_blas_thread_caps.py enforces the order.
import python_ai  # noqa: E402,F401

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import clash_royale_env  # noqa: E402
import torch  # noqa: E402
from python_ai.envs import gym_wrapper, scenarios  # noqa: E402
from python_ai.opponents import deck_pool  # noqa: E402
from python_ai.opponents.teacher import remap_legacy_stage  # noqa: E402
from python_ai.rewards.shaping import building_hp_end  # noqa: E402
from python_ai.rl.base_trainer import BaseTrainer  # noqa: E402
from python_ai.rl.checkpointing import (  # noqa: E402
    STAGE_CHECKPOINT_DIR, run_path, save_stage_snapshot, weights_path,
)
from python_ai.rl.config import PHASE1_ENTROPY, PPOConfig  # noqa: E402
from python_ai.rl.seeding import worker_seeds  # noqa: E402
from python_ai.rl.curriculum import (  # noqa: E402
    CURRICULUM_STAGES, PLATEAU_MIN_WIN_RATE, PLATEAU_PATIENCE_EPISODES,
    PLATEAU_WINDOW, STALL_WIN_RATE, CurriculumManager,
)

# Who plays team 1 in phase 1: "teacher" (the utility-search bot,
# opponents/teacher.py) or "builtin" (the C++ HeuristicOpponent, kept for
# comparison runs).
PHASE1_OPPONENT = os.environ.get("CLASH_PHASE1_OPPONENT", "teacher")

#: Fraction of phase-1 episodes that start in a defensive emergency, the same
#: as pipeline 2 so phase 2 is not the first time the policy meets one. Without
#: it the states that make defensive cards worth playing never occur.
PHASE1_DEFENSIVE_SCENARIO_PROB = float(os.environ.get(
    "CLASH_PHASE1_SCENARIO_PROB", scenarios.SCENARIO_INJECTION_PROB))

#: Train against the meta deck pool (opponents/decks/meta_decks.json), sampled
#: per worker per episode by PFSP weight; `CLASH_PHASE1_DECK_POOL=0` restores
#: the mirror. A mirror never asks the questions some deck cards exist to
#: answer, so the agent learns to leave them in hand; the mirror is still one
#: deck in the pool.
PHASE1_DECK_POOL = os.environ.get("CLASH_PHASE1_DECK_POOL", "1") not in (
    "0", "false", "False", "")

# Win rate that masters a random deck at the per-deck ladder's final stage
# (then it rotates).
PHASE2_WIN_RATE_GATE = 0.80

# Stage needed to leave the mirror phase: the rung the old six-rung stage 4 was
# (70 ticks of lookahead), mapped through the same remap as checkpoints, since
# a literal index would mean something else in the eleven-rung table.
PHASE2_MIN_CURRICULUM_STAGE = remap_legacy_stage(4)

# Win rate required to leave the mirror phase; separate from
# PHASE2_WIN_RATE_GATE, which decides deck mastery. Overridable only for
# controlled experiments: above 1.0 pins a run to the mirror.
PHASE2_ENTRY_WIN_RATE = float(os.environ.get("CLASH_PHASE2_ENTRY_WIN_RATE", 0.60))
# Each random deck replays the stage ladder, so rotation is performance-gated:
# a deck the agent handles clears quickly, one it cannot answer keeps getting
# training. This cap is only a safety valve for a pathological draw, sized so
# the phase budget still covers at least 4 decks.
MAX_EPISODES_PER_RANDOM_DECK = 1250
# Episodes spent in random_opponent before the handoff to pipeline 2, counted
# within the phase. Kept small: it is a cheap overfitting check before the
# expensive league, not where generalization is learned (pipeline 2's scripted
# bots also play random decks).
RANDOM_OPPONENT_EPISODE_BUDGET = 5000

def sample_random_deck():
    """The random_opponent phase's deck source: the meta pool when enabled, else
    the engine's uniform registry draw.

    The two are different distributions: a uniform draw produces decks no
    player queues and that crush a cycle deck, while pool decks are curated.
    Both respect deck-slot rules (sampleRandomDeck by construction; the pool
    refuses Champions and Heroes).
    """
    if PHASE1_DECK_POOL:
        decks = deck_pool.load_pool()
        return list(random.choice(decks).card_ids)
    return clash_royale_env.sample_random_deck()


def make_env(seed=None):
    """A phase-1 worker factory. `seed` is this worker's own (from
    `rl.seeding.worker_seeds`).
    """
    def _init():
        # The C++ heuristic remains an evaluation anchor
        # (train_selfplay.BUILTIN_ANCHORS).
        return gym_wrapper.MicroRoyaleEnv({
            "opponent": PHASE1_OPPONENT,
            "teacher_stage": 0,
            "scenario_seed": seed,
            "deck_pool": PHASE1_DECK_POOL,
            "defensive_scenario_prob": PHASE1_DEFENSIVE_SCENARIO_PROB})
    return _init


class Phase1Trainer(BaseTrainer):
    """PPO vs the teacher, with the two-phase curriculum on top."""

    #: Set by load_checkpoint and read by _apply_curriculum_to_envs, which also
    #: runs on a fresh start before they are assigned; empty means nothing to
    #: restore.
    _restored_deck_stats = {}
    _restored_deck_counts = {}

    pipeline_name = "pipeline1"
    #: Defensive scenario windows can expire here too; the critic must
    #: bootstrap through them, not treat holding a defence as worth zero.
    uses_truncation_bootstrap = True
    #: Charge DRAW_PENALTY only on real endings: an expiring scenario window
    #: with ~0 reward is what a successful defence looks like.
    draw_on_terminated_only = True
    replay_prefix = "replay"

    def __init__(self, cfg=None):
        super().__init__(cfg or PPOConfig(), PHASE1_ENTROPY)
        # Overridable together, so an experiment arm cannot clobber the live
        # checkpoint or share its TensorBoard directory. Anchored, never
        # cwd-relative.
        self.weight_path = weights_path(
            os.environ.get("CLASH_WEIGHTS", "model_weights.pth"))
        self.log_dir = run_path(
            os.environ.get("CLASH_LOGDIR", "runs/clash_royale_experiment"))
        self.curriculum = CurriculumManager(
            entry_win_rate=PHASE2_ENTRY_WIN_RATE,
            min_stage_for_phase2=PHASE2_MIN_CURRICULUM_STAGE,
            phase2_win_rate_gate=PHASE2_WIN_RATE_GATE,
            max_episodes_per_deck=MAX_EPISODES_PER_RANDOM_DECK,
            random_opponent_budget=RANDOM_OPPONENT_EPISODE_BUDGET)

    # -- environment --------------------------------------------------------
    def build_envs(self):
        return gym.vector.AsyncVectorEnv(
            [make_env(seed=s)
             for s in worker_seeds(self.cfg.seed, self.cfg.num_envs)])

    def build_replay_env(self):
        env = gym_wrapper.MicroRoyaleEnv({"opponent": PHASE1_OPPONENT,
                                          "teacher_stage": 0})
        # Match the replay env to the curriculum in progress (rung, and the
        # random deck in phase 2).
        env.set_teacher_stage(self.curriculum.teacher_stage)
        if (self.curriculum.phase == "random_opponent"
                and self.curriculum.current_random_deck is not None):
            env.set_opponent_deck(self.curriculum.current_random_deck)
        return env

    # -- resume -------------------------------------------------------------
    def load_checkpoint(self):
        if not os.path.exists(self.weight_path):
            return False
        checkpoint = torch.load(self.weight_path, map_location=self.device,
                                weights_only=False)
        # A failed restore crashes rather than starting fresh: the restore is
        # not transactional, and falling back used to move the checkpoint aside
        # and delete the TensorBoard log while carrying on with its weights.
        if not (isinstance(checkpoint, dict) and "model" in checkpoint
                and "optimizer" in checkpoint):
            # Legacy checkpoint: bare model state_dict, no training state.
            from python_ai.models.policy_io import load_state_dict_flexible
            load_state_dict_flexible(
                self.net, checkpoint,
                f"pipeline1 legacy resume ({self.weight_path})")
            print(f"Loaded legacy weights-only checkpoint from "
                  f"{self.weight_path} (training state starts fresh).")
            return False
        self.restore_common(checkpoint)
        self.curriculum.load_state_dict(checkpoint)
        self._restored_deck_stats = checkpoint.get("deck_pool_stats") or {}
        self._restored_deck_counts = checkpoint.get("deck_pool_counts") or {}
        self._apply_curriculum_to_envs()
        print(f"Resumed from {self.weight_path}: "
              f"episode {self.episodes_completed}, "
              f"curriculum stage {self.curriculum.stage}, "
              f"phase {self.curriculum.phase}")
        return True

    def _apply_curriculum_to_envs(self):
        """Push the restored curriculum state into the workers.

        `CurriculumManager.teacher_stage` resolves which ladder applies (phase
        1's, or the current random deck's).
        """
        self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        if (self.curriculum.phase == "random_opponent"
                and self.curriculum.current_random_deck is not None):
            self.envs.call("set_opponent_deck",
                           self.curriculum.current_random_deck)
        # Seed every worker with the pooled deck estimates, which beat any
        # single worker's.
        if self._restored_deck_stats:
            self.envs.call("set_deck_pool_stats", self._restored_deck_stats,
                           self._restored_deck_counts)
            worst = min(self._restored_deck_stats.items(), key=lambda kv: kv[1])
            print(f">>> Restored PFSP deck estimates for "
                  f"{len(self._restored_deck_stats)} decks "
                  f"(worst: {worst[0]} {worst[1]:.3f}) -- without this the pool "
                  f"resets to meta_decks.json priors and the win rate dips for "
                  f"~500 episodes while it re-learns.")

    def _merged_deck_pool(self):
        """(rates, counts) pooled across workers, or (None, None).

        Rates are averaged (each worker's EWMA covers its own episodes). Counts
        only set alpha = max(0.05, 1/(n+1)), which floors at n = 19, so
        averaging them is fine.
        """
        if not PHASE1_DECK_POOL or self.envs is None:
            return None, None
        try:
            rates = self.envs.call("get_deck_pool_stats")
            counts = self.envs.call("get_deck_pool_counts")
        except Exception:
            return None, None       # a worker that cannot answer must not
                                    # break checkpointing
        def _avg(dicts, cast):
            merged = {}
            for d in dicts:
                for k, v in (d or {}).items():
                    merged.setdefault(k, []).append(v)
            return {k: cast(sum(v) / len(v)) for k, v in merged.items()} or None
        return _avg(rates, float), _avg(counts, int)

    def checkpoint_payload(self):
        payload = self.curriculum.state_dict()
        # The PFSP deck estimates, so a restart does not reset them to the
        # priors; see MicroRoyaleEnv.set_deck_pool_stats.
        rates, counts = self._merged_deck_pool()
        if rates:
            payload["deck_pool_stats"] = rates
            payload["deck_pool_counts"] = counts or {}
        return payload

    # -- the phase machine --------------------------------------------------
    def should_stop(self):
        # The random-deck phase has no natural end; its budget triggers the
        # handoff to train_selfplay.py.
        return (self.episodes_completed >= 1_000_000
                or self.curriculum.budget_exhausted(self.episodes_completed))

    def on_episode_end(self, i, ctx):
        # A scenario episode is not a match result: it is a short window that
        # rarely ends in a crown, and counting it would cap the achievable win
        # rate below the gate
        # (tests/test_phase1_scenarios_do_not_break_the_gate.py).
        is_scenario = ctx.infos.get(
            "is_scenario", np.zeros(self.cfg.num_envs, dtype=np.float32))[i] > 0.5
        # Before the scenario return, or scenario episodes would drop the deck
        # read-out and the plateau valve's progress signal it feeds. It
        # self-gates on the episode count.
        self._print_deck_pool()
        if is_scenario:
            self.metrics.reset_env(i)
            return

        ally_end, enemy_end = building_hp_end(ctx.next_obs[i])
        self.metrics.finish_episode(i, ctx.raw_rewards[i], ally_end, enemy_end)
        # The plateau detector's own series, fed from the same episodes (and
        # scenario exclusion) as the gate window.
        if self.metrics.outcomes:
            self.curriculum.note_outcome(self.metrics.outcomes[-1])
        if self.episodes_completed % 10 == 0:
            self._print_progress()

        # Order is load-bearing: the phase gate first, because the stage gate
        # clears the outcome window when it fires.
        outcomes = self.metrics.outcomes
        entered = self.curriculum.maybe_enter_random_phase(
            outcomes, self.episodes_completed)
        if entered is not None:
            self._on_entered_random_phase(entered)
            return

        advanced = self.curriculum.maybe_advance_stage(
            outcomes, self.episodes_completed)
        if advanced is not None:
            self._on_stage_advanced(*advanced)
            return

        # Demotion; exclusive with the advance gate by construction, so its
        # order is free.
        demoted = self.curriculum.maybe_demote_stage(
            outcomes, self.episodes_completed)
        if demoted is not None:
            self._on_stage_demoted(demoted)
            return

        # Rung 0 has no valve of its own; see curriculum.FLOOR_ALARM_*.
        alarm = self.curriculum.floor_alarm(outcomes, self.episodes_completed)
        if alarm is not None:
            print(f">>> [FLOOR] {alarm}")
            self.writer.add_scalar("Training/Curriculum_FloorAlarm", 1.0,
                                   self.episodes_completed)

        event = self.curriculum.step_random_deck_curriculum(
            outcomes, self.episodes_completed)
        if event is None:
            return
        kind, payload = event
        if kind == "rotate":
            self._rotate_random_deck(payload)
        else:
            self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
            print(f">>> Deck curriculum advanced to stage {payload} "
                  f"(teacher_stage={self.curriculum.teacher_stage}) against "
                  "current random deck")

    def _on_entered_random_phase(self, win_rate):
        c = self.curriculum
        # The end of the mirror phase, captured before the opponents change.
        save_stage_snapshot(
            self.net, STAGE_CHECKPOINT_DIR, c.stage, self.episodes_completed,
            CURRICULUM_STAGES[c.stage]["teacher_stage"],
            f"entered phase 2 at win_rate={win_rate:.2f} (end of mirror phase)")
        # With the pool on, the deck is drawn per episode in reset(); a deck
        # set here would be overwritten at once.
        if not PHASE1_DECK_POOL:
            c.current_random_deck = sample_random_deck()
            self.envs.call("set_opponent_deck", c.current_random_deck)
        self.envs.call("set_teacher_stage", c.teacher_stage)
        deck_note = ("pool sampling continues" if PHASE1_DECK_POOL
                     else f"deck={c.current_random_deck}")
        print(f">>> Phase advanced to random_opponent from stage {c.stage} "
              f"({deck_note}) - mirror win rate "
              f"{win_rate:.2f} reached the {PHASE2_ENTRY_WIN_RATE} entry "
              "threshold")
        print(f">>> Random-deck budget: {RANDOM_OPPONENT_EPISODE_BUDGET} "
              f"episodes (handoff to train_selfplay.py at episode "
              f"{self.episodes_completed + RANDOM_OPPONENT_EPISODE_BUDGET})")
        self.writer.add_scalar("Training/Phase", 1, self.episodes_completed)

    def _on_stage_advanced(self, new_stage, reason="gate"):
        """One rung up, by mastery ("gate") or convergence ("plateau").

        The reason is logged: a rung reached by plateau only says the rung
        stopped teaching.
        """
        cleared = new_stage - 1
        save_stage_snapshot(
            self.net, STAGE_CHECKPOINT_DIR, cleared, self.episodes_completed,
            CURRICULUM_STAGES[cleared]["teacher_stage"],
            f"cleared stage {cleared} by {reason}")
        self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        if reason == "plateau":
            print(f">>> [PLATEAU] Curriculum advanced to stage {new_stage} "
                  f"(teacher_stage={self.curriculum.teacher_stage}): the "
                  f"{PLATEAU_WINDOW}-episode win rate stopped improving for "
                  f"{PLATEAU_PATIENCE_EPISODES} episodes while staying above "
                  f"{PLATEAU_MIN_WIN_RATE:.0%}. Plateau advances this run: "
                  f"{self.curriculum.plateau_advances}.")
        else:
            print(f">>> Curriculum advanced to stage {new_stage} "
                  f"(teacher_stage={self.curriculum.teacher_stage})")
        self.writer.add_scalar("Training/Curriculum_Stage", new_stage,
                               self.episodes_completed)
        self.writer.add_scalar("Training/Curriculum_PlateauAdvances",
                               self.curriculum.plateau_advances,
                               self.episodes_completed)

    def _on_stage_demoted(self, new_stage):
        """Demotion: the ladder position was ahead of the policy's competence.
        Logged loudly.
        """
        self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        reason = (getattr(self.curriculum, "last_demotion_reason", "")
                  or f"sustained win rate at or below {STALL_WIN_RATE:.0%}")
        print(f">>> [DEMOTED] Curriculum DEMOTED to stage {new_stage} "
              f"(teacher_stage={self.curriculum.teacher_stage}): {reason}. "
              f"Demotions this run: {self.curriculum.demotions}.")
        self.writer.add_scalar("Training/Curriculum_Stage", new_stage,
                               self.episodes_completed)
        self.writer.add_scalar("Training/Curriculum_Demotions",
                               self.curriculum.demotions,
                               self.episodes_completed)

    def _rotate_random_deck(self, reason):
        c = self.curriculum
        # With the pool on there is nothing to rotate; see
        # _on_entered_random_phase.
        if PHASE1_DECK_POOL:
            self.envs.call("set_teacher_stage", c.teacher_stage)
            print(f">>> Deck ladder reset ({reason}); the meta pool keeps "
                  "sampling per episode")
            return
        c.current_random_deck = sample_random_deck()
        self.envs.call("set_opponent_deck", c.current_random_deck)
        self.envs.call("set_teacher_stage", c.teacher_stage)
        print(f">>> New random opponent deck: {c.current_random_deck} "
              f"(previous deck {reason})")

    # -- read-out --
    DECK_READOUT_EVERY = 200

    def _print_deck_pool(self):
        """Per-deck win rates, averaged over the workers' local estimates.

        Aggregates cannot see a single deck being lost; this read-out can.
        Averaged, not summed: each worker's EWMA covers its own episodes.
        """
        if not PHASE1_DECK_POOL or self.episodes_completed % self.DECK_READOUT_EVERY:
            return
        try:
            per_worker = self.envs.call("get_deck_pool_stats")
        except Exception:
            return          # a worker that cannot answer must not kill a run
        merged = {}
        for stats in per_worker:
            for name, rate in (stats or {}).items():
                merged.setdefault(name, []).append(rate)
        if not merged:
            return
        avg = {n: sum(v) / len(v) for n, v in merged.items()}
        ordered = sorted(avg.items(), key=lambda kv: kv[1])
        cells = "  ".join(f"{n[:14]} {r:.2f}" for n, r in ordered)
        print(f"     Decks(win): {cells}")
        for name, rate in avg.items():
            self.writer.add_scalar(f"Decks/WinRate/{name}", rate,
                                   self.episodes_completed)
        # Watch the minimum: a pool average stays healthy while one matchup is
        # lost outright.
        self.writer.add_scalar("Decks/WinRate_Min", min(avg.values()),
                               self.episodes_completed)
        self.writer.add_scalar("Decks/WinRate_Spread",
                               max(avg.values()) - min(avg.values()),
                               self.episodes_completed)
        # The plateau valve's trend signal: the unweighted per-deck mean. PFSP
        # pins the readable win rate near the hardest matchups however much the
        # agent improves; re-weighting here would bring that blindness back.
        self.curriculum.note_progress(sum(avg.values()) / len(avg))
        self.writer.add_scalar("Decks/WinRate_UnweightedMean",
                               sum(avg.values()) / len(avg),
                               self.episodes_completed)

    def _print_progress(self):
        m = self.metrics.summary()
        c = self.curriculum
        ep = self.episodes_completed
        deck_stage_str = (f"/{c.deck_stage}"
                          if c.phase == "random_opponent" else "")
        print(f"Episodes: {ep} | Avg(50): {m['avg_reward']:.2f} | "
              f"W/L/D: {m['win_rate']:.2f}/{m['loss_rate']:.2f}/"
              f"{m['draw_rate']:.2f} | Decisive: {m['decisive_win_rate']:.2f} | "
              f"Stage: {c.stage}{deck_stage_str} | Phase: {c.phase} | "
              f"EntCoef c/p: {self.entropy.coef_card:.4f}/"
              f"{self.entropy.coef_placement:.4f}")
        # The ladder's provenance counters on this cadence, so a ladder climbed
        # by plateau alone shows as a curve.
        self.writer.add_scalar("Training/Curriculum_PlateauAdvances",
                               self.curriculum.plateau_advances,
                               self.episodes_completed)
        self.writer.add_scalar("Training/Curriculum_Demotions",
                               self.curriculum.demotions, self.episodes_completed)
        w = self.writer
        w.add_scalar("Training/Avg_Reward_50", m["avg_reward"], ep)
        w.add_scalar("Reward/Episode_Shaping_Sum", m["avg_shaping"], ep)
        w.add_scalar("Training/Win_Rate_100", m["win_rate"], ep)
        w.add_scalar("Training/Phase", 0 if c.phase == "mirror" else 1, ep)
        # A rising win share should come out of the loss share (stronger) or
        # the draw share (closing games).
        w.add_scalar("Rates/Loss_100", m["loss_rate"], ep)
        w.add_scalar("Rates/Draw_100", m["draw_rate"], ep)
        w.add_scalar("Rates/Decisive_Win_100", m["decisive_win_rate"], ep)
        # How many of the last 100 games were decided.
        w.add_scalar("Training/Decided_Count_100", m["decided"], ep)
        if math.isfinite(m["decisive_win_rate_long"]):
            w.add_scalar("Rates/Decisive_Win_500",
                         m["decisive_win_rate_long"], ep)
        # Closing games shows as shorter episodes and less enemy building HP
        # left.
        w.add_scalar("Progress/Episode_Length_50", m["avg_length"], ep)
        w.add_scalar("Progress/Enemy_Building_HP_End_50",
                     m["enemy_building_hp_end"], ep)
        w.add_scalar("Progress/Ally_Building_HP_End_50",
                     m["ally_building_hp_end"], ep)
        w.add_scalar("Training/Curriculum_Stage", c.stage, ep)
        w.add_scalar("Training/Entropy_Coef_Card", self.entropy.coef_card, ep)
        w.add_scalar("Training/Entropy_Coef_Placement",
                     self.entropy.coef_placement, ep)

    # -- shutdown -----------------------------------------------------------
    def on_finish(self):
        # A fresh save at the stop point: pipeline 2 bootstraps from this file.
        self.save_checkpoint(verbose=False)
        self.envs.close()
        print(f">>> Pipeline #1 stopped at episode {self.episodes_completed} "
              f"(phase={self.curriculum.phase}) -- final checkpoint saved to "
              f"{self.weight_path}.")
        launch_pipeline2(log_dir=os.path.dirname(os.path.abspath(self.log_dir)))


def launch_pipeline2(log_dir=None, wait_seconds=60.0):
    """Start train_selfplay.py, confirm it survives startup, and return.

    `-u` so the redirected stdout is not block-buffered. The child is watched
    for `wait_seconds`, and a non-zero exit inside that window raises, so phase
    1 exits non-zero too instead of ending the run silently. Logs are appended
    next to the run's TensorBoard directories.
    """
    import time as _time
    log_dir = log_dir or run_path("runs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "training_selfplay_pfsp.log")
    err_path = os.path.join(log_dir, "training_selfplay_pfsp_err.log")
    out = open(out_path, "a")
    err = open(err_path, "a")
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "train_selfplay.py")
    proc = subprocess.Popen([sys.executable, "-u", target], stdout=out, stderr=err)
    deadline = _time.monotonic() + float(wait_seconds)
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                raise RuntimeError(
                    f"pipeline #2 exited with code {rc} during startup -- see "
                    f"{err_path}")
            break
        if _time.monotonic() >= deadline:
            break
        _time.sleep(min(1.0, float(wait_seconds)))
    print(f">>> Launched train_selfplay.py (pipeline #2) -- see {out_path} / {err_path}")


def train_ppo():
    """Entry point."""
    Phase1Trainer().run()


if __name__ == "__main__":
    # Needed for Windows multiprocessing.
    import multiprocessing
    multiprocessing.freeze_support()
    train_ppo()
