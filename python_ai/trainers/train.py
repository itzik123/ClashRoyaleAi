"""Pipeline 1: PPO against the UtilityTeacher, then against randomized decks.

    python_ai/venv/Scripts/python.exe python_ai/trainers/train.py

Two phases inside this one file, then an automatic handoff:

    "mirror"           team 1 plays OUR deck, driven by opponents.teacher at a
                       symmetric 1.0x economy. Difficulty is the teacher's
                       lookahead, climbing rl.curriculum.CURRICULUM_STAGES.
      | win rate >= PHASE2_ENTRY_WIN_RATE at stage >= PHASE2_MIN_CURRICULUM_STAGE
      v
    "random_opponent"  team 1 plays a rotating random deck, replaying the same
                       stage ladder per deck. An overfitting pre-flight check,
                       not where generalization is learned.
      | episodes in phase >= RANDOM_OPPONENT_EPISODE_BUDGET
      v
    trainers/train_selfplay.py, launched by subprocess (pipeline 2, PFSP league)

THE HANDOFF IS A PLAIN EPISODE COUNT, not a win-rate gate, despite
`PHASE2_ENTRY_WIN_RATE`'s name -- that constant gates mirror -> random_opponent
only. An earlier version of this diagram put the 0.60 gate on the handoff arrow
and cost a live run two wrong predictions about when it would transition.

Everything that is not pipeline-1 policy lives elsewhere now: the PPO algorithm
in `rl/`, the reward terms in `rewards/`, the curriculum state machine in
`rl/curriculum.py`. What remains here is the opponent, the phase machine, and
the console read-out.
"""
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# FIRST, and before gymnasium/numpy/torch below. Importing this package caps
# OPENBLAS_NUM_THREADS, and OpenBLAS reads that when it LOADS -- so an import
# that pulls numpy ahead of this line makes the cap a silent no-op worth
# ~353 MB of private commit per process (measured; see python_ai/__init__.py).
# tests/test_blas_thread_caps.py fails if this order is ever reversed.
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

# Who plays team 1 in phase 1. "teacher" is the utility-search bot at a
# symmetric 1.0x economy (opponents/teacher.py); "builtin" is the old C++
# HeuristicOpponent, kept switchable ONLY so a comparison run against the
# historical setup is one env var away.
#
# The default changed on 2026-08-19 and it is GAMEPLAY-AFFECTING: every win rate
# earned against heuristic@{1.0..1.5}x is historical and is not comparable to
# anything produced after this. Checkpoints are NOT invalidated -- the
# observation, action space and architecture are untouched.
PHASE1_OPPONENT = os.environ.get("CLASH_PHASE1_OPPONENT", "teacher")

#: Fraction of phase-1 episodes that START in a defensive emergency.
#: Pipeline 2 has done this since 2026-08-09; phase 1 did not, and the
#: 2026-08-28 run is the measurement of what that costs -- 32,680
#: episodes in which the agent never met a "defend or lose the tower"
#: moment, ending with Cannon / The Log / Fireball at P(play | in hand)
#: of 0.0053 / 0.0091 / 0.0011. A well-placed Cannon prevents a full
#: Princess Tower (2,536 HP) in a real threat state, so those cards are
#: not weak -- the states that make them worth playing were absent.
#:
#: Set to scenarios.SCENARIO_INJECTION_PROB so both pipelines present
#: the same threat distribution and phase 2 is not the first time the
#: policy sees one. GAMEPLAY-AFFECTING: it changes the start-state
#: distribution, so curriculum win-rate gates are calibrated against a
#: different episode mix and win rates are not comparable across it.
PHASE1_DEFENSIVE_SCENARIO_PROB = float(os.environ.get(
    "CLASH_PHASE1_SCENARIO_PROB", scenarios.SCENARIO_INJECTION_PROB))

#: THE OPPONENT'S DECK, and the single biggest change of 2026-09-03. `True` puts
#: every enabled deck of `opponents/decks/meta_decks.json` in rotation, sampled
#: per worker per episode by PFSP weight; `CLASH_PHASE1_DECK_POOL=0` restores
#: the mirror, which is what every run before this date played.
#:
#: WHY THE MIRROR HAD TO GO, and it is not a diversity argument. Three of
#: DEFAULT_DECK's eight cards sat at P(play | in hand) <= 0.009 for 30,000
#: consecutive episodes, and the 2026-08-29 autopsy proved the card head was
#: RIGHT: forced through `env.step`, a Cannon at the policy's own cell was worth
#: +185 tower HP over the policy's own action, better in 6 of 14 states. Every
#: attempt to overrule that -- a coverage floor at two coefficients, a threat
#: gate, forced sampling at five doses -- cost win rate.
#:
#: The card was not underpriced. It was correctly priced FOR THE MIRROR. Cannon
#: answers a tank walking at your tower, Fireball answers a medium-HP cluster,
#: The Log answers a ground swarm, and a 2.6 mirror produces one Hog, one
#: Musketeer and 1-elixir Skeletons respectively. The deck was dead because the
#: opponent never asked the questions those three cards answer.
#:
#: The mirror is still IN the pool (`hog_26_mirror`), so nothing is lost; it is
#: now one matchup of sixteen instead of all of them.
#:
#: GAMEPLAY-AFFECTING in the way that matters most: it changes the opponent
#: distribution, so every curriculum gate is calibrated against a different
#: opponent and NO win rate is comparable across this date. Checkpoints still
#: load -- the observation and the action space are untouched.
PHASE1_DECK_POOL = os.environ.get("CLASH_PHASE1_DECK_POOL", "1") not in (
    "0", "false", "False", "")

# --- Phase 2: once the agent is consistently strong against the mirror-
# deck opponent at the final curriculum stage, switch to randomized
# opponent decks -- pipeline #1 (this file) is "beat a random-but-fixed-
# deck opponent"; the eventual goal is a bot that beats a real player, and
# this phase is what actually exposes it to card interactions it's never
# seen (it only ever played its own 8 cards against themselves up to this
# point), and doubles as an overfitting check: if mirror-deck performance
# doesn't transfer at all, that's a sign the policy memorized this one
# matchup rather than learning transferable play.
# Raw win rate (not decisive), matching the CURRICULUM_STAGES gate above.
# Also doubles as "deck mastered" gate at the per-deck curriculum's final
# stage, below.
# Lowered 0.90 -> 0.80 to match the per-stage gate every other transition
# uses. 0.90 was measurably a dead end: the final stage's opponent gets 1.5x
# elixir, and the strongest policy produced so far tops out around 0.79 raw
# win rate there after ~76k episodes of dedicated stage-5 training. So the
# bot would have sat at stage 5 indefinitely and NEVER reached phase 2
# (random opponent decks) or, past it, pipeline #2 self-play -- the only two
# sources of genuinely harder opposition left, and the only path to skill
# that transfers to a human.
#
# Using the same 0.80 the stage gates use is also the internally consistent
# choice: there is no principled reason the phase transition should demand a
# strictly higher bar than the stage transitions leading up to it.
PHASE2_WIN_RATE_GATE = 0.80

# Which curriculum stage is enough to leave the mirror phase. Was implicitly
# the FINAL stage (5, opponent at 1.5x elixir); now 4.
#
# The reasoning is about what each remaining obstacle actually teaches. A
# 1.5x-elixir opponent is a resource handicap that does not exist in the real
# game -- clearing it trains "survive being out-resourced", not "play Clash
# Royale". Random opponent decks and, past them, pipeline #2 self-play are
# the things that teach transferable skill, and the stage-5 requirement was
# holding the bot behind the obstacle that teaches least.
#
# It was also close to unreachable in practice: two 0.80 windows were needed
# (stage 4 -> 5, then stage 5 -> phase 2), the second against a HARDER
# opponent, while measurement put the strongest policy so far at ~0.79 peak
# at stage 5 after ~76k episodes. Run I sat at stage 4 for 13k episodes with
# Win_Rate_100 climbing 0.555 -> 0.624 (max 0.73) -- real progress, but on a
# trajectory that would spend many more hours to clear a gate whose reward
# is a harder version of an artificial handicap.
#
# RENUMBERED 2026-09-03, and this is a trap the eleven-rung table sets for
# every constant that names a stage by INDEX. The literal 4 meant "70 ticks of
# lookahead" against the six-rung table and means "30 ticks" against this one,
# so leaving it alone would have quietly moved phase-2 entry two rungs EARLIER
# while looking like no change at all. Derived through the same remap the
# checkpoint loader uses, so the intent -- the rung the old stage 4 was -- is
# stated once and cannot drift from it.
PHASE2_MIN_CURRICULUM_STAGE = remap_legacy_stage(4)

# Win rate required to LEAVE the mirror phase. Split out from
# PHASE2_WIN_RATE_GATE, which one constant was doing two unrelated jobs for:
# this decides "stop training against a resource-handicapped clone", while
# PHASE2_WIN_RATE_GATE also decides "this random deck is mastered, rotate to
# the next one" inside phase 2. Lowering one should not silently change the
# other, and it did.
#
# Set to 0.60 deliberately, which the policy already clears (measured
# 0.62-0.69 at stage 4), so the transition happens on the next full window
# rather than after hours of grinding. That is not a lowered standard, it is
# the recognition that this gate was never measuring the right thing:
# beating a bot that gets 1.4x elixir and plays random cards at random
# positions is not a prerequisite for learning from varied decks, it is a
# different and less useful skill. The real tests come after it.
# Overridable from the environment for CONTROLLED EXPERIMENTS only. A
# mirror -> random_opponent flip swaps the opponent's whole deck mid-run,
# which silently changes the task underneath any A/B that is measuring
# something else. Setting CLASH_PHASE2_ENTRY_WIN_RATE above 1.0 makes the
# gate unreachable and pins the run to the mirror deck for its duration.
# Not a training knob -- leave it unset for real runs.
PHASE2_ENTRY_WIN_RATE = float(os.environ.get("CLASH_PHASE2_ENTRY_WIN_RATE", 0.60))
# Phase 2 replays the SAME CURRICULUM_STAGES gated progression (win rate
# threshold -> escalate elixir multiplier) against each random deck, from
# stage 0 (1.0x, normal speed), instead of a flat elixir speed for a flat
# episode count. This makes rotation performance-gated rather than
# timer-gated: a deck the agent already handles well clears every stage
# (each needs just one 100-episode window at/above threshold) and rotates
# out quickly; a deck it has no answer for stalls at whatever stage it's
# failing, and keeps accumulating real training time there for as long as
# it takes -- training effort lands on whatever the agent still can't
# handle, instead of being capped by an arbitrary count regardless of
# difficulty. Confirmed via generalization.log that a flat count (the
# previous design) wasn't giving the agent a real chance to adapt per deck
# at all (see the entropy-floor finding from that analysis).
#
# MAX_EPISODES_PER_RANDOM_DECK is a safety valve, not the intended
# rotation trigger: some random 8-card draws may be pathological (no real
# win condition, or a genuinely overwhelming one) and could otherwise
# stall forever. This only cuts in for the rare deck that isn't converging.
#
# Lowered 5000 -> 1250 on 2026-08-09, when the phase's total budget became
# RANDOM_OPPONENT_EPISODE_BUDGET = 5000 (see below). At the old value the
# safety valve EQUALLED the whole phase budget, so a single pathological
# first draw could consume all of it and the agent would leave phase 1
# having faced exactly ONE random deck -- which is the precise opposite of
# what the phase is for. 1250 guarantees at least 4 distinct decks even in
# the worst case. It rarely binds: rotation is normally performance-gated,
# and mastering a deck takes ~600 episodes (6 stages x one 100-episode
# window each), so a comfortable budget still turns over ~8 decks.
MAX_EPISODES_PER_RANDOM_DECK = 1250
# Unlike phase 1 (which naturally terminates via the stage-5 + PHASE2_
# WIN_RATE_GATE transition into phase 2), phase 2 itself has no completion
# condition of its own -- it just keeps rotating random decks forever
# (mastered or timed out, on to the next one), since generalization is a
# continuous process with no natural "done" point the way a single fixed
# curriculum is. User decision: once total episodes_completed reaches this
# many (while in phase 2), that's judged as enough random-opponent
# exposure to hand off to pipeline #2 (self-play/PFSP) -- at that point
# training stops itself and automatically launches train_selfplay.py, so
# this doesn't depend on anyone watching for the right moment.
# Lowered 150000 -> 40000. Phase 2's job here is narrow: confirm the policy
# is not overfitted to the mirror matchup. That question is already answered
# -- the greedy policy took 95% against randomly drawn opponent decks on
# FIRST contact, before any phase-2 training, and 97.5%/93.8% against the
# scripted opponents with randomized decks in earlier runs. There is no
# overfitting to grind out.
#
# Spending the remaining ~113k episodes here would buy very little: random
# decks vary WHAT the opponent plays but it still plays randomly at random
# positions, so it cannot punish the degenerate no-win-condition strategy
# the policy has settled into. Pipeline #2 self-play can, because there the
# opponent is a frozen copy of the bot itself and actually defends.
#
# ---- 2026-08-09: replaced by a BUDGET measured inside the phase ----------
# A cap on TOTAL episodes was the wrong quantity, and it was arbitrary in a
# way that mattered: how long the agent spends against random decks depended
# entirely on how fast it cleared the mirror curriculum. Clear the stages in
# 3k episodes and you get 37k of random decks; take 35k and you get 5k. The
# phase's value has nothing to do with either number.
#
# Kept rather than deleted, at a deliberately small budget, because the
# measured case for deleting it does not currently hold:
#
#  * The "95% on first contact" evidence above predates BOTH the 2026-08-07
#    movement-speed fix and the 2026-08-09 placement-head fix. CLAUDE.md is
#    explicit that no win rate from before the speed fix survives it, so the
#    overfitting question is once again unanswered.
#  * This is the only overfitting check that runs BEFORE the handoff. The
#    league costs 30+ hours; finding out there that the policy memorized
#    DEFAULT_DECK is the expensive way to learn it.
#  * It is not substitutable by scenario injection, which is a pipeline-2
#    feature (train_selfplay.py) and never runs here. Scenario injection also
#    TELEPORTS 1-2 units from a hand-picked list of ~13 ids onto the bridge;
#    it never has an opponent play an unfamiliar deck through a real match,
#    so it exercises no elixir management, no cycle and no placement pattern.
#
# The counter-argument is real and is why the budget is 5000 and not more:
# pipeline 2's four scripted bots DO get randomized decks
# (set_scripted_opponent -> sample_random_deck), and two of them hold a 0.8
# PFSP weight floor, so random-deck exposure continues there at a far larger
# total volume than this phase can provide. This phase is a cheap pre-flight
# check, not the place generalization is actually learned.
RANDOM_OPPONENT_EPISODE_BUDGET = 5000

def sample_random_deck():
    """The `random_opponent` phase's deck source.

    Draws from the META POOL (opponents/deck_pool.py) when it is enabled, and
    falls back to the engine's uniform registry draw when it is not.

    THE FALLBACK IS THE OLD BEHAVIOUR AND IT IS A DIFFERENT DISTRIBUTION, not a
    weaker version of the same one. `clash_royale_env.sample_random_deck()`
    picks 8 of 132 cards uniformly, which produces Golem + P.E.K.K.A. + Mega
    Knight + Sparky far more often than any human would queue; gym_wrapper's own
    comment records that such decks beat a cycle deck "by tens of win-rate
    points" in this engine. A RoyaleAPI ladder deck is curated and elixir-
    balanced. Screening one of those two by measured win rate is a curriculum;
    screening the other is damage control.

    Correct-by-construction on the fallback path (not a raw random.sample over
    every registered card id, which would routinely violate
    CardRegistry::validateDeckSlots since Champions/Evolutions only fit some
    deck slots) -- see sampleRandomDeck's own comment in ClashEnv.h. The pool
    path cannot violate it either: deck_pool rejects Champions and Heroes at
    load time.
    """
    if PHASE1_DECK_POOL:
        decks = deck_pool.load_pool()
        return list(random.choice(decks).card_ids)
    return clash_royale_env.sample_random_deck()


def make_env(seed=None):
    """A phase-1 worker factory. `seed` is this worker's own, from
    `rl.seeding.worker_seeds` -- see selfplay_env.make_env for the contract.
    Both pipelines take the same `scenario_seed` config key rather than two
    conventions."""
    def _init():
        # THE PHASE-1 OPPONENT IS THE UTILITY TEACHER, at a symmetric 1.0x
        # economy. See rl/curriculum.py and opponents/teacher.py for why the C++
        # HeuristicOpponent no longer trains the agent (it remains an EVAL
        # anchor in train_selfplay.BUILTIN_ANCHORS, so historical numbers stay
        # comparable).
        return gym_wrapper.MicroRoyaleEnv({
            "opponent": PHASE1_OPPONENT,
            "teacher_stage": 0,
            "scenario_seed": seed,
            "deck_pool": PHASE1_DECK_POOL,
            "defensive_scenario_prob": PHASE1_DEFENSIVE_SCENARIO_PROB})
    return _init


class Phase1Trainer(BaseTrainer):
    """PPO vs the teacher, with the two-phase curriculum on top."""

    pipeline_name = "pipeline1"
    #: Defensive scenario windows can now expire here, exactly as in pipeline 2.
    #: Without this the window would be reported as a TERMINAL and the critic
    #: would bootstrap 0.0 through it -- teaching that holding a defence
    #: successfully is worth nothing, which inverts the lesson the scenario is
    #: injected to deliver. See BaseTrainer._truncation_bootstrap.
    uses_truncation_bootstrap = True
    #: ...and the other half of the same change. `draw_source` is
    #: `terminateds | truncateds` when this is False, so a scenario window
    #: expiring with a near-zero reward -- which is what a SUCCESSFUL defence
    #: looks like -- would be classified as a passivity draw and charged the
    #: full DRAW_PENALTY of 1.0. Injecting scenarios without this flag punishes
    #: exactly the behaviour the scenarios exist to teach.
    draw_on_terminated_only = True
    replay_prefix = "replay"

    def __init__(self, cfg=None):
        super().__init__(cfg or PPOConfig(), PHASE1_ENTROPY)
        # Overridable so a smoke run or an experiment arm cannot clobber the
        # real checkpoint. Both are redirected TOGETHER: a run writing scratch
        # weights into the live TensorBoard directory would silently interleave
        # two runs' curves and make both unreadable.
        # Anchored, never cwd-relative: the bare "model_weights.pth" this used
        # to be resolved against os.getcwd(), so a run launched from the repo
        # root started FRESH and one launched from python_ai/ RESUMED -- both
        # silently. See rl/checkpointing.weights_path.
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
        # Match the standalone replay env to the curriculum state actually in
        # progress -- otherwise it silently records against a stage-0 opponent
        # regardless of how far training has advanced, and keeps recording
        # mirror-deck replays once training has moved on to random decks.
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
        try:
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
            self._apply_curriculum_to_envs()
            print(f"Resumed from {self.weight_path}: "
                  f"episode {self.episodes_completed}, "
                  f"curriculum stage {self.curriculum.stage}, "
                  f"phase {self.curriculum.phase}")
            return True
        except RuntimeError:
            # Genuinely unexpected/corrupt checkpoint -- load_state_dict_flexible
            # already handles ordinary architecture-shape mismatches without
            # raising, so reaching here means something else is wrong. Keep it
            # as a backup and start fresh.
            backup = self.weight_path + ".bak"
            os.replace(self.weight_path, backup)
            print(f"Saved weights are incompatible with the current "
                  f"architecture; moved to {backup}, starting fresh.")
            return False

    def _apply_curriculum_to_envs(self):
        """Push the restored curriculum state into the workers.

        Phase 2 runs its OWN ladder per random deck, independent of phase 1's
        stage -- `CurriculumManager.teacher_stage` already resolves which one
        applies, so this cannot re-apply phase 1's final rung to a fresh deck
        the way the two separate branches it replaces could.
        """
        self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        if (self.curriculum.phase == "random_opponent"
                and self.curriculum.current_random_deck is not None):
            self.envs.call("set_opponent_deck",
                           self.curriculum.current_random_deck)

    def checkpoint_payload(self):
        return self.curriculum.state_dict()

    # -- the phase machine --------------------------------------------------
    def should_stop(self):
        # Second condition: see RANDOM_OPPONENT_EPISODE_BUDGET -- the
        # random-deck phase has no natural stopping point, so this is what ends
        # pipeline 1 and triggers the handoff to train_selfplay.py.
        return (self.episodes_completed >= 1_000_000
                or self.curriculum.budget_exhausted(self.episodes_completed))

    def on_episode_end(self, i, ctx):
        # A SCENARIO EPISODE IS NOT A MATCH RESULT. It is a 15-25 step window
        # on a live game that rarely ends in a crown, so recording it would
        # enter a non-win into the curriculum's 100-episode window ~30% of the
        # time and cap the achievable win rate near 0.70 against a 0.80 gate --
        # freezing the curriculum at whatever stage it happened to reach, with
        # no error and no log line. Pipeline 2 has always excluded these; phase
        # 1 gained scenario injection on 2026-08-29 and needs the same rule.
        # Pinned by tests/test_phase1_scenarios_do_not_break_the_gate.py.
        is_scenario = ctx.infos.get(
            "is_scenario", np.zeros(self.cfg.num_envs, dtype=np.float32))[i] > 0.5
        if is_scenario:
            self.metrics.reset_env(i)
            return

        ally_end, enemy_end = building_hp_end(ctx.next_obs[i])
        self.metrics.finish_episode(i, ctx.raw_rewards[i], ally_end, enemy_end)
        # The plateau detector reads its OWN 500-episode series, because the
        # gate clears its 100-episode window on every advance and demotion and
        # a "has the trend stopped" test cannot run on a series that is reset
        # whenever anything happens. Fed here, from the same episodes and under
        # the same scenario exclusion as the gate window above.
        if self.metrics.outcomes:
            self.curriculum.note_outcome(self.metrics.outcomes[-1])
        if self.episodes_completed % 10 == 0:
            self._print_progress()

        # ORDER IS LOAD-BEARING. The phase gate is evaluated BEFORE the stage
        # gate because the stage gate clears the outcome window when it fires;
        # the other order would let the stage advance always consume a window
        # that satisfies both, and reaching phase 2 would then need an extra
        # full window at the harder stage.
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

        # The STALL valve. Mutually exclusive with the advance gate by
        # construction (that one needs >=0.80, this one <=0.10), so the
        # ordering here is free -- unlike the phase/stage ordering above.
        demoted = self.curriculum.maybe_demote_stage(
            outcomes, self.episodes_completed)
        if demoted is not None:
            self._on_stage_demoted(demoted)
            return

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
        # The end of phase 1: the strongest mirror-deck policy, captured before
        # random opponent decks change the problem.
        save_stage_snapshot(
            self.net, STAGE_CHECKPOINT_DIR, c.stage, self.episodes_completed,
            CURRICULUM_STAGES[c.stage]["teacher_stage"],
            f"entered phase 2 at win_rate={win_rate:.2f} (end of mirror phase)")
        # WITH THE POOL ON, THE TRAINER MUST NOT PICK THE DECK. Pool sampling
        # runs first in `reset()`, so a `set_opponent_deck` here is overwritten
        # on the very next episode -- the trainer would print a deck the agent
        # never plays, which is precisely the silent divergence this repo keeps
        # paying for. The phase still does its real job (the budget, and the
        # handoff to pipeline 2); it just stops pretending to choose an opponent
        # that is already being chosen per episode, per worker.
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
        """One rung up, by mastery ("gate") or by convergence ("plateau").

        The REASON is printed and logged, not just the stage. A ladder position
        reached by plateau is a weaker claim than one reached by the gate --
        it says "this rung stopped teaching", not "this rung was beaten" -- and
        a run that plateaued up every rung is at the top without having won
        anything. That distinction has to survive into the logs, for the same
        reason `demotions` does.
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
        """The agent stopped scoring at this rung for a sustained stretch.

        Loud on purpose. A demotion means the ladder position was ahead of the
        policy's actual competence, so every stage number reported before it --
        including in any run summary already written down -- described a
        teacher the agent was not in fact beating.
        """
        self.envs.call("set_teacher_stage", self.curriculum.teacher_stage)
        print(f">>> [STALL] Curriculum DEMOTED to stage {new_stage} "
              f"(teacher_stage={self.curriculum.teacher_stage}) after a "
              f"sustained win rate at or below {STALL_WIN_RATE:.0%}. "
              f"Demotions this run: {self.curriculum.demotions}.")
        self.writer.add_scalar("Training/Curriculum_Stage", new_stage,
                               self.episodes_completed)
        self.writer.add_scalar("Training/Curriculum_Demotions",
                               self.curriculum.demotions,
                               self.episodes_completed)

    def _rotate_random_deck(self, reason):
        c = self.curriculum
        # See _on_entered_random_phase: with the pool on, the deck is already
        # re-drawn every episode by every worker, so "rotate" has nothing left
        # to rotate and setting one would be silently undone.
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

    # -- read-out -----------------------------------------------------------
    #: How often the per-deck read-out is printed, in episodes. Rarer than the
    #: progress line because it is a whole extra row and the EWMA behind it
    #: moves on a scale of hundreds of episodes anyway.
    DECK_READOUT_EVERY = 200

    def _print_deck_pool(self):
        """Per-deck win rates, averaged over the workers' local estimates.

        PHASE 1 HAD NO PER-DECK DIAGNOSTIC AT ALL, and that is half of why the
        2026-08-28 deck collapse ran 30,000 episodes unseen: every instrument
        in the loop was an AGGREGATE, and an aggregate cannot see a per-member
        failure. This is the eighth time this project has written that sentence
        down, so the pool ships with its own read-out rather than waiting for
        the first run to need one.

        Averaged and not summed: each worker holds an independent EWMA over the
        episodes IT played, so the mean across workers is the pooled estimate.
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
        # The MINIMUM is the number to watch, for the reason `ByCard_Min`
        # exists on the placement head: a pool average stays healthy while one
        # matchup is a total loss, and a matchup at ~0 is where the policy is
        # learning nothing and the PFSP floor is all that keeps it in rotation.
        self.writer.add_scalar("Decks/WinRate_Min", min(avg.values()),
                               self.episodes_completed)
        self.writer.add_scalar("Decks/WinRate_Spread",
                               max(avg.values()) - min(avg.values()),
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
        self._print_deck_pool()
        w = self.writer
        w.add_scalar("Training/Avg_Reward_50", m["avg_reward"], ep)
        w.add_scalar("Reward/Episode_Shaping_Sum", m["avg_shaping"], ep)
        w.add_scalar("Training/Win_Rate_100", m["win_rate"], ep)
        w.add_scalar("Training/Phase", 0 if c.phase == "mirror" else 1, ep)
        # Full outcome split: a rising win share should come out of the LOSS
        # share (getting stronger) or the DRAW share (closing games).
        w.add_scalar("Rates/Loss_100", m["loss_rate"], ep)
        w.add_scalar("Rates/Draw_100", m["draw_rate"], ep)
        w.add_scalar("Rates/Decisive_Win_100", m["decisive_win_rate"], ep)
        # How many of the last 100 games actually got decided -- a low value
        # explains why the curriculum gate has not advanced yet.
        w.add_scalar("Training/Decided_Count_100", m["decided"], ep)
        if math.isfinite(m["decisive_win_rate_long"]):
            w.add_scalar("Rates/Decisive_Win_500",
                         m["decisive_win_rate_long"], ep)
        # Learning to close games shows up as shorter episodes and less enemy
        # building HP left standing at the end.
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
        # Guaranteed fresh save at the stop point, not just whatever the
        # periodic cadence happened to catch -- pipeline 2 bootstraps from this
        # exact file next, so it must reflect the truly-latest trained state.
        self.save_checkpoint(verbose=False)
        self.envs.close()
        print(f">>> Pipeline #1 stopped at episode {self.episodes_completed} "
              f"(phase={self.curriculum.phase}) -- final checkpoint saved to "
              f"{self.weight_path}.")
        launch_pipeline2()


def launch_pipeline2():
    """Start train_selfplay.py and return, so nobody has to watch for the
    moment pipeline 1 finishes.

    `-u` is not optional. Python block-buffers stdout when it is redirected to
    a file, so without it the handoff produces a 0-byte log for a long stretch
    and the live strategy read-out -- whose entire purpose is watching the run
    as it happens -- is invisible until a buffer happens to flush.

    An ABSOLUTE path, not a bare filename: the two pipelines live in
    python_ai/trainers/ while the run's cwd is python_ai/ (where the
    checkpoints are).
    """
    out = open("training_selfplay_pfsp.log", "w")
    err = open("training_selfplay_pfsp_err.log", "w")
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "train_selfplay.py")
    subprocess.Popen([sys.executable, "-u", target], stdout=out, stderr=err)
    print(">>> Launched train_selfplay.py (pipeline #2) -- see "
          "training_selfplay_pfsp.log / training_selfplay_pfsp_err.log")


def train_ppo():
    """Entry point, kept under its historical name."""
    Phase1Trainer().run()


if __name__ == "__main__":
    # Prevent safe pickling errors in Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    train_ppo()
