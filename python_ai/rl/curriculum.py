"""The phase-1 curriculum: WHICH opponent, and WHEN it gets harder.

DIFFICULTY IS COMPETENCE, NOT ECONOMY (2026-08-19). This used to escalate the
OPPONENT'S ELIXIR MULTIPLIER, 1.0 -> 1.5 in 0.1 steps, and that is what priced
the win condition negatively. Measured, SMART-forced A/B (advisor timing gate +
advisor bridge cell), n=120 paired:

    opponent   baseline win   forced win   delta      p
    1.00x         1.000          1.000     +0.0000    VOID (ceiling)
    1.25x         0.950          0.825     -0.1250    0.0059
    1.50x         0.617          0.317     -0.3000    3.2e-06

A punish window lasts about `answer_cost / (m * r)`, so at m=1.5 it is two
thirds its natural length, while our 4 elixir is spent regardless and what the
opponent does with their surplus scales with m. Defence moves the OPPOSITE way,
because a multiplier increases exactly the threat volume defence is priced
against. The multiplier does not shift the optimum, it INVERTS the ranking of
strategy classes -- which is why four separate interventions to rehabilitate the
Hog all returned null: every one of them moved a POLICY and none of them changed
the PAYOFF.

DELETING THE MULTIPLIER ALONE WOULD NOT WORK. At 1.0x the C++ heuristic is
beaten ~100%, so that converts a MISPRICED environment into a ZERO-GRADIENT one.
The multiplier is therefore REPLACED by competence: both sides always run at
1.0x and the ladder is the teacher's lookahead / candidate width / epsilon
(`opponents.teacher.TEACHER_STAGES`).

`set_opponent_elixir_multiplier` is deliberately still BOUND and still callable
-- ~15 measurement harnesses sweep it, including the falsifier that justified
this change. It is simply never used by training again, and
`tests/test_curriculum.py` fails if it ever is.

WHY THIS LIVES AT MODULE SCOPE. It used to be a local of `train_ppo()`, so no
test could import it: the suite parsed `train.py` with `ast` to recover the
literal. That worked and was a bad sign -- a constant nothing can import is a
constant nothing can check.
"""
import os
from collections import deque

#: The ordering around the gate is unchanged and still load-bearing: the PHASE
#: transition is evaluated BEFORE stage advancement, because the stage gate
#: clears the outcome window when it fires and would otherwise always consume a
#: window that satisfies both.
#:
#: Advance gate, lowered 0.80 -> 0.65 on 2026-09-03 together with the eleven-rung
#: teacher table. The two changes are one change and must not be separated: 0.80
#: was priced against SIX rungs, and holding it while tripling the number of
#: steps would make the ladder strictly harder to climb, not easier.
#:
#: 0.80 was also never really 0.80. This module's own stall-valve note records
#: the measurement: the gate is re-tested on every episode over hundreds of
#: OVERLAPPING windows, so an agent whose true skill is 0.70 clears one window
#: 1.6% of the time and clears at least one within 3,000 episodes 91.6% of the
#: time. The nominal number was an optional-stopping test, not a threshold.
#:
#: AND IT IS A MIRROR-MATCH GATE, which is what made it unreachable rather than
#: merely strict. Demanding 0.80 while the opponent plays OUR OWN deck is
#: demanding domination of a forward-simulating opponent holding identical
#: resources; a symmetric matchup between comparable players sits near 0.50 by
#: construction. The deck pool (opponents/deck_pool.py) is what makes a number
#: above 0.50 a coherent ask at all, because a diverse pool contains matchups
#: the agent can be genuinely favoured in.
STAGE_WIN_RATE_GATE = float(os.environ.get("CLASH_STAGE_GATE", 0.65))

#: Eleven rungs, indexing `opponents.teacher.TEACHER_STAGES` 1:1. The final rung
#: keeps `None` -- there is still no auto-advance past the top and a stopping
#: rule is still chosen by hand.
CURRICULUM_STAGES = [
    {"teacher_stage": i, "win_rate_threshold": STAGE_WIN_RATE_GATE}
    for i in range(10)
] + [{"teacher_stage": 10, "win_rate_threshold": None}]

#: Window the gates read. 100 episodes is a +/-0.1 sampling band on a win rate,
#: which is the coarsest resolution the thresholds below are meaningful at.
OUTCOME_WINDOW = 100

#: --- the plateau valve -----------------------------------------------------
#: THE STALL THAT MOTIVATED THIS, measured off runs/phase4: the 2026-08-28 run
#: spent episodes 9,640 -> 32,680 -- 23,040 consecutive episodes, ~24 hours at
#: the measured 943 ep/h -- at stage 3, win rate oscillating 0.15-0.60, and
#: advanced ZERO times. Nothing in the loop could end that. The advance gate
#: needs 0.80 and never saw it; the stall valve needs <=0.10 sustained over
#: fifteen windows and only caught the two deepest troughs, demoting twice and
#: climbing back both times.
#:
#: So the ladder had a gate for MASTERY and a valve for CATASTROPHE and nothing
#: at all for the case that actually happened: an agent that is competitive,
#: learning nothing further, and not failing. That band is where a curriculum
#: run spends its time, and it was the one band with no exit.
#:
#: The rule is ordinary early stopping, on the quantity the noise does not move:
#: a 500-episode mean rather than the 100-episode window, because the thing
#: being detected is the absence of a TREND and a 100-episode window swings
#: +/-0.1 on sampling alone. Advance when that mean has not improved by
#: `PLATEAU_IMPROVEMENT` for `PLATEAU_PATIENCE_EPISODES` AND is at least
#: `PLATEAU_MIN_WIN_RATE` -- competitive but converged. Below that floor the
#: agent is not being held back by the rung, it is losing to it, and promotion
#: would be the over-promotion the stall valve exists to undo.
#:
#: Applied to that stall the valve fires within ~2,000 episodes of entering the
#: rung (a 500-episode window to establish the trend, then 1,500 of patience)
#: instead of never. `test_curriculum_plateau.py` replays the real oscillation
#: and bounds it under episode 4,000, so the saving is ~19,000 episodes --
#: about 20 hours at the measured 943 ep/h.
#:
#: THE PLATEAU IS THE WORKHORSE AND THE GATE IS THE FAST PATH, once the deck
#: pool is on, and that division of labour is forced by PFSP rather than chosen.
#: PFSP weights a deck by `(1 - win_rate)^2`, so it deliberately spends the run
#: on whatever the agent is doing WORST against -- which REGULATES the readable
#: win rate toward the hard end of the pool. Measured on the 2026-09-03 sweep's
#: own numbers (8 decks, 0.07-1.00): the agent beats three decks at 0.70-1.00
#: and the episode-weighted pool win rate still reads 0.581. A LEVEL gate can
#: therefore be structurally unreachable on a heterogeneous pool no matter how
#: good the agent gets, which is the same shape of unreachability the 0.80
#: mirror gate had, arriving by a different road.
#:
#: So `PLATEAU_MIN_WIN_RATE` is 0.40 and is NOT comparable to a mirror win rate.
#: It is a floor on a PFSP-weighted mixture that is deliberately biased toward
#: the agent's worst matchups; 0.40 there is a genuinely competitive agent,
#: while 0.40 against a single fixed opponent would not be.
PLATEAU_WINDOW = 500
PLATEAU_PATIENCE_EPISODES = 1500
PLATEAU_MIN_WIN_RATE = 0.40
PLATEAU_IMPROVEMENT = 0.02

#: --- the backstop ----------------------------------------------------------
#: A HARD CEILING ON EPISODES AT ONE RUNG, and the reason it exists even though
#: the plateau valve above should make it unreachable: the 2026-08-28 run burned
#: 23,040 episodes at one rung, and the whole point of this change is that that
#: must not be able to happen again for ANY reason -- including a reason nobody
#: has thought of, and including a bug in the plateau valve itself.
#:
#: It fires DOWNWARD only, and covers exactly the gap the plateau valve leaves:
#: a rung the agent has converged on BELOW `PLATEAU_MIN_WIN_RATE`. That state is
#: not catastrophic enough for the 0.10 stall valve and not competitive enough
#: to promote, so before this it could hold a run indefinitely -- which is
#: precisely what the 2026-08-28 run did. Such a run is over-promoted, so it
#: goes down a rung and tries again from a teacher it can score against.
#:
#: MEASURED FROM THE LAST IMPROVEMENT, not from the start of the rung. The first
#: version of this used elapsed time and would have cut off an agent that was
#: still climbing, which is worse than the stall it was added to prevent;
#: `test_a_still_improving_rung_is_never_moved` pins that it cannot.
#:
#: 4,000 is ~2x the plateau's own worst case (a 500-episode window to establish
#: the trend plus 1,500 of patience), so it never pre-empts the measured path.
MAX_EPISODES_PER_RUNG = int(os.environ.get("CLASH_MAX_EPISODES_PER_RUNG", 4000))

#: --- the stall valve -------------------------------------------------------
#: The ladder above is otherwise STRICTLY ONE-WAY: `maybe_advance_stage` only
#: increments and nothing reduced `stage`. A run promoted past its competence
#: therefore had no way back, which is the dead end this project has already
#: hit -- 0-for-2000+ episodes with zero improvement.
#:
#: And promotion is optimistic. The 0.80 gate is re-tested on EVERY episode, so
#: it is an optional-stopping test over hundreds of overlapping windows.
#: Measured 2026-08-26 by simulation: an agent whose TRUE skill is 0.70 clears
#: a single window 1.6% of the time but clears at least one within 3000
#: episodes 91.6% of the time. The effective gate is ~0.70, not 0.80, and five
#: rungs compound it.
#:
#: These two numbers are set so the valve CANNOT fire on a run that is merely
#: finding a stage hard: 0.10 is far below any healthy win rate at any rung,
#: and the patience is fifteen full windows of it. A run this far gone has
#: already stopped producing gradient.
STALL_WIN_RATE = 0.10
STALL_PATIENCE_EPISODES = 1500


def window_win_rate(outcome_history):
    """Raw win rate over a FULL window, or None if the window is not full yet.

    Raw (wins / all games), not decisive (wins / decided): a high draw rate lets
    a mediocre bot read well on the decisive rate -- 40% win / 13% loss / 47%
    draw is 75% decisive while only actually winning 40% of games.
    """
    if len(outcome_history) != outcome_history.maxlen:
        return None
    outcomes = list(outcome_history)
    return sum(1 for o in outcomes if o == 1) / len(outcomes)


class CurriculumManager:
    """Pipeline 1's opponent-difficulty state machine.

    Two phases, and the second one replays the SAME stage ladder per deck:

        mirror            the opponent plays our own 8-card deck; clearing
                          stage >= PHASE2_MIN_CURRICULUM_STAGE at
                          `entry_win_rate` moves us on
        random_opponent   the opponent plays a rotating random deck, each of
                          which climbs the ladder from stage 0 again

    Rotation inside `random_opponent` is PERFORMANCE-gated, not timer-gated: a
    deck the agent handles clears every stage quickly and rotates out; one it has
    no answer for keeps accumulating training time where it is failing.
    `max_episodes_per_deck` is a SAFETY VALVE for a pathological draw, not the
    intended trigger.
    """

    def __init__(self, *, entry_win_rate, min_stage_for_phase2,
                 phase2_win_rate_gate, max_episodes_per_deck,
                 random_opponent_budget):
        self.entry_win_rate = entry_win_rate
        self.min_stage_for_phase2 = min_stage_for_phase2
        self.phase2_win_rate_gate = phase2_win_rate_gate
        self.max_episodes_per_deck = max_episodes_per_deck
        self.random_opponent_budget = random_opponent_budget

        self.stage = 0
        self.stage_start_episode = 0
        #: How many times the stall valve has fired. A run that has demoted is
        #: a run whose ladder position is NOT evidence of competence, so this
        #: has to be visible and has to survive a resume.
        self.demotions = 0
        self.last_demotion_reason = ""
        #: How many rungs were cleared by PLATEAU rather than by the gate. Read
        #: it as "how much of this ladder position is mastery and how much is
        #: only convergence" -- a run that plateaued up every rung is at the top
        #: without ever having beaten anything, and that has to be visible in
        #: the same way `demotions` is.
        self.plateau_advances = 0
        #: The plateau detector's own state. `rung_history` is deliberately NOT
        #: the gate's window: the gate CLEARS its window on every advance and
        #: demotion, and a detector for "no trend" cannot run on a series that
        #: is reset whenever anything happens.
        self.rung_history = deque(maxlen=PLATEAU_WINDOW)
        self.best_rung_mean = 0.0
        self.best_rung_episode = 0
        #: An improvement signal PFSP does NOT regulate, or None on a path that
        #: has none (the mirror, where there are no per-deck estimates).
        #:
        #: MEASURED 2026-09-06, live phase-9 run, ep 83,128 -> 87,540: the
        #: unweighted per-deck mean went 0.301 -> 0.534 with every one of
        #: sixteen decks improving, while the readable win rate sat at ~0.50 --
        #: so this valve fired TWICE, calling the fastest learning of the run a
        #: plateau, and the rung it promoted to then made the agent WORSE
        #: (mean 0.535 -> 0.522, readable rate to 0.21, within 600 episodes).
        #:
        #: The cause is stated in this module's own docstring and then applied
        #: only to the level GATE: PFSP weights a deck by `(1 - win_rate)^2`, so
        #: it spends the run on the worst matchups and pins the readable rate no
        #: matter how good the agent gets. "Has not improved for 1,500 episodes"
        #: is therefore satisfied BY CONSTRUCTION, which turns the valve from a
        #: convergence detector into a timer with a ~2,200-episode period.
        #:
        #: Raising PLATEAU_PATIENCE_EPISODES would only slow a blind timer. What
        #: changes is WHAT THE IMPROVEMENT TEST READS. The
        #: `PLATEAU_MIN_WIN_RATE` floor still reads the readable rate, which is
        #: correct: that one is a competitiveness check, not a trend.
        self.progress_signal = None
        self.phase = "mirror"
        self.deck_stage = 0
        self.deck_episode_start = 0
        self.random_phase_episode_start = 0
        self.current_random_deck = None

    # -- what the env should be set to -------------------------------------
    @property
    def teacher_stage(self):
        """The teacher rung implied by the current phase and stage."""
        table = (CURRICULUM_STAGES[self.deck_stage] if self.phase == "random_opponent"
                 else CURRICULUM_STAGES[self.stage])
        return table["teacher_stage"]

    def budget_exhausted(self, episodes_completed):
        """True once the random-deck phase has had its full budget.

        Counted from when the PHASE started, not from episode 0: the budget is
        "how much random-deck exposure is enough", which has nothing to do with
        how many episodes the mirror curriculum happened to take.
        """
        return (self.phase == "random_opponent"
                and episodes_completed - self.random_phase_episode_start
                >= self.random_opponent_budget)

    # -- the transitions ----------------------------------------------------
    def maybe_enter_random_phase(self, outcome_history, episodes_completed):
        """Mirror -> random_opponent. Returns the win rate that fired it, or None.

        Checked BEFORE `maybe_advance_stage` and that ordering is load-bearing:
        the stage gate clears `outcome_history`, so if it ran first a window
        satisfying both gates would always be consumed by the stage advance and
        the phase transition could never see it.
        """
        if self.phase != "mirror" or self.stage < self.min_stage_for_phase2:
            return None
        win_rate = window_win_rate(outcome_history)
        if win_rate is None or win_rate < self.entry_win_rate:
            return None
        self.phase = "random_opponent"
        self.deck_stage = 0
        outcome_history.clear()
        self.deck_episode_start = episodes_completed
        self.random_phase_episode_start = episodes_completed
        # Re-boost exploration for the new opponent variety.
        self.stage_start_episode = episodes_completed
        return win_rate

    def note_outcome(self, outcome):
        """Feed the plateau detector one episode result (1 win / 0 otherwise).

        Separate from `outcome_history` on purpose -- see `rung_history`. The
        trainer calls this for exactly the episodes it also enters into the
        gate's window, so a scenario episode is excluded from both or neither.
        """
        self.rung_history.append(1 if outcome == 1 else 0)

    def rung_mean(self):
        """Long-window win rate at this rung, or None until the window fills."""
        if len(self.rung_history) < self.rung_history.maxlen:
            return None
        return sum(self.rung_history) / len(self.rung_history)

    def note_progress(self, value):
        """Offer an improvement signal PFSP does not regulate.

        The caller owns what it means; phase 1 passes the UNWEIGHTED mean of the
        per-deck win-rate estimates, which is the quantity that moved while the
        readable rate did not. `None` restores the pre-2026-09-06 behaviour
        exactly, which is what the mirror path (no deck pool) gets.
        """
        self.progress_signal = None if value is None else float(value)

    def _improvement_signal(self, long_mean):
        """What the plateau's trend test reads: the progress signal if the
        caller supplied one, else the long window as before."""
        return long_mean if self.progress_signal is None else self.progress_signal

    def _reset_rung_tracking(self, episodes_completed):
        self.rung_history.clear()
        self.best_rung_mean = 0.0
        self.best_rung_episode = episodes_completed

    def maybe_advance_stage(self, outcome_history, episodes_completed):
        """Mirror-phase stage gate. Returns (new_stage, reason), or None.

        Two ways up, and the second one is the 2026-09-03 addition:

          "gate"     the 100-episode window cleared `win_rate_threshold`.
                     Mastery.
          "plateau"  the 500-episode mean has stopped improving while staying
                     above `PLATEAU_MIN_WIN_RATE`. Convergence -- this rung has
                     no gradient left to give, and 23,040 episodes of the
                     2026-08-28 run went into proving there was no other exit.

        Guarded on the mirror phase: this block used to run in BOTH, harmless
        only while phase 2 required the FINAL stage. Now that phase 2 can start
        earlier, an unguarded version would keep escalating `stage` DURING
        phase 2 and fight the per-deck ladder for control of the opponent.
        """
        if self.phase != "mirror":
            return None
        threshold = CURRICULUM_STAGES[self.stage]["win_rate_threshold"]
        if threshold is None or self.stage + 1 >= len(CURRICULUM_STAGES):
            return None

        reason = None
        win_rate = window_win_rate(outcome_history)
        long_mean = self.rung_mean()
        if win_rate is not None and win_rate >= threshold:
            reason = "gate"
        else:
            if long_mean is not None:
                # TREND on a signal PFSP does not regulate; FLOOR on the
                # readable rate. Two different questions, so two different
                # quantities -- see `progress_signal`.
                trend = self._improvement_signal(long_mean)
                if trend > self.best_rung_mean + PLATEAU_IMPROVEMENT:
                    self.best_rung_mean = trend
                    self.best_rung_episode = episodes_completed
                elif (episodes_completed - self.best_rung_episode
                        >= PLATEAU_PATIENCE_EPISODES
                        and trend >= PLATEAU_MIN_WIN_RATE):
                    reason = "plateau"
        if reason is None:
            return None

        self.stage += 1
        if reason == "plateau":
            self.plateau_advances += 1
        outcome_history.clear()
        self._reset_rung_tracking(episodes_completed)
        self.stage_start_episode = episodes_completed
        return self.stage, reason

    def maybe_demote_stage(self, outcome_history, episodes_completed):
        """Mirror-phase STALL valve. Returns the new stage, or None.

        Steps one rung back down when the agent has been failing at the current
        rung for a sustained stretch. Without it the ladder is one-way and a
        run promoted past its competence burns indefinitely at a teacher it
        cannot score against -- see STALL_PATIENCE_EPISODES for the measurement
        showing the promotion gate over-promotes by ~10 win-rate points.

        Mirror phase only. The random-deck ladder already has its own valve
        (`max_episodes_per_deck` rotates a deck it cannot beat), and two valves
        driving one stage number would fight each other.

        Stage 0 is a floor and not merely an index guard: losing at stage 0
        means the teacher is rules-only and is NOT what is wrong, so stepping
        back would fix nothing while hiding the real fault.
        """
        if self.phase != "mirror" or self.stage <= 0:
            return None
        if episodes_completed - self.stage_start_episode < STALL_PATIENCE_EPISODES:
            return None
        win_rate = window_win_rate(outcome_history)
        long_mean = self.rung_mean()
        # The ordinary catastrophe trigger, unchanged...
        catastrophic = win_rate is not None and win_rate <= STALL_WIN_RATE
        # ...plus the DOWNWARD BACKSTOP (2026-09-03), which closes the one gap
        # the plateau valve leaves open. `maybe_advance_stage` promotes a
        # converged rung only when it is at least PLATEAU_MIN_WIN_RATE; a run
        # converged BELOW that floor is exactly the 2026-08-28 stall -- not
        # catastrophic enough for the 0.10 valve, not competitive enough to
        # promote, and previously able to sit there forever. It is over-promoted,
        # so it goes DOWN.
        #
        # Conditioned on time since the last IMPROVEMENT and not on time at the
        # rung, which the first version of this got wrong: an agent still
        # climbing must never be moved, and `best_rung_episode` is the only
        # quantity that distinguishes "stuck" from "slow".
        #
        # DEPENDS ON `maybe_advance_stage` HAVING RUN THIS EPISODE, because that
        # is what refreshes `best_rung_episode`. train.py calls them in that
        # order and returns early when the advance fires, so the value is always
        # current here; reversing the order would freeze this backstop's clock
        # and it would fire on an improving run. Pinned by
        # `test_a_still_improving_rung_is_never_moved_by_the_backstop`, which
        # calls both in the trainer's order.
        # THE FLOOR READS THE UNREGULATED SIGNAL TOO, since 2026-09-06, and for
        # the same reason the plateau's trend test does one function up.
        #
        # PFSP weights a deck by (1 - win_rate)^2, so the readable rate is
        # driven toward the agent's WORST matchups and stays there however
        # strong it gets. Measured on the live phase-9 run: an unweighted
        # per-deck mean of 0.62 -- beating fourteen decks of sixteen -- while
        # `long_mean` read 0.29. This test fired on the 0.29 and demoted rung
        # 3 -> 2 -> 1, weakening the teacher on an agent that was improving,
        # and would have continued to rung 0.
        #
        # "Is the agent competitive here" is a LEVEL question and so needs a
        # level the sampler does not regulate. Falls back to `long_mean` when no
        # progress signal is supplied, which is the mirror path.
        floor_signal = self._improvement_signal(long_mean)
        capped_out = (floor_signal is not None
                      and floor_signal < PLATEAU_MIN_WIN_RATE
                      and episodes_completed - self.best_rung_episode
                      >= MAX_EPISODES_PER_RUNG)
        if not (catastrophic or capped_out):
            return None
        # WHICH valve fired, for the caller's log line. Both paths call
        # `stage -= 1`, and the trainer's message used to hardcode the
        # catastrophe wording -- so a BACKSTOP demotion reported "sustained win
        # rate at or below 10%" while the actual rate was 0.39, and cost a
        # reader a full investigation to discover the message was wrong.
        self.last_demotion_reason = (
            f"win rate {win_rate:.2f} at or below the {STALL_WIN_RATE:.0%} "
            f"catastrophe floor" if catastrophic else
            f"500-episode mean {long_mean:.2f} below the "
            f"{PLATEAU_MIN_WIN_RATE:.0%} floor with no improvement for "
            f"{MAX_EPISODES_PER_RUNG} episodes (backstop, not catastrophe)")
        self.stage -= 1
        self.demotions += 1
        # Same bookkeeping an advance does: the rung must be judged on fresh
        # episodes, and exploration re-boosts for the changed opponent. The
        # plateau tracker is reset here too -- a 500-episode mean carried across
        # a rung change describes an opponent that is no longer being played,
        # and would let a demotion immediately look like a plateau.
        outcome_history.clear()
        self._reset_rung_tracking(episodes_completed)
        self.stage_start_episode = episodes_completed
        return self.stage

    def step_random_deck_curriculum(self, outcome_history, episodes_completed):
        """The per-deck ladder inside `random_opponent`.

        Returns one of ("rotate", reason), ("advance", new_deck_stage), or None.
        The caller owns deck SAMPLING -- this class never touches the engine.
        """
        if self.phase != "random_opponent":
            return None
        win_rate = window_win_rate(outcome_history)
        at_final = self.deck_stage == len(CURRICULUM_STAGES) - 1
        mastered = (at_final and win_rate is not None
                    and win_rate >= self.phase2_win_rate_gate)
        timed_out = (episodes_completed - self.deck_episode_start
                     >= self.max_episodes_per_deck)
        if mastered or timed_out:
            self.deck_stage = 0
            outcome_history.clear()
            self.deck_episode_start = episodes_completed
            self.stage_start_episode = episodes_completed
            reason = (f"mastered it (>={self.phase2_win_rate_gate:.0%} at the "
                      "final stage)" if mastered else
                      f"hit the {self.max_episodes_per_deck}-episode safety cap "
                      "without mastering it")
            return ("rotate", reason)

        threshold = CURRICULUM_STAGES[self.deck_stage]["win_rate_threshold"]
        if (not at_final and threshold is not None
                and win_rate is not None and win_rate >= threshold):
            self.deck_stage += 1
            outcome_history.clear()
            # Re-boost exploration for the harder version of the SAME deck.
            self.stage_start_episode = episodes_completed
            return ("advance", self.deck_stage)
        return None

    # -- persistence --------------------------------------------------------
    def state_dict(self):
        return {
            "curriculum_stage": self.stage,
            "curriculum_demotions": self.demotions,
            "curriculum_plateau_advances": self.plateau_advances,
            #: Stamps which teacher table this index was written against. Its
            #: ABSENCE is what identifies a pre-2026-09-03 checkpoint, so the
            #: remap in load_state_dict runs exactly once and can never run on
            #: an index it already converted.
            "teacher_table_size": len(CURRICULUM_STAGES),
            "stage_start_episode": self.stage_start_episode,
            "phase": self.phase,
            "deck_curriculum_stage": self.deck_stage,
            "phase_deck_episode_start": self.deck_episode_start,
            "random_phase_episode_start": self.random_phase_episode_start,
            "current_random_deck": self.current_random_deck,
        }

    def load_state_dict(self, state):
        """Restore, tolerating checkpoints written before a key existed.

        `random_phase_episode_start` falls back to `phase_deck_episode_start`
        and NOT to 0: a checkpoint written before that key existed was already
        in the phase, and defaulting to 0 would make the elapsed budget look
        like the full episode count and hand off immediately on resume.
        """
        # A saved stage is an INDEX into the teacher table that was live when it
        # was written. The 2026-09-03 table went 6 rungs -> 11, so a checkpoint
        # with no `teacher_table_size` stamp is indexing the old one and must be
        # remapped by HORIZON -- otherwise a run saved at old stage 3 (5 s
        # lookahead) silently resumes at new rung 3 (2 s), a two-rung demotion
        # with no log line. Exactly the failure class as reinterpreting an
        # observation layout across a size change.
        from python_ai.opponents.teacher import remap_legacy_stage
        saved_table = state.get("teacher_table_size")
        stage = int(state["curriculum_stage"])
        if saved_table is None or saved_table != len(CURRICULUM_STAGES):
            remapped = remap_legacy_stage(stage)
            if remapped != stage:
                print(f">>> Curriculum: checkpoint stage {stage} was written "
                      f"against a {saved_table or 6}-rung teacher table; "
                      f"remapped to rung {remapped} (same lookahead horizon).")
            stage = remapped
        self.stage = stage
        self.demotions = state.get("curriculum_demotions", 0)
        self.plateau_advances = state.get("curriculum_plateau_advances", 0)
        self.stage_start_episode = state["stage_start_episode"]
        # Never restored: a 500-episode trend belongs to the episodes that
        # produced it, and a resume has none of them in flight.
        self._reset_rung_tracking(self.stage_start_episode)
        self.phase = state.get("phase", "mirror")
        self.deck_stage = state.get("deck_curriculum_stage", 0)
        self.deck_episode_start = state.get("phase_deck_episode_start", 0)
        self.random_phase_episode_start = state.get(
            "random_phase_episode_start", self.deck_episode_start)
        self.current_random_deck = state.get("current_random_deck", None)


def new_outcome_window(maxlen=OUTCOME_WINDOW):
    return deque(maxlen=maxlen)
