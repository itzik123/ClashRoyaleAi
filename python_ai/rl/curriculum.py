"""The phase-1 curriculum: which opponent, and when it gets harder.

Difficulty is the teacher's competence (lookahead, candidate width, epsilon;
`opponents.teacher.TEACHER_STAGES`), never the opponent's elixir multiplier. A
multiplier inverts the ranking of strategies: it shortens punish windows and
inflates exactly the threat volume defence is priced against, so it made the
win condition a losing play. `set_opponent_elixir_multiplier` stays bound for
measurement harnesses; `tests/test_curriculum.py` fails if training uses it.
"""
import os
from collections import deque

#: Advance gate. The gate is re-tested every episode over overlapping windows,
#: so its effective threshold is lower than its nominal one. Only meaningful
#: against a diverse deck pool: a mirror match between comparable players sits
#: near 0.50.
STAGE_WIN_RATE_GATE = float(os.environ.get("CLASH_STAGE_GATE", 0.65))

#: Eleven rungs, indexing TEACHER_STAGES 1:1. No auto-advance past the top.
CURRICULUM_STAGES = [
    {"teacher_stage": i, "win_rate_threshold": STAGE_WIN_RATE_GATE}
    for i in range(10)
] + [{"teacher_stage": 10, "win_rate_threshold": None}]

#: The gates' window; 100 episodes is a +/-0.1 sampling band.
OUTCOME_WINDOW = 100

#: Plateau valve: advance when the 500-episode trend has not improved by
#: PLATEAU_IMPROVEMENT for PLATEAU_PATIENCE_EPISODES and the agent is still
#: competitive (>= PLATEAU_MIN_WIN_RATE). This is the exit for a
#: converged-but-competitive agent, the band a run spends most of its time in.
#:
#: With the deck pool on, PFSP samples the agent's worst matchups, which pins
#: the readable win rate low however strong the agent gets; the plateau is then
#: the workhorse and the gate the fast path. For the same reason 0.40 here is
#: not comparable to a mirror win rate.
PLATEAU_WINDOW = 500
PLATEAU_PATIENCE_EPISODES = 1500
PLATEAU_MIN_WIN_RATE = 0.40
PLATEAU_IMPROVEMENT = 0.02

#: Twice PLATEAU_IMPROVEMENT: a converged rung wobbles within one
#: PLATEAU_IMPROVEMENT by construction, so a fall of twice that is not noise.
REGRESSION_MARGIN = 2 * PLATEAU_IMPROVEMENT

#: Backstop: demote a rung converged below PLATEAU_MIN_WIN_RATE with no
#: improvement for this long. Counted from the last improvement, so a
#: still-climbing agent is never moved. ~2x the plateau's worst case, so it
#: never pre-empts it.
MAX_EPISODES_PER_RUNG = int(os.environ.get("CLASH_MAX_EPISODES_PER_RUNG", 4000))

#: Stall valve: demote on a sustained catastrophic win rate. Set so it cannot
#: fire on a run that merely finds a rung hard.
STALL_WIN_RATE = 0.10
STALL_PATIENCE_EPISODES = 1500

#: Floor alarm: rung 0 has nowhere to demote to, so a dead run there would
#: otherwise be silent. Not a demotion, just loud.
FLOOR_ALARM_WIN_RATE = 0.05
FLOOR_ALARM_PATIENCE_EPISODES = 1000


def window_win_rate(outcome_history):
    """Raw win rate over a full window, or None if the window is not full.

    Raw (wins / all games), not decisive: a high draw rate flatters the
    decisive rate.
    """
    if len(outcome_history) != outcome_history.maxlen:
        return None
    outcomes = list(outcome_history)
    return sum(1 for o in outcomes if o == 1) / len(outcomes)


class CurriculumManager:
    """Pipeline 1's opponent-difficulty state machine.

        mirror            the opponent plays our deck; clearing
                          stage >= PHASE2_MIN_CURRICULUM_STAGE at `entry_win_rate`
                          moves us on
        random_opponent   the opponent plays a rotating random deck, each of which
                          climbs the ladder from stage 0 again

    Rotation inside `random_opponent` is performance-gated;
    `max_episodes_per_deck` is only a safety valve.
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
        #: Not persisted: re-arming once after a resume is correct.
        self._last_floor_alarm = -10 ** 9
        #: Visible and persisted: a demoted run's rung is not evidence of
        #: competence.
        self.demotions = 0
        self.last_demotion_reason = ""
        #: Rungs cleared by plateau rather than by the gate: how much of the
        #: ladder position is convergence rather than mastery.
        self.plateau_advances = 0
        #: Separate from the gate's window, which is cleared on every advance
        #: and demotion; a trend detector needs an uninterrupted series.
        self.rung_history = deque(maxlen=PLATEAU_WINDOW)
        self.best_rung_mean = 0.0
        self.best_rung_episode = 0
        #: An improvement signal PFSP does not regulate (phase 1 passes the
        #: unweighted per-deck mean), or None on the mirror path. The readable
        #: win rate cannot serve: PFSP pins it, so "has not improved" would
        #: hold by construction and the plateau valve would fire on a timer.
        self.progress_signal = None
        #: Progress when the current rung began; see `_reset_rung_tracking`.
        self.rung_entry_progress = None
        self.phase = "mirror"
        self.deck_stage = 0
        self.deck_episode_start = 0
        self.random_phase_episode_start = 0
        self.current_random_deck = None

    @property
    def teacher_stage(self):
        """The teacher rung implied by the current phase and stage."""
        table = (CURRICULUM_STAGES[self.deck_stage] if self.phase == "random_opponent"
                 else CURRICULUM_STAGES[self.stage])
        return table["teacher_stage"]

    def budget_exhausted(self, episodes_completed):
        """True once the random-deck phase has had its full budget, counted from
        when the phase started.
        """
        return (self.phase == "random_opponent"
                and episodes_completed - self.random_phase_episode_start
                >= self.random_opponent_budget)

    def maybe_enter_random_phase(self, outcome_history, episodes_completed):
        """Mirror -> random_opponent. Returns the win rate that fired it, or None.

        Must run before `maybe_advance_stage`, which clears `outcome_history`:
        otherwise a window satisfying both gates is always consumed by the
        stage advance.
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
        # Re-boost exploration for the new opponents.
        self.stage_start_episode = episodes_completed
        return win_rate

    def note_outcome(self, outcome):
        """Feed the plateau detector one episode result (1 win / 0 otherwise).

        Called for exactly the episodes that enter the gate's window.
        """
        self.rung_history.append(1 if outcome == 1 else 0)

    def rung_mean(self):
        """Long-window win rate at this rung, or None until the window fills."""
        if len(self.rung_history) < self.rung_history.maxlen:
            return None
        return sum(self.rung_history) / len(self.rung_history)

    def note_progress(self, value):
        """Offer an improvement signal PFSP does not regulate; None on the mirror
        path.
        """
        self.progress_signal = None if value is None else float(value)
        # Latch the entry level on the first signal: a rung usually starts
        # before deck stats are polled (always on a resume).
        if self.rung_entry_progress is None:
            self.rung_entry_progress = self.progress_signal

    def _improvement_signal(self, long_mean):
        """The progress signal if supplied, else the long window."""
        return long_mean if self.progress_signal is None else self.progress_signal

    def _reset_rung_tracking(self, episodes_completed):
        self.rung_history.clear()
        self.best_rung_mean = 0.0
        self.best_rung_episode = episodes_completed
        # The level this rung started at, to judge whether it helped or hurt.
        # `best_rung_mean` only updates once the 500-episode window fills, so
        # it misses the start of a decline.
        self.rung_entry_progress = self.progress_signal

    def maybe_advance_stage(self, outcome_history, episodes_completed):
        """Mirror-phase stage gate. Returns (new_stage, reason), or None.

          "gate"     the 100-episode window cleared `win_rate_threshold` (mastery)
          "plateau"  the long trend stopped improving while staying above
                     PLATEAU_MIN_WIN_RATE (convergence)

        Mirror phase only, or it would fight the per-deck ladder in
        random_opponent.
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
                # Trend on the unregulated signal; floor on the readable rate.
                trend = self._improvement_signal(long_mean)
                if trend > self.best_rung_mean + PLATEAU_IMPROVEMENT:
                    self.best_rung_mean = trend
                    self.best_rung_episode = episodes_completed
                elif (episodes_completed - self.best_rung_episode
                        >= PLATEAU_PATIENCE_EPISODES
                        and trend >= PLATEAU_MIN_WIN_RATE
                        # A rung that is making the agent worse also "has not
                        # improved"; that is a regression, which the demotion
                        # path owns.
                        and (self.rung_entry_progress is None
                             or trend >= self.rung_entry_progress
                             - REGRESSION_MARGIN)):
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

    def floor_alarm(self, outcome_history, episodes_completed):
        """A message when rung 0 has stopped producing wins, else None.

        At most once per FLOOR_ALARM_PATIENCE_EPISODES. Above rung 0 the stall
        valve owns the case.
        """
        if self.phase != "mirror" or self.stage > 0:
            return None
        if episodes_completed - self.stage_start_episode < FLOOR_ALARM_PATIENCE_EPISODES:
            return None
        if episodes_completed - self._last_floor_alarm < FLOOR_ALARM_PATIENCE_EPISODES:
            return None
        win_rate = window_win_rate(outcome_history)
        if win_rate is None or win_rate > FLOOR_ALARM_WIN_RATE:
            return None
        self._last_floor_alarm = episodes_completed
        return (f"win rate {win_rate:.2f} at the BOTTOM rung for "
                f"{episodes_completed - self.stage_start_episode} episodes. There "
                f"is no easier teacher to demote to, so this will not correct "
                f"itself: the opponent is not what is wrong. Check the reward "
                f"magnitudes, the masks and the deck before waiting longer.")

    def maybe_demote_stage(self, outcome_history, episodes_completed):
        """Mirror-phase demotion. Returns the new stage, or None.

        Three triggers: a catastrophic win rate, a rung converged below the
        plateau floor (the backstop), and a regression below the rung's entry
        level. Mirror phase only; the random-deck ladder rotates decks instead.
        Rung 0 is a floor: losing to a rules-only teacher means the teacher is
        not the problem.
        """
        if self.phase != "mirror" or self.stage <= 0:
            return None
        if episodes_completed - self.stage_start_episode < STALL_PATIENCE_EPISODES:
            return None
        win_rate = window_win_rate(outcome_history)
        long_mean = self.rung_mean()
        catastrophic = win_rate is not None and win_rate <= STALL_WIN_RATE
        # Backstop. Reads time since the last improvement, which
        # `maybe_advance_stage` refreshes, so it must run first (train.py does;
        # pinned by
        # test_a_still_improving_rung_is_never_moved_by_the_backstop). The
        # floor reads the unregulated signal, since PFSP pins the readable rate
        # low on an improving agent.
        floor_signal = self._improvement_signal(long_mean)
        capped_out = (floor_signal is not None
                      and floor_signal < PLATEAU_MIN_WIN_RATE
                      and episodes_completed - self.best_rung_episode
                      >= MAX_EPISODES_PER_RUNG)

        # Over-promotion shows as a regression from the rung's entry level, not
        # as a low level.
        regressed = (self.progress_signal is not None
                     and self.rung_entry_progress is not None
                     and self.progress_signal
                     < self.rung_entry_progress - REGRESSION_MARGIN)
        if not (catastrophic or capped_out or regressed):
            return None
        # Name the trigger for the caller's log line.
        self.last_demotion_reason = (
            f"win rate {win_rate:.2f} at or below the {STALL_WIN_RATE:.0%} "
            f"catastrophe floor" if catastrophic else
            f"progress fell to {self.progress_signal:.2f} from {self.rung_entry_progress:.2f} "
            f"at this rung -- the rung is making the agent worse, not merely "
            f"failing to help (regression, not catastrophe)" if regressed else
            f"500-episode mean {long_mean:.2f} below the "
            f"{PLATEAU_MIN_WIN_RATE:.0%} floor with no improvement for "
            f"{MAX_EPISODES_PER_RUNG} episodes (backstop, not catastrophe)")
        self.stage -= 1
        self.demotions += 1
        # Judge the new rung on fresh episodes, and reset the plateau tracker
        # so a demotion cannot immediately read as a plateau.
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
            # Re-boost exploration for the harder version of the same deck.
            self.stage_start_episode = episodes_completed
            return ("advance", self.deck_stage)
        return None

    def state_dict(self):
        return {
            "curriculum_stage": self.stage,
            "curriculum_demotions": self.demotions,
            "curriculum_plateau_advances": self.plateau_advances,
            #: Which teacher table this index refers to. Its absence marks a
            #: checkpoint from before the 11-rung table, so the legacy remap
            #: runs exactly once.
            "teacher_table_size": len(CURRICULUM_STAGES),
            "stage_start_episode": self.stage_start_episode,
            "phase": self.phase,
            "deck_curriculum_stage": self.deck_stage,
            "phase_deck_episode_start": self.deck_episode_start,
            "random_phase_episode_start": self.random_phase_episode_start,
            "current_random_deck": self.current_random_deck,
            # The plateau/regression tracker, persisted so a resume does not
            # restart the plateau window.
            "rung_history": list(self.rung_history),
            "best_rung_mean": self.best_rung_mean,
            "best_rung_episode": self.best_rung_episode,
            "rung_entry_progress": self.rung_entry_progress,
            "progress_signal": self.progress_signal,
        }

    def load_state_dict(self, state):
        """Restore, tolerating checkpoints written before a key existed.

        `random_phase_episode_start` falls back to `phase_deck_episode_start`,
        not 0: defaulting to 0 would make the budget look spent and hand off at
        once.
        """
        # A saved stage indexes the teacher table live when it was written.
        # Unstamped checkpoints predate the 6 -> 11 rung change and are
        # remapped by horizon; read raw, old stage 3 (5 s) would silently
        # become rung 3 (2 s).
        from python_ai.opponents.teacher import remap_legacy_stage
        saved_table = state.get("teacher_table_size")
        stage = int(state["curriculum_stage"])
        if saved_table is None:
            remapped = remap_legacy_stage(stage)
            if remapped != stage:
                print(f">>> Curriculum: checkpoint stage {stage} was written "
                      f"against the 6-rung teacher table; "
                      f"remapped to rung {remapped} (same lookahead horizon).")
            stage = remapped
        elif saved_table != len(CURRICULUM_STAGES):
            # Only the unstamped 6 -> 11 migration is known; any other table
            # change needs its own. Keep the index and say so.
            clamped = min(stage, len(CURRICULUM_STAGES) - 1)
            print(f">>> Curriculum: checkpoint stage {stage} was written against "
                  f"a {saved_table}-rung table and this one has "
                  f"{len(CURRICULUM_STAGES)}. NOT remapping (remap_legacy_stage "
                  f"only knows the 6->11 transition); resuming at rung {clamped}. "
                  f"Verify the rung by hand.")
            stage = clamped
        self.stage = stage
        self.demotions = state.get("curriculum_demotions", 0)
        self.plateau_advances = state.get("curriculum_plateau_advances", 0)
        self.stage_start_episode = state["stage_start_episode"]
        # Legacy checkpoints without the tracker keys get a fresh one.
        self._reset_rung_tracking(self.stage_start_episode)
        if "rung_history" in state:
            self.rung_history.extend(state["rung_history"])
            self.best_rung_mean = float(state.get("best_rung_mean", self.best_rung_mean))
            self.best_rung_episode = int(state.get("best_rung_episode", self.best_rung_episode))
            self.rung_entry_progress = state.get("rung_entry_progress", self.rung_entry_progress)
            self.progress_signal = state.get("progress_signal", self.progress_signal)
        self.phase = state.get("phase", "mirror")
        self.deck_stage = state.get("deck_curriculum_stage", 0)
        self.deck_episode_start = state.get("phase_deck_episode_start", 0)
        self.random_phase_episode_start = state.get(
            "random_phase_episode_start", self.deck_episode_start)
        self.current_random_deck = state.get("current_random_deck", None)


def new_outcome_window(maxlen=OUTCOME_WINDOW):
    return deque(maxlen=maxlen)
