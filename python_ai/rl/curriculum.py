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
from collections import deque

#: Six rungs, indexing `opponents.teacher.TEACHER_STAGES`. The gate (raw win
#: rate >= 0.80 over a full 100-episode window) is unchanged from the elixir era
#: and so is the load-bearing ordering around it: the PHASE transition is
#: evaluated BEFORE stage advancement, because the stage gate clears the outcome
#: window when it fires and would otherwise always consume a window that
#: satisfies both.
CURRICULUM_STAGES = [
    {"teacher_stage": 0, "win_rate_threshold": 0.80},
    {"teacher_stage": 1, "win_rate_threshold": 0.80},
    {"teacher_stage": 2, "win_rate_threshold": 0.80},
    {"teacher_stage": 3, "win_rate_threshold": 0.80},
    {"teacher_stage": 4, "win_rate_threshold": 0.80},
    {"teacher_stage": 5, "win_rate_threshold": None},   # no further auto-advance
]

#: Window the gates read. 100 episodes is a +/-0.1 sampling band on a win rate,
#: which is the coarsest resolution the thresholds below are meaningful at.
OUTCOME_WINDOW = 100


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

    def maybe_advance_stage(self, outcome_history, episodes_completed):
        """Mirror-phase stage gate. Returns the new stage, or None.

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
        win_rate = window_win_rate(outcome_history)
        if win_rate is None or win_rate < threshold:
            return None
        self.stage += 1
        outcome_history.clear()
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
        self.stage = state["curriculum_stage"]
        self.stage_start_episode = state["stage_start_episode"]
        self.phase = state.get("phase", "mirror")
        self.deck_stage = state.get("deck_curriculum_stage", 0)
        self.deck_episode_start = state.get("phase_deck_episode_start", 0)
        self.random_phase_episode_start = state.get(
            "random_phase_episode_start", self.deck_episode_start)
        self.current_random_deck = state.get("current_random_deck", None)


def new_outcome_window(maxlen=OUTCOME_WINDOW):
    return deque(maxlen=maxlen)
