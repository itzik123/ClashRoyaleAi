"""The phase-1 curriculum state machine.

Its ORDER is the load-bearing part and was previously untestable: everything
lived inside `train_ppo()`, so the suite parsed `train.py` with `ast` to recover
CURRICULUM_STAGES and could not exercise a single transition.
"""
import pytest

from python_ai.rl.curriculum import (
    CURRICULUM_STAGES, OUTCOME_WINDOW, CurriculumManager, new_outcome_window,
    window_win_rate,
)


def _window(wins, losses=0, draws=0):
    w = new_outcome_window()
    w.extend([1] * wins + [-1] * losses + [0] * draws)
    return w


def _full_window(win_rate):
    wins = round(win_rate * OUTCOME_WINDOW)
    return _window(wins, OUTCOME_WINDOW - wins)


def manager(**kwargs):
    defaults = dict(entry_win_rate=0.60, min_stage_for_phase2=4,
                    phase2_win_rate_gate=0.80, max_episodes_per_deck=1250,
                    random_opponent_budget=5000)
    defaults.update(kwargs)
    return CurriculumManager(**defaults)


# --------------------------------------------------------------- the stages --
def test_stages_are_competence_not_economy():
    assert len(CURRICULUM_STAGES) == 6
    for s in CURRICULUM_STAGES:
        assert "opp_elixir_multiplier" not in s, (
            "a curriculum stage must never carry an elixir multiplier again")
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in CURRICULUM_STAGES] == [0, 1, 2, 3, 4, 5]
    assert [s["win_rate_threshold"] for s in CURRICULUM_STAGES] == [0.8] * 5 + [None]


def test_stage_count_matches_the_teacher_ladder():
    """Drift here would index past the end of TEACHER_STAGES -- or, worse,
    silently stop one rung short of the top."""
    from python_ai.opponents.teacher import TEACHER_STAGES
    assert len(CURRICULUM_STAGES) == len(TEACHER_STAGES)


# ------------------------------------------------------------- the win rate --
def test_a_partial_window_yields_no_win_rate():
    """The gates must not fire on 12 episodes. Returning None rather than a
    small-sample rate is what makes that structural."""
    assert window_win_rate(_window(12)) is None
    assert window_win_rate(_full_window(1.0)) == 1.0


def test_the_gate_reads_RAW_win_rate_not_decisive():
    """40 win / 13 loss / 47 draw reads as 75% DECISIVE while only winning 40%
    of games -- a high draw rate would otherwise let a mediocre bot advance."""
    w = _window(40, 13, 47)
    assert window_win_rate(w) == pytest.approx(0.40)


# ------------------------------------------------------------- transitions --
def test_the_stage_gate_needs_a_full_window_at_or_above_threshold():
    m = manager()
    assert m.maybe_advance_stage(_window(90), 100) is None      # not full
    assert m.maybe_advance_stage(_full_window(0.79), 100) is None
    assert m.maybe_advance_stage(_full_window(0.80), 100) == 1
    assert m.stage == 1


def test_advancing_a_stage_clears_the_window_and_restarts_the_entropy_clock():
    m = manager()
    w = _full_window(0.85)
    m.maybe_advance_stage(w, 4321)
    assert len(w) == 0
    assert m.stage_start_episode == 4321


def test_the_final_stage_does_not_auto_advance():
    m = manager()
    m.stage = len(CURRICULUM_STAGES) - 1
    assert m.maybe_advance_stage(_full_window(1.0), 100) is None
    assert m.stage == len(CURRICULUM_STAGES) - 1


def test_the_phase_gate_requires_the_minimum_stage():
    m = manager()
    m.stage = 3
    assert m.maybe_enter_random_phase(_full_window(1.0), 100) is None
    m.stage = 4
    assert m.maybe_enter_random_phase(_full_window(1.0), 100) == 1.0
    assert m.phase == "random_opponent"


def test_the_phase_gate_must_be_evaluated_BEFORE_the_stage_gate():
    """THE ORDERING, and it is load-bearing.

    The stage gate clears the outcome window when it fires. If it ran first, a
    window satisfying BOTH gates would always be consumed by the stage advance
    and the phase transition could never see it -- reaching phase 2 would then
    need an extra full window at the harder stage, which is exactly the
    behaviour the ordering exists to remove.

    Simulated here by running them in the WRONG order and showing the phase
    transition is starved.
    """
    m = manager()
    m.stage = 4
    w = _full_window(0.90)          # clears both gates at once
    assert m.maybe_advance_stage(w, 100) == 5
    assert len(w) == 0
    assert m.maybe_enter_random_phase(w, 100) is None
    assert m.phase == "mirror"

    # ...and in the RIGHT order the phase transition happens.
    m2 = manager()
    m2.stage = 4
    w2 = _full_window(0.90)
    assert m2.maybe_enter_random_phase(w2, 100) is not None
    assert m2.phase == "random_opponent"


def test_the_stage_ladder_is_inert_during_the_random_phase():
    """Unguarded, this block would keep escalating `stage` during phase 2 and
    fight the per-deck ladder for control of the opponent."""
    m = manager()
    m.phase = "random_opponent"
    m.stage = 2
    assert m.maybe_advance_stage(_full_window(1.0), 100) is None
    assert m.stage == 2


# ------------------------------------------------------- the per-deck ladder --
def test_a_deck_advances_its_own_ladder_from_zero():
    m = manager()
    m.phase = "random_opponent"
    assert m.step_random_deck_curriculum(_full_window(0.85), 10) == ("advance", 1)
    assert m.deck_stage == 1
    assert m.stage == 0, "phase 1's own stage must not move"


def test_a_mastered_deck_rotates():
    m = manager()
    m.phase = "random_opponent"
    m.deck_stage = len(CURRICULUM_STAGES) - 1
    kind, reason = m.step_random_deck_curriculum(_full_window(0.85), 10)
    assert kind == "rotate" and "mastered" in reason
    assert m.deck_stage == 0


def test_the_safety_valve_rotates_a_deck_that_never_converges():
    """A pathological draw could otherwise stall the whole phase budget on ONE
    deck -- the precise opposite of what the phase is for."""
    m = manager(max_episodes_per_deck=50)
    m.phase = "random_opponent"
    m.deck_episode_start = 0
    kind, reason = m.step_random_deck_curriculum(_window(3), 50)
    assert kind == "rotate" and "safety cap" in reason


def test_the_budget_is_measured_from_when_the_PHASE_started():
    """Not from episode 0: how much random-deck exposure is enough has nothing
    to do with how many episodes the mirror curriculum happened to take."""
    m = manager(random_opponent_budget=500)
    assert not m.budget_exhausted(10_000)         # still in the mirror phase
    m.phase = "random_opponent"
    m.random_phase_episode_start = 10_000
    assert not m.budget_exhausted(10_400)
    assert m.budget_exhausted(10_500)


# ------------------------------------------------------------------ resume --
def test_teacher_stage_resolves_the_ladder_that_actually_applies():
    m = manager()
    m.stage = 3
    assert m.teacher_stage == CURRICULUM_STAGES[3]["teacher_stage"]
    m.phase = "random_opponent"
    m.deck_stage = 1
    assert m.teacher_stage == CURRICULUM_STAGES[1]["teacher_stage"], (
        "in phase 2 the DECK's own rung applies, not phase 1's")


def test_state_survives_a_checkpoint_roundtrip():
    m = manager()
    m.stage, m.deck_stage, m.phase = 4, 2, "random_opponent"
    m.current_random_deck = [1, 2, 3, 4, 5, 6, 7, 8]
    m.deck_episode_start, m.random_phase_episode_start = 900, 800
    m.stage_start_episode = 700
    restored = manager()
    restored.load_state_dict(m.state_dict())
    assert restored.state_dict() == m.state_dict()


def test_a_legacy_checkpoint_without_the_phase_key_resumes_into_mirror():
    m = manager()
    m.load_state_dict({"curriculum_stage": 2, "stage_start_episode": 40})
    assert m.phase == "mirror" and m.stage == 2


def test_random_phase_start_falls_back_to_the_deck_start_not_to_zero():
    """Defaulting to 0 would make the elapsed budget look like the full episode
    count and hand off to pipeline 2 immediately on resume."""
    m = manager()
    m.load_state_dict({"curriculum_stage": 5, "stage_start_episode": 0,
                       "phase": "random_opponent",
                       "phase_deck_episode_start": 30_000})
    assert m.random_phase_episode_start == 30_000
    assert not m.budget_exhausted(31_000)
