"""The phase-1 curriculum state machine; its order is the load-bearing part."""
import pytest

from python_ai.rl.curriculum import (
    CURRICULUM_STAGES, OUTCOME_WINDOW, PLATEAU_IMPROVEMENT,
    PLATEAU_MIN_WIN_RATE, PLATEAU_PATIENCE_EPISODES, PLATEAU_WINDOW,
    STAGE_WIN_RATE_GATE, CurriculumManager, new_outcome_window,
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


# --- the stages ---
def test_stages_are_competence_not_economy():
    assert len(CURRICULUM_STAGES) == 11
    for s in CURRICULUM_STAGES:
        assert "opp_elixir_multiplier" not in s, (
            "a curriculum stage must never carry an elixir multiplier again")
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in CURRICULUM_STAGES] == list(range(11))
    assert ([s["win_rate_threshold"] for s in CURRICULUM_STAGES]
            == [STAGE_WIN_RATE_GATE] * 10 + [None])


def test_the_gate_was_lowered_WITH_the_ladder_and_not_alone():
    """The lower gate and the finer ladder are one change: 0.80 across eleven
    rungs is harder than the six-rung ladder it replaced, and a lower gate
    without the split just promotes faster into the same cliff.
    """
    assert len(CURRICULUM_STAGES) >= 11
    assert STAGE_WIN_RATE_GATE < 0.80
    # Still a winning margin: at or below 0.50 it would advance an agent losing
    # the matchup.
    assert STAGE_WIN_RATE_GATE > 0.50


def test_stage_count_matches_the_teacher_ladder():
    """Drift would index past the end of TEACHER_STAGES, or stop a rung short of
    the top.
    """
    from python_ai.opponents.teacher import TEACHER_STAGES
    assert len(CURRICULUM_STAGES) == len(TEACHER_STAGES)


# --- the win rate ---
def test_a_partial_window_yields_no_win_rate():
    """The gates must not fire on a partial window; None makes that structural.
    """
    assert window_win_rate(_window(12)) is None
    assert window_win_rate(_full_window(1.0)) == 1.0


def test_the_gate_reads_RAW_win_rate_not_decisive():
    """40 win / 13 loss / 47 draw is 75% decisive but a 40% win rate; a high draw
    rate must not advance a mediocre bot.
    """
    w = _window(40, 13, 47)
    assert window_win_rate(w) == pytest.approx(0.40)


# --- transitions ---
def test_the_stage_gate_needs_a_full_window_at_or_above_threshold():
    m = manager()
    below = round((STAGE_WIN_RATE_GATE - 0.05) * OUTCOME_WINDOW) / OUTCOME_WINDOW
    assert m.maybe_advance_stage(_window(90), 100) is None      # not full
    assert m.maybe_advance_stage(_full_window(below), 100) is None
    assert (m.maybe_advance_stage(_full_window(STAGE_WIN_RATE_GATE), 100)
            == (1, "gate"))
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
    """The phase gate runs before the stage gate. The stage gate clears the
    window, so run first it would consume a window satisfying both and starve
    the phase transition. Shown here by running them in the wrong order.
    """
    m = manager()
    m.stage = 4
    w = _full_window(0.90)          # clears both gates at once
    assert m.maybe_advance_stage(w, 100) == (5, "gate")
    assert len(w) == 0
    assert m.maybe_enter_random_phase(w, 100) is None
    assert m.phase == "mirror"

    # ...and in the right order the phase transition happens.
    m2 = manager()
    m2.stage = 4
    w2 = _full_window(0.90)
    assert m2.maybe_enter_random_phase(w2, 100) is not None
    assert m2.phase == "random_opponent"


def test_the_stage_ladder_is_inert_during_the_random_phase():
    """During the random phase the mirror ladder must not escalate and fight the
    per-deck ladder.
    """
    m = manager()
    m.phase = "random_opponent"
    m.stage = 2
    assert m.maybe_advance_stage(_full_window(1.0), 100) is None
    assert m.stage == 2


# --- the per-deck ladder ---
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
    """A pathological draw must not stall the whole phase budget on one deck.
    """
    m = manager(max_episodes_per_deck=50)
    m.phase = "random_opponent"
    m.deck_episode_start = 0
    kind, reason = m.step_random_deck_curriculum(_window(3), 50)
    assert kind == "rotate" and "safety cap" in reason


def test_the_budget_is_measured_from_when_the_PHASE_started():
    """The budget counts from the phase start, not episode 0."""
    m = manager(random_opponent_budget=500)
    assert not m.budget_exhausted(10_000)         # still in the mirror phase
    m.phase = "random_opponent"
    m.random_phase_episode_start = 10_000
    assert not m.budget_exhausted(10_400)
    assert m.budget_exhausted(10_500)


# --- resume ---
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
    m.load_state_dict({"curriculum_stage": 2, "stage_start_episode": 40,
                       "teacher_table_size": len(CURRICULUM_STAGES)})
    assert m.phase == "mirror" and m.stage == 2


def test_a_pre_2026_09_03_stage_is_remapped_by_HORIZON_not_by_index():
    """A saved stage indexes the table live when it was saved; the 6 -> 11 rung
    change would silently demote an old index (old stage 3 is 5 s, new rung 3
    is 2 s). A missing `teacher_table_size` stamp identifies an old checkpoint,
    so the remap runs exactly once.
    """
    from python_ai.opponents.teacher import TEACHER_STAGES

    for legacy, horizon in enumerate([0, 10, 30, 50, 70, 100]):
        m = manager()
        m.load_state_dict({"curriculum_stage": legacy, "stage_start_episode": 0})
        assert TEACHER_STAGES[m.stage]["horizon_ticks"] == horizon, (
            f"legacy stage {legacy} must resume at the SAME lookahead")

    # A checkpoint that carries the stamp is left alone.
    m = manager()
    m.load_state_dict({"curriculum_stage": 3, "stage_start_episode": 0,
                       "teacher_table_size": len(CURRICULUM_STAGES)})
    assert m.stage == 3


def test_random_phase_start_falls_back_to_the_deck_start_not_to_zero():
    """A 0 default would make the elapsed budget the full episode count and hand
    off to pipeline 2 on resume.
    """
    m = manager()
    m.load_state_dict({"curriculum_stage": 5, "stage_start_episode": 0,
                       "phase": "random_opponent",
                       "phase_deck_episode_start": 30_000})
    assert m.random_phase_episode_start == 30_000
    assert not m.budget_exhausted(31_000)


# --- the stall / demotion valve ---
# Promotion is optimistic: the gate is re-tested every episode over overlapping
# windows, an optional-stopping test whose effective threshold sits below its
# nominal one. A run promoted past its competence needs a way back. The valve
# is patient and set far below any healthy win rate, so a merely hard stage
# never trips it.

def test_a_dead_run_steps_back_down_a_rung():
    m = manager()
    m.stage = 3
    m.stage_start_episode = 0
    assert m.maybe_demote_stage(_full_window(0.02), 5000) == 2
    assert m.stage == 2


def test_a_hard_new_stage_is_given_time_before_demoting():
    """A stage just entered is supposed to be hard; reacting within one window
    would demote every real step up.
    """
    m = manager()
    m.stage = 3
    m.stage_start_episode = 4900
    assert m.maybe_demote_stage(_full_window(0.02), 5000) is None
    assert m.stage == 3


def test_a_healthy_run_is_never_demoted():
    m = manager()
    m.stage = 3
    m.stage_start_episode = 0
    assert m.maybe_demote_stage(_full_window(0.55), 5000) is None


def test_stage_zero_has_nowhere_to_fall_to():
    """Losing at stage 0 means the teacher is not the problem; there is nowhere to
    demote to.
    """
    m = manager()
    m.stage = 0
    m.stage_start_episode = 0
    assert m.maybe_demote_stage(_full_window(0.00), 9000) is None
    assert m.stage == 0


def test_a_partial_window_never_demotes():
    """No verdict on an unfilled window, as for promotion."""
    m = manager()
    m.stage = 2
    m.stage_start_episode = 0
    assert m.maybe_demote_stage(_window(0, 10), 5000) is None


def test_the_random_phase_uses_its_own_valve_not_this_one():
    """`step_random_deck_curriculum` already rotates a deck it cannot beat; two
    valves on one ladder would fight.
    """
    m = manager()
    m.phase = "random_opponent"
    m.stage = 3
    m.stage_start_episode = 0
    assert m.maybe_demote_stage(_full_window(0.00), 5000) is None


def test_demoting_clears_the_window_and_restarts_the_entropy_clock():
    """The demoted stage is judged on fresh episodes, and exploration re-boosts
    for the changed opponent.
    """
    m = manager()
    m.stage = 4
    m.stage_start_episode = 0
    w = _full_window(0.01)
    m.maybe_demote_stage(w, 7000)
    assert len(w) == 0
    assert m.stage_start_episode == 7000


def test_demotions_are_counted_and_survive_a_checkpoint_roundtrip():
    """A demoted run's rung is not evidence of competence, and that must survive a
    resume.
    """
    m = manager()
    m.stage = 2
    m.stage_start_episode = 0
    m.maybe_demote_stage(_full_window(0.00), 6000)
    assert m.demotions == 1

    restored = manager()
    restored.load_state_dict(m.state_dict())
    assert restored.demotions == 1
    assert restored.stage == 1


def test_a_legacy_checkpoint_without_the_demotion_key_loads_as_zero():
    m = manager()
    state = m.state_dict()
    del state["curriculum_demotions"]
    m.load_state_dict(state)
    assert m.demotions == 0
