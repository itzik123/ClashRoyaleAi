"""The plateau valve: the exit the 2026-08-28 run did not have.

THE MEASUREMENT THIS ENCODES, off runs/phase4/train-20260828-rolling-spells.log:
episodes 9,640 -> 32,680 -- 23,040 consecutive episodes, ~24 hours at the
measured 943 ep/h -- were spent at curriculum stage 3, win rate oscillating
0.15-0.60, with ZERO advances. The ladder had a gate for MASTERY (>= 0.80) and a
valve for CATASTROPHE (<= 0.10 sustained) and nothing at all for the band in
between, which is where a curriculum run actually spends its time.

These tests pin the valve AND its refusals. A plateau rule that fires on a
losing agent is worse than no rule: it promotes past competence, which is the
over-promotion the stall valve exists to undo.
"""
import pytest

from python_ai.rl.curriculum import (
    CURRICULUM_STAGES, PLATEAU_IMPROVEMENT, PLATEAU_MIN_WIN_RATE,
    PLATEAU_PATIENCE_EPISODES, PLATEAU_WINDOW, STAGE_WIN_RATE_GATE,
    CurriculumManager, new_outcome_window,
)


def manager(stage=3):
    m = CurriculumManager(entry_win_rate=0.60, min_stage_for_phase2=4,
                          phase2_win_rate_gate=0.80, max_episodes_per_deck=1250,
                          random_opponent_budget=5000)
    m.stage = stage
    return m


def _fill_rung(m, win_rate, n=None):
    """Push `n` episodes at `win_rate` through the plateau series."""
    n = n or PLATEAU_WINDOW
    for i in range(n):
        m.note_outcome(1 if (i % 100) < round(win_rate * 100) else -1)


def _below_gate():
    """A window that cannot fire the ordinary gate."""
    w = new_outcome_window()
    wins = round((STAGE_WIN_RATE_GATE - 0.15) * w.maxlen)
    w.extend([1] * wins + [-1] * (w.maxlen - wins))
    return w


# ------------------------------------------------------------------ it fires --
def test_a_converged_competitive_rung_advances_without_ever_clearing_the_gate():
    """The headline case: the exact shape of the 23,040-episode stall."""
    m = manager()
    start = m.stage
    _fill_rung(m, 0.55)
    # The first full window establishes the best; nothing has stalled yet.
    assert m.maybe_advance_stage(_below_gate(), 0) is None
    # ...and after the patience elapses with no improvement, it advances.
    result = m.maybe_advance_stage(_below_gate(), PLATEAU_PATIENCE_EPISODES + 1)
    assert result == (start + 1, "plateau")
    assert m.plateau_advances == 1


def test_the_plateau_advance_is_reported_separately_from_the_gate():
    """A rung cleared by plateau is a weaker claim than one cleared by the gate
    -- "this stopped teaching", not "this was beaten". A run that plateaued up
    every rung is at the top having won nothing, and the logs must say so."""
    m = manager()
    _fill_rung(m, 0.55)
    # The first full window ESTABLISHES the baseline; the plateau is measured
    # from there, so a fire always takes two evaluations by construction.
    m.maybe_advance_stage(_below_gate(), 0)
    _, reason = m.maybe_advance_stage(_below_gate(),
                                      PLATEAU_PATIENCE_EPISODES + 1)
    assert reason == "plateau"
    assert m.state_dict()["curriculum_plateau_advances"] == 1


def test_it_would_have_ended_the_2026_08_28_stall_within_its_patience():
    """23,040 episodes of a 0.15-0.60 oscillation, replayed."""
    import random
    rng = random.Random(7)
    m = manager()
    fired_at = None
    for ep in range(23040):
        m.note_outcome(1 if rng.random() < 0.45 else -1)
        if m.maybe_advance_stage(_below_gate(), ep) is not None:
            fired_at = ep
            break
    assert fired_at is not None, "the stall must terminate"
    assert fired_at < 4000, (
        f"fired at episode {fired_at}; the whole point is to reclaim the other "
        "~20,000")


# --------------------------------------------------------------- it refuses --
def test_a_LOSING_rung_never_plateau_advances():
    """Below the floor the agent is not being held back by the rung, it is
    losing to it. Promoting there is exactly the over-promotion the stall valve
    exists to undo, and the two would fight."""
    m = manager()
    _fill_rung(m, PLATEAU_MIN_WIN_RATE - 0.15)
    for ep in range(PLATEAU_PATIENCE_EPISODES * 3):
        assert m.maybe_advance_stage(_below_gate(), ep) is None
    assert m.stage == 3


def test_a_still_improving_rung_is_not_advanced():
    """Patience must reset on every real improvement, or a steadily-learning
    run gets promoted mid-climb."""
    m = manager()
    rate = 0.45
    ep = 0
    for _ in range(6):
        _fill_rung(m, rate)
        ep += PLATEAU_PATIENCE_EPISODES - 1
        assert m.maybe_advance_stage(_below_gate(), ep) is None
        rate += PLATEAU_IMPROVEMENT * 3      # clearly improving
    assert m.stage == 3


def test_a_partial_series_cannot_fire_it():
    """A 500-episode trend needs 500 episodes. Firing on 50 would make the
    valve a coin flip on the first noisy stretch of a fresh rung."""
    m = manager()
    _fill_rung(m, 0.60, n=PLATEAU_WINDOW - 1)
    assert m.rung_mean() is None
    assert m.maybe_advance_stage(_below_gate(),
                                 PLATEAU_PATIENCE_EPISODES * 5) is None


def test_the_top_rung_never_plateau_advances_off_the_end():
    m = manager(stage=len(CURRICULUM_STAGES) - 1)
    _fill_rung(m, 0.90)
    assert m.maybe_advance_stage(_below_gate(),
                                 PLATEAU_PATIENCE_EPISODES * 5) is None


# ------------------------------------------------------- interaction & state --
def test_a_rung_change_resets_the_trend():
    """A 500-episode mean carried across a rung change describes an opponent
    that is no longer being played -- and would let a demotion immediately look
    like a plateau on the rung below."""
    m = manager()
    _fill_rung(m, 0.55)
    m.maybe_advance_stage(_below_gate(), 0)
    m.maybe_advance_stage(_below_gate(), PLATEAU_PATIENCE_EPISODES + 1)
    assert m.rung_mean() is None
    assert m.best_rung_mean == 0.0


def test_demotion_also_resets_the_trend():
    m = manager()
    _fill_rung(m, 0.55)
    m.stage_start_episode = 0
    w = new_outcome_window()
    w.extend([-1] * w.maxlen)
    assert m.maybe_demote_stage(w, 99999) == 2
    assert m.rung_mean() is None


def test_the_gate_still_wins_when_both_could_fire():
    """Mastery must be reported as mastery. If a window clears 0.65 the reason
    is "gate", even if the trend has also flattened."""
    m = manager()
    _fill_rung(m, 0.70)
    w = new_outcome_window()
    wins = round(STAGE_WIN_RATE_GATE * w.maxlen) + 5
    w.extend([1] * wins + [-1] * (w.maxlen - wins))
    stage, reason = m.maybe_advance_stage(w, PLATEAU_PATIENCE_EPISODES * 9)
    assert reason == "gate"
    assert m.plateau_advances == 0


def test_the_plateau_series_is_not_the_gate_window():
    """The gate CLEARS its window on every advance and demotion. A detector for
    'has the trend stopped' cannot run on a series that is reset whenever
    anything happens, so it must own a separate one."""
    m = manager()
    w = new_outcome_window()
    for _ in range(200):
        m.note_outcome(1)
        w.append(1)
    w.clear()                                # what an advance does to the gate
    assert len(m.rung_history) == 200, (
        "clearing the gate window must not touch the plateau series")


# ------------------------------------------------------------- the backstop --
def test_a_rung_converged_BELOW_the_floor_is_demoted_not_left_forever():
    """The gap the plateau valve leaves, and the exact shape of the real stall.

    Converged at 0.30: not catastrophic enough for the 0.10 stall valve, not
    competitive enough to promote. Before the backstop this state could hold a
    run indefinitely -- and did, for 23,040 episodes.
    """
    from python_ai.rl.curriculum import MAX_EPISODES_PER_RUNG
    m = manager()
    _fill_rung(m, 0.30)
    w = new_outcome_window()
    w.extend([1] * 30 + [-1] * 70)          # 0.30: above the 0.10 stall valve
    m.maybe_advance_stage(_below_gate(), 0)  # establishes the baseline
    assert m.maybe_advance_stage(_below_gate(), 999) is None
    assert m.maybe_demote_stage(w, 999) is None, "must not fire early"
    assert m.maybe_demote_stage(w, MAX_EPISODES_PER_RUNG + 1) == 2
    assert m.demotions == 1


def test_a_still_improving_rung_is_never_moved_by_the_backstop():
    """The first version of the backstop used elapsed time at the rung and would
    have cut off an agent that was still climbing -- worse than the stall it was
    added to prevent. It measures time since the last IMPROVEMENT instead."""
    from python_ai.rl.curriculum import MAX_EPISODES_PER_RUNG
    m = manager()
    w = new_outcome_window()
    w.extend([1] * 30 + [-1] * 70)
    rate, ep = 0.20, 0
    for _ in range(8):                       # 8 * 3999 episodes, all improving
        _fill_rung(m, rate)
        ep += MAX_EPISODES_PER_RUNG - 1
        assert m.maybe_advance_stage(_below_gate(), ep) is None
        assert m.maybe_demote_stage(w, ep) is None, (
            f"demoted at episode {ep} while still improving")
        rate += PLATEAU_IMPROVEMENT * 2
    assert m.stage == 3


def test_no_rung_can_hold_a_run_that_has_stopped_improving():
    """The property the whole change exists to guarantee, stated once.

    Whatever the agent's level, a rung it has stopped improving at is LEFT --
    up if competitive, down if not. Swept across the whole win-rate range so
    neither branch can be the only one that works.
    """
    from python_ai.rl.curriculum import MAX_EPISODES_PER_RUNG
    for level in (0.05, 0.20, 0.30, 0.42, 0.55, 0.75):
        m = manager()
        _fill_rung(m, level)
        w = new_outcome_window()
        wins = round(level * w.maxlen)
        w.extend([1] * wins + [-1] * (w.maxlen - wins))
        moved_at = None
        for ep in range(0, MAX_EPISODES_PER_RUNG * 2, 50):
            if (m.maybe_advance_stage(_below_gate(), ep) is not None
                    or m.maybe_demote_stage(w, ep) is not None):
                moved_at = ep
                break
        assert moved_at is not None, (
            f"a rung converged at win rate {level} held the run for "
            f"{MAX_EPISODES_PER_RUNG * 2} episodes")
        assert moved_at <= MAX_EPISODES_PER_RUNG + 50


# --------------------------------------------------------------------------
# 2026-09-06: the improvement test must read a signal PFSP does not regulate.
# --------------------------------------------------------------------------

def _flat_readable_rising_progress(mgr, episodes, progress_at):
    """Feed a win rate pinned at 0.50 while the progress signal climbs."""
    hist = new_outcome_window()
    out = []
    for ep in range(0, episodes, 10):
        for i in range(10):
            w = 1 if i < 5 else 0          # exactly 0.50, forever
            mgr.note_outcome(w)
            hist.append(w)
        mgr.note_progress(progress_at(ep))
        r = mgr.maybe_advance_stage(hist, ep)
        if r:
            out.append((ep, r))
    return out


def test_a_regulated_win_rate_does_not_look_like_a_plateau():
    """PFSP weights a deck by (1-wr)^2, so it drives the readable win rate
    toward the agent's worst matchups and PINS it no matter how much the agent
    improves. Measured on the live phase-9 run, ep 83,128 -> 87,540: the
    unweighted per-deck mean went 0.301 -> 0.534 on all sixteen decks while the
    readable rate sat at ~0.50 -- and the valve fired twice, calling the fastest
    learning of the run a plateau.

    So the improvement test must read a quantity PFSP does not regulate.
    """
    mgr = manager(stage=2)
    fired = _flat_readable_rising_progress(
        mgr, 8000, lambda ep: 0.30 + 0.00003 * ep)   # 0.30 -> 0.54, as measured
    assert fired == [], (
        f"advanced {len(fired)} times while the agent was improving: {fired}")


def test_a_genuine_plateau_still_advances_on_the_progress_signal():
    """The anti-stall purpose is preserved: when the signal the agent actually
    moves stops moving, the rung must still end. Otherwise this trades the
    2026-08-28 stall back in."""
    mgr = manager(stage=2)
    fired = _flat_readable_rising_progress(mgr, 8000, lambda ep: 0.50)
    assert fired, "a genuinely flat progress signal must still plateau out"
    assert fired[0][1][1] == "plateau"


def test_without_a_progress_signal_the_old_behaviour_is_unchanged():
    """The mirror path has no deck estimates, so it must keep reading the long
    window exactly as before -- the fallback is what makes this additive."""
    mgr = manager(stage=2)
    hist = new_outcome_window()
    fired = []
    for ep in range(0, 8000, 10):
        for i in range(10):
            w = 1 if i < 5 else 0
            mgr.note_outcome(w)
            hist.append(w)
        r = mgr.maybe_advance_stage(hist, ep)
        if r:
            fired.append((ep, r))
    assert fired and fired[0][1][1] == "plateau", (
        "with no progress signal the valve must behave as it did before")


def test_a_regulated_win_rate_does_not_look_like_over_promotion():
    """The BACKSTOP has the same blind spot the plateau valve had, and the fix
    is the same signal.

    Measured on the live phase-9 run 2026-09-06: the agent sat at an unweighted
    per-deck mean of 0.62 across all sixteen decks -- beating fourteen of them --
    while the PFSP-weighted readable rate was pinned at 0.29, because PFSP
    concentrates sampling on the two it cannot beat. The backstop read 0.29
    against its 0.40 floor and demoted rung 3 -> 2 -> 1, weakening the teacher
    on an agent that was getting stronger. Left alone it walks to rung 0.

    A floor is a LEVEL question, so it needs a level the sampler does not
    regulate. `PLATEAU_MIN_WIN_RATE` is applied to the progress signal for the
    same reason the trend test already is.
    """
    mgr = manager(stage=3)
    hist = new_outcome_window()
    for ep in range(0, 12000, 10):
        for i in range(10):
            w = 1 if i < 3 else 0          # readable rate pinned at 0.30
            mgr.note_outcome(w)
            hist.append(w)
        mgr.note_progress(0.62)            # ...while the agent is plainly strong
        mgr.maybe_advance_stage(hist, ep)  # trainer order: advance, then demote
        assert mgr.maybe_demote_stage(hist, ep) is None, (
            f"demoted at ep {ep} an agent averaging 0.62 across the pool")


def test_a_genuinely_weak_rung_is_still_demoted():
    """The backstop's purpose is not traded away: an agent that really is under
    the floor on the unregulated signal, and not improving, still goes down."""
    mgr = manager(stage=3)
    hist = new_outcome_window()
    demoted = None
    for ep in range(0, 12000, 10):
        for i in range(10):
            w = 1 if i < 3 else 0
            mgr.note_outcome(w)
            hist.append(w)
        mgr.note_progress(0.22)            # weak on every deck, not just the mix
        mgr.maybe_advance_stage(hist, ep)
        if mgr.maybe_demote_stage(hist, ep) is not None:
            demoted = ep
            break
    assert demoted is not None, "a genuinely weak rung must still be demoted"
