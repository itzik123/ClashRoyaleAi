"""The plateau valve: an exit for a rung the agent has stopped improving on.

The ladder's gate handles mastery and the stall valve handles catastrophe; this
covers the band in between, where a curriculum run spends most of its time. The
valve's refusals are pinned too: a plateau rule that fires on a losing agent
promotes past competence.
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


# --- it fires ---
def test_a_converged_competitive_rung_advances_without_ever_clearing_the_gate():
    """A converged, competitive rung advances without ever clearing the gate.
    """
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
    """A plateau advance ("this stopped teaching") is a weaker claim than a gate
    advance ("this was beaten"), and the logs must distinguish them.
    """
    m = manager()
    _fill_rung(m, 0.55)
    # The first full window establishes the baseline, so a fire takes two
    # evaluations.
    m.maybe_advance_stage(_below_gate(), 0)
    _, reason = m.maybe_advance_stage(_below_gate(),
                                      PLATEAU_PATIENCE_EPISODES + 1)
    assert reason == "plateau"
    assert m.state_dict()["curriculum_plateau_advances"] == 1


def test_it_would_have_ended_the_2026_08_28_stall_within_its_patience():
    """A long 0.15-0.60 oscillation, replayed: the valve must end it early."""
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


# --- it refuses ---
def test_a_LOSING_rung_never_plateau_advances():
    """Below the floor the agent is losing to the rung, not held back by it;
    promoting there would fight the stall valve.
    """
    m = manager()
    _fill_rung(m, PLATEAU_MIN_WIN_RATE - 0.15)
    for ep in range(PLATEAU_PATIENCE_EPISODES * 3):
        assert m.maybe_advance_stage(_below_gate(), ep) is None
    assert m.stage == 3


def test_a_still_improving_rung_is_not_advanced():
    """Patience resets on every real improvement, or a steadily learning run is
    promoted mid-climb.
    """
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
    """A trend over the plateau window needs that many episodes; firing sooner
    would be a coin flip on a fresh rung.
    """
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


# --- interaction & state ---
def test_a_rung_change_resets_the_trend():
    """A trend carried across a rung change describes an opponent no longer
    played, and would let a demotion immediately look like a plateau below.
    """
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
    """Mastery is reported as mastery: a window that clears the gate advances with
    reason "gate".
    """
    m = manager()
    _fill_rung(m, 0.70)
    w = new_outcome_window()
    wins = round(STAGE_WIN_RATE_GATE * w.maxlen) + 5
    w.extend([1] * wins + [-1] * (w.maxlen - wins))
    stage, reason = m.maybe_advance_stage(w, PLATEAU_PATIENCE_EPISODES * 9)
    assert reason == "gate"
    assert m.plateau_advances == 0


def test_the_plateau_series_is_not_the_gate_window():
    """The gate clears its window on every advance and demotion, so the plateau
    detector owns a separate series.
    """
    m = manager()
    w = new_outcome_window()
    for _ in range(200):
        m.note_outcome(1)
        w.append(1)
    w.clear()                                # what an advance does to the gate
    assert len(m.rung_history) == 200, (
        "clearing the gate window must not touch the plateau series")


# --- the backstop ---
def test_a_rung_converged_BELOW_the_floor_is_demoted_not_left_forever():
    """A rung converged below the floor but above the stall valve is demoted
    rather than held forever.
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
    """The backstop measures time since the last improvement, not time at the
    rung, so it never cuts off an agent still climbing.
    """
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
    """Whatever the level, a rung the agent has stopped improving at is left: up
    if competitive, down if not. Swept across the whole range so both branches
    are exercised.
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


# --- the improvement test reads a signal PFSP does not regulate ---

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
    """PFSP weights decks by (1-wr)^2, which pins the readable win rate near the
    agent's worst matchups however much it improves; the improvement test must
    read an unregulated progress signal.
    """
    mgr = manager(stage=2)
    fired = _flat_readable_rising_progress(
        mgr, 8000, lambda ep: 0.30 + 0.00003 * ep)   # 0.30 -> 0.54
    assert fired == [], (
        f"advanced {len(fired)} times while the agent was improving: {fired}")


def test_a_genuine_plateau_still_advances_on_the_progress_signal():
    """When the progress signal genuinely stops moving, the rung still ends."""
    mgr = manager(stage=2)
    fired = _flat_readable_rising_progress(mgr, 8000, lambda ep: 0.50)
    assert fired, "a genuinely flat progress signal must still plateau out"
    assert fired[0][1][1] == "plateau"


def test_without_a_progress_signal_the_old_behaviour_is_unchanged():
    """The mirror path has no deck estimates, so it reads the long window as
    before; the fallback keeps the change additive.
    """
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
    """The backstop's floor is a level question, so it reads the same unregulated
    progress signal: a strong agent whose readable rate is pinned low by PFSP
    must not be demoted.
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
    """A rung genuinely under the floor on the unregulated signal, and not
    improving, is still demoted.
    """
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


def test_a_rung_that_makes_the_agent_WORSE_is_demoted():
    """Over-promotion shows as a regression, not a low level: a sustained fall
    from the best this rung has seen demotes even while the agent stays above
    the floor.
    """
    mgr = manager(stage=4)
    hist = new_outcome_window()
    demoted = None
    trace = [0.637, 0.626, 0.608, 0.599, 0.596, 0.591, 0.583]   # a measured series
    for i, ep in enumerate(range(0, 14000, 200)):
        for k in range(200):
            w = k % 2                      # 0.50 in any trailing window (a block of
            mgr.note_outcome(w)            # wins then losses would leave the 100-deep
            hist.append(w)                 # window all zeros and trip the stall valve)

        mgr.note_progress(trace[min(i, len(trace) - 1)])
        mgr.maybe_advance_stage(hist, ep)
        if mgr.maybe_demote_stage(hist, ep) is not None:
            demoted = ep
            break
    assert demoted is not None, (
        "a rung that took the agent from 0.637 to 0.583 was never demoted")


def test_ordinary_noise_around_a_plateau_is_not_a_regression():
    """Contrast: wobbling inside the noise band must not demote, or every plateau
    becomes a demotion.
    """
    mgr = manager(stage=4)
    hist = new_outcome_window()
    wobble = [0.640, 0.637, 0.641, 0.638, 0.642, 0.639, 0.640]
    for i, ep in enumerate(range(0, 14000, 200)):
        for k in range(200):
            w = k % 2                      # 0.50 in any trailing window (a block of
            mgr.note_outcome(w)            # wins then losses would leave the 100-deep
            hist.append(w)                 # window all zeros and trip the stall valve)

        mgr.note_progress(wobble[i % len(wobble)])
        mgr.maybe_advance_stage(hist, ep)
        assert mgr.maybe_demote_stage(hist, ep) is None, (
            f"demoted at ep {ep} on noise inside a plateau")
