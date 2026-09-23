"""Predicting the opponent's elixir is not a memory task, so its error is not a
memory diagnostic.

The opponent's elixir is reconstructible from two scalars the observation
already carries:

  elixir(t) = start + income(t) - spent(t)

`t` is extra scalar 0 and `spent` is extra scalar 2, so least squares with no
recurrence fits it exactly. `income(t)` integrates the 1x/2x/3x elixir
schedule; a single slope on `t` no longer fits.

The one exception is overflow: the 10.0 cap discards income that no scalar
records. A rising error is therefore an overflow detector, still not a memory
diagnostic. The opponent-elixir head this was written about has been replaced
by the next-card task.
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai.envs.gym_wrapper import MicroRoyaleEnv


_CE = clash_royale_env.ClashRoyaleEnv
N_EXTRA = _CE.NUM_EXTRA_SCALARS

#: Forward offsets from the start of the observation, derived from the
#: bindings, so appending a section cannot move them.
_SCALAR_START = _CE.BOARD_WIDTH * _CE.BOARD_HEIGHT * _CE.NUM_CHANNELS
_EXTRA_START = (_SCALAR_START + 1 + _CE.HAND_SIZE
                + _CE.HAND_SIZE * _CE.NUM_CARD_IDS)
#: Ordered as ClashEnv.h appends them: time, own elixir spent, opponent elixir
#: spent, then 6 tower HP readings.
T_IDX, SPENT_SELF_IDX, SPENT_OPP_IDX = (
    _EXTRA_START + 0, _EXTRA_START + 1, _EXTRA_START + 2)


def test_the_extra_scalar_offsets_still_point_at_the_extra_scalars():
    """Guards the offsets: the time scalar is exactly 0.0 on a fresh board and
    rises as the match runs, a property no other column has.
    """
    env = MicroRoyaleEnv({})
    obs = env.reset()
    o = np.asarray(obs[0] if isinstance(obs, tuple) else obs, dtype=np.float32)
    assert o[T_IDX] == 0.0, (
        f"observation[{T_IDX}] is {o[T_IDX]}, expected the time fraction to be "
        "0.0 on a fresh board; the extra-scalar offsets have drifted")
    for _ in range(20):
        res = env.step({"card_index": 4, "target_x": 0.0, "target_y": 0.0})
        obs = res[0]
    o2 = np.asarray(obs[0] if isinstance(obs, tuple) else obs, dtype=np.float32)
    assert o2[T_IDX] > o[T_IDX], (
        "the time scalar did not increase over 20 steps; T_IDX is not "
        "pointing at the time fraction")


#: ELIXIR_REGEN_RATE is private to GameManager and not bound, so it is restated
#: here and measured below.
_REGEN_PER_TICK = 0.035     # GameManager.h: `const float ELIXIR_REGEN_RATE`

#: The bar's ceiling (PlayerState clamps to it) and the env's default
#: max_ticks, which normalises the time scalar. Neither is bound.
_CAP = 10.0
_MAX_TICKS = 3600.0

#: Column indices into the design matrix built by `_collect`.
C_T, C_OPP_SPENT, C_OWN_SPENT, C_BIAS, C_INCOME = 0, 1, 2, 3, 4


def _income(t):
    """Exact cumulative regen from tick 0 to `t`, integrating the phases. Both
    boundaries are read from the engine.
    """
    d, tr = _CE.DOUBLE_ELIXIR_TICK, _CE.TRIPLE_ELIXIR_TICK
    r = _REGEN_PER_TICK
    return (r * min(t, d)
            + 2.0 * r * max(0.0, min(t, tr) - d)
            + 3.0 * r * max(0.0, t - tr))


def _collect(episodes=6, seed=0):
    """Design matrix and target for reconstructing the opponent's elixir.

    Columns: `[t_ticks, opp_spent, own_spent, 1, income(t)]`; `t_ticks` is kept
    to contrast the single-slope basis with the correct one.

    Also returns, per sample, whether the opponent's bar has hit the cap at any
    point so far this episode. Once the cap discards income every later sample
    carries the loss as an offset, so "currently at 10.0" undercounts the
    contamination badly.
    """
    rng = np.random.default_rng(seed)
    X, y, tainted = [], [], []
    for _ in range(episodes):
        env = MicroRoyaleEnv({})
        obs = env.reset()
        done, n, capped = False, 0, False
        while not done and n < 400:
            res = env.step({"card_index": int(rng.integers(0, 5)),
                            "target_x": float(rng.integers(0, 18)),
                            "target_y": float(rng.integers(0, 16))})
            obs, info = res[0], res[-1]
            done = (bool(res[2]) or bool(res[3])) if len(res) == 5 else bool(res[2])
            o = np.asarray(obs[0] if isinstance(obs, tuple) else obs,
                           dtype=np.float32)
            if "opp_elixir" in info:
                elixir = float(np.atleast_1d(info["opp_elixir"])[0])
                if elixir >= _CAP - 1e-3:
                    capped = True
                t_ticks = float(o[T_IDX]) * _MAX_TICKS
                X.append([t_ticks,
                          float(o[SPENT_OPP_IDX]) * _CE.MAX_MATCH_ELIXIR,
                          float(o[SPENT_SELF_IDX]) * _CE.MAX_MATCH_ELIXIR,
                          1.0,
                          _income(t_ticks)])
                y.append(elixir)
                tainted.append(capped)
            n += 1
    return np.asarray(X), np.asarray(y), np.asarray(tainted, dtype=bool)


@pytest.fixture(scope="module")
def samples():
    X, y, tainted = _collect()
    assert len(y) > 500, f"only {len(y)} samples collected; too few to conclude"
    assert (~tainted).sum() > 200, (
        f"only {(~tainted).sum()} overflow-free samples; the reconstruction "
        "tests below are defined on that regime and cannot conclude")
    return X, y, tainted


def _mae(X, y, cols):
    A = X[:, cols]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.abs(y - A @ w).mean())


def test_the_regen_rate_this_file_assumes_is_still_the_engines():
    """`_REGEN_PER_TICK` restates a private constant, so measure it: read the bar
    over a window inside single elixir with nobody spending.
    """
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    env = clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()
    env.set_elixir_for_team(0, 0.0)
    before = env.get_elixir_for_team(0)
    ticks = 20                     # 2 s, entirely within phase 1
    env.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, ticks)
    measured = (env.get_elixir_for_team(0) - before) / ticks
    assert measured == pytest.approx(_REGEN_PER_TICK, abs=1e-6), (
        f"measured {measured} per tick against this file's assumed "
        f"{_REGEN_PER_TICK}; update _REGEN_PER_TICK and re-derive _income")


def test_opponent_elixir_is_an_affine_function_of_present_scalars(samples):
    """On overflow-free samples, ordinary least squares on present scalars
    reconstructs the target exactly.
    """
    X, y, tainted = samples
    ok = ~tainted
    mae = _mae(X[ok], y[ok], [C_INCOME, C_OPP_SPENT, C_BIAS])
    assert mae < 0.05, (
        f"linear reconstruction MAE is {mae:.4f} on overflow-free samples; "
        "elixir = start + integral(rate) - spent is expected to hold to within "
        "float noise there. If this has risen, check whether the engine "
        "changed ELIXIR_REGEN_RATE or the phase schedule (both feed _income), "
        "or whether the extra-scalar ORDER moved -- the offsets here are "
        "positional.")


def test_a_single_slope_no_longer_fits_because_income_is_piecewise(samples):
    """A single slope on `t` no longer fits, because income is piecewise-linear.
    Asserted as a failure so that reverting the phases, or making them
    continuous, is noticed.
    """
    X, y, tainted = samples
    ok = ~tainted
    naive = _mae(X[ok], y[ok], [C_T, C_OPP_SPENT, C_BIAS])
    correct = _mae(X[ok], y[ok], [C_INCOME, C_OPP_SPENT, C_BIAS])
    assert naive > 10 * max(correct, 1e-3), (
        f"the single-slope basis scores {naive:.4f} and the phase-integrated "
        f"one {correct:.4f}. They are supposed to differ by an order of "
        "magnitude: if they no longer do, elixir regen has become single-rate "
        "again and CLAUDE.md's phase section is out of date.")


#: Every card costs exactly 3, so the order of plays cannot change what a side
#: has spent, only when. Five spells that do nothing on empty ground plus three
#: troops; nothing here touches anyone's elixir.
_THREE_COST_DECK = [3, 103, 104, 105, 106, 0, 1, 41]
_THREE = 3.0
_START_ELIXIR = 5.0     # PlayerState's opening bar; checked on tick 0 below


def _passive_board_where_team1_plays_at(play_ticks, until):
    """Team 0 never plays; team 1 plays one 3-cost card at each tick in
    `play_ticks`, stepped one tick at a time. Returns (team-0 observation, team
    1's true elixir) at tick `until`.

    Every play is verified, since an unaffordable one would silently no-op.
    Each uses the next hand slot: a slot is locked for 20 ticks after it
    cycles.
    """
    env = _CE(list(_THREE_COST_DECK), list(_THREE_COST_DECK), 3600)
    env.seed(0)
    hand = _CE.HAND_SIZE
    assert env.get_elixir_for_team(1) == pytest.approx(_START_ELIXIR)
    plays = 0
    for tick in range(until):
        if tick in play_ticks:
            before = env.get_elixir_spent(1)
            # Own frame (9, 5): team 1's back court, where nothing is hit.
            env.step_self_play(hand, 0.0, 0.0, plays % hand, 9.0, 5.0, 1)
            plays += 1
            assert env.get_elixir_spent(1) == pytest.approx(before + _THREE), (
                f"team 1's play at tick {tick} did not happen (bar "
                f"{env.get_elixir_for_team(1):.2f}); the construction is broken")
        else:
            env.step_self_play(hand, 0.0, 0.0, hand, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    return obs, float(env.get_elixir_for_team(1))


def test_overflow_is_what_makes_this_task_non_trivial():
    """Overflow is what makes the task non-trivial: the cap discards income that
    nothing in (t, own spent, opp spent) records.

    Constructed, not sampled: two boards with identical scalars (same tick,
    same spend on both sides) whose opponent holds different elixir, differing
    only in how long it sat on a full bar. A spends as it goes and never caps;
    B banks, caps, then spends the same four cards. No function of those
    scalars can return two values for one input.
    """
    until = 400
    spend_as_you_go = {100, 190, 280, 370}      # bar peaks at 8.95, never caps
    bank_then_spend = {330, 340, 350, 390}      # capped from ~tick 143 to 330
    obs_a, elixir_a = _passive_board_where_team1_plays_at(spend_as_you_go, until)
    obs_b, elixir_b = _passive_board_where_team1_plays_at(bank_then_spend, until)

    for idx, name in ((T_IDX, "time"), (SPENT_SELF_IDX, "own spent"),
                      (SPENT_OPP_IDX, "opponent spent")):
        assert obs_a[idx] == obs_b[idx], (
            f"the {name} scalar differs ({obs_a[idx]} vs {obs_b[idx]}); the two "
            "boards are supposed to be indistinguishable on it")

    # A never capped, so the analytic model reconstructs it: the control.
    model = _START_ELIXIR + _income(until) - 4 * _THREE
    assert elixir_a == pytest.approx(model, abs=2 * _REGEN_PER_TICK), (
        f"the never-capped board holds {elixir_a:.3f}, the model says "
        f"{model:.3f}; the exact reconstruction this file rests on is broken")

    # B lost exactly what it earned while pinned at the cap, from the tick its
    # bar filled to its first play.
    filled_at = (_CAP - _START_ELIXIR) / _REGEN_PER_TICK
    discarded = (min(bank_then_spend) - filled_at) * _REGEN_PER_TICK
    assert elixir_a - elixir_b == pytest.approx(discarded, abs=2 * _REGEN_PER_TICK), (
        f"identical scalars, elixir {elixir_a:.3f} vs {elixir_b:.3f}: the gap "
        f"should be the {discarded:.3f} the cap discarded")
    assert discarded > 5.0, "fixture: the cap must have discarded a LOT"


def test_neither_scalar_alone_is_enough(samples):
    """The control for the reconstruction: each scalar alone must score no better
    than predicting the mean, so the fit genuinely uses both.
    """
    X, y, tainted = samples
    ok = ~tainted
    yy = y[ok]
    baseline = float(np.abs(yy - yy.mean()).mean())
    assert baseline > 1.0, f"target too flat to conclude anything (MAE {baseline:.3f})"
    for label, col in (("integrated income", C_INCOME),
                       ("opponent elixir spent", C_OPP_SPENT)):
        alone = _mae(X[ok], yy, [col, C_BIAS])
        assert alone > 0.9 * baseline, (
            f"{label} alone reaches MAE {alone:.4f} against a "
            f"predict-the-mean baseline of {baseline:.4f}; the two-scalar "
            "result would then not be evidence of a joint reconstruction")


def test_the_metric_cannot_distinguish_memory_from_arithmetic(samples):
    """The operational warning: a stateless affine map clears the old ~1.3
    "memory" threshold by more than an order of magnitude.
    """
    X, y, tainted = samples
    ok = ~tainted
    memoryless = _mae(X[ok], y[ok], [C_INCOME, C_OPP_SPENT, C_BIAS])
    claimed_memory_threshold = 1.3
    assert memoryless < claimed_memory_threshold / 10, (
        f"a stateless least-squares fit scores {memoryless:.4f} against the "
        f"{claimed_memory_threshold} threshold documented as indicating that "
        "'the recurrent state genuinely counts'. Any reading below that "
        "threshold is therefore consistent with zero memory, and the trained "
        "head's reported 0.77-0.83 is a SHORTFALL against this bound, not a "
        "capability measured against predict-the-mean.")
