"""`Aux/OppElixir_MAE` does not measure what CLAUDE.md says it measures.

THE CLAIM UNDER TEST. CLAUDE.md lists three metrics to watch during a run, and
the first is:

    "Aux/OppElixir_MAE (below ~1.3 means the recurrent state genuinely counts)"

and the baselines section defends the trained head's 0.77-0.83 as "a real
capability, not a trivial correlation", on the strength of having ruled out one
confound: reading your OWN elixir scores no better than predicting the mean
(ratio 0.96-1.05, corr 0.27).

THE CONFOUND THAT WAS NOT RULED OUT. The opponent's elixir is an affine
function of two scalars the observation ALREADY CARRIES, in the appended
`NUM_EXTRA_SCALARS` tail:

    elixir(t) = start + regen_rate * t - spent(t)

`t` is scalar 0 (`currentTick / maxTicks`) and `spent(t)` is scalar 2
(`elixirSpent(1 - team) / MAX_MATCH_ELIXIR`). Both were added deliberately --
`ClashEnv.h` explains that spend is included precisely BECAUSE it is the
observable half of the hidden quantity. So the target is reconstructible from
the present observation by a linear layer, with no memory of any kind.

MEASURED 2026-08-27, 2,606 samples over 8 episodes of random play:

    predictor                                MAE
    predict-the-mean                       1.4271   (CLAUDE.md quotes ~1.35)
    time only                              1.4223
    opponent elixir spent only             1.4268
    time + opp_spent, ORDINARY LEAST SQUARES  0.0000

Four parameters and no recurrence solve it exactly. So a low MAE is not
evidence that "the recurrent state genuinely counts" -- a feedforward net with
no LSTM at all would score the same, and the head reaching 0.77 is not a
capability but a shortfall of about 0.77 against an achievable zero.

WHY THIS MATTERS BEYOND THE METRIC. The auxiliary loss exists to shape the
representation -- to make the trunk carry opponent-modelling information it
would not otherwise need. An auxiliary task solvable by an affine map on two
inputs exerts almost none of that pressure; it teaches arithmetic on two
scalars. The genuinely useful version of this task -- predicting which CARDS
the opponent holds or when their win condition cycles back -- does require
memory and does require opponent modelling, and is not currently asked for
anywhere. See `perception/UPSTREAM_REQUESTS.md` for the observation-side
proposal.

WHAT THIS FILE DOES NOT CLAIM. It does not claim the aux head is useless or
should be removed; a head that is failing an easy task may still be
regularizing the trunk usefully, and that is a training-run question. It
claims only that its MAE cannot be read as a memory diagnostic, which is how
CLAUDE.md currently instructs a reader to use it.

The exact-zero residual has one honest caveat, asserted below rather than
hidden: the relation is exact only while neither side's elixir CLAMPS at the
10 cap, since a clamp destroys information that neither scalar records. Under
the random play sampled here the opponent never capped. A policy that baits
the opponent into overflowing would make the task genuinely non-trivial -- and
that, not memory, is what a rising MAE would actually be detecting.

WHAT THE ELIXIR PHASES CHANGED, 2026-09-02
-------------------------------------------
The caveat above stopped being hypothetical, and it did so without any policy
baiting anyone. `UPSTREAM_REQUESTS.md` item 26 gave the match a real 1x/2x/3x
schedule, and two things followed:

  1. `regen_rate * t` is no longer a single slope. Cumulative income is
     piecewise-linear with breakpoints at 1200 and 1800 ticks, so the ORIGINAL
     basis `[t, spent, 1]` cannot fit it at all -- measured 1.4265, which is
     no better than predict-the-mean. `_income()` below integrates the
     schedule, and with that term the reconstruction is exact again. The
     finding is unchanged; only the basis moved.

  2. Tripling the income made the opponent overflow on its own, and THAT is
     genuinely unrecoverable -- the cap discards elixir no scalar records.
     Measured per episode against the exact analytic model: the three that
     never capped residual at -0.035 (one tick of regen, i.e. noise); the
     three that did carry +0.76, +1.63 and +5.38.

So the conclusion this file was written to establish still holds in the
overflow-free regime, and a rising MAE under phases is an OVERFLOW detector.
It is still not a memory diagnostic. Note that the head this was written about
was deleted on 2026-08-28 (replaced by `Aux/NextCard_CE`); the file is kept for
the reasoning, which is what generalises.
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai.envs.gym_wrapper import MicroRoyaleEnv


_CE = clash_royale_env.ClashRoyaleEnv
N_EXTRA = _CE.NUM_EXTRA_SCALARS

#: FORWARD offsets, measured from the start of the observation.
#:
#: These read from the END until 2026-08-27, as `-N_EXTRA + k`, and that was a
#: latent bug the very next observation change detonated: item 24 appended two
#: NUM_CARD_IDS-wide card-cycle blocks AFTER the extra scalars, at which point
#: every negative offset here silently re-pointed into the cycle data. The
#: tests would have kept passing while measuring entirely different columns --
#: the exact failure `UPSTREAM_REQUESTS.md` item 25 describes.
#:
#: Measured forward, an append cannot move them. Every term is derived from the
#: bindings, so a change to the board, the channel count or the card roster
#: carries these with it.
_SCALAR_START = _CE.BOARD_WIDTH * _CE.BOARD_HEIGHT * _CE.NUM_CHANNELS
_EXTRA_START = (_SCALAR_START + 1 + _CE.HAND_SIZE
                + _CE.HAND_SIZE * _CE.NUM_CARD_IDS)
#: Ordered as ClashEnv.h appends them: time, own elixir spent, opponent elixir
#: spent, then 6 tower HP readings.
T_IDX, SPENT_SELF_IDX, SPENT_OPP_IDX = (
    _EXTRA_START + 0, _EXTRA_START + 1, _EXTRA_START + 2)


def test_the_extra_scalar_offsets_still_point_at_the_extra_scalars():
    """Guards the offsets themselves, because everything below is meaningless
    if they drift. The time scalar is `currentTick / maxTicks`, so on a fresh
    board it must be exactly 0.0 and it must RISE as the match runs -- a
    property no other column in the observation has, which is what makes it a
    usable fingerprint.
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


#: `ELIXIR_REGEN_RATE` is not bound (it is a private member of GameManager), so
#: it is restated here with its source named -- the pattern perception/
#: geometry.py uses for genuinely underivable values. Guarded by
#: `test_the_regen_rate_this_file_assumes_is_still_the_engines` below, which
#: MEASURES it rather than trusting this line.
_REGEN_PER_TICK = 0.035     # GameManager.h: `const float ELIXIR_REGEN_RATE`

#: The bar's ceiling (PlayerState clamps to it) and the env's default
#: `max_ticks`, which the time scalar is normalised by. Neither is bound.
_CAP = 10.0
_MAX_TICKS = 3600.0

#: Column indices into the design matrix built by `_collect`.
C_T, C_OPP_SPENT, C_OWN_SPENT, C_BIAS, C_INCOME = 0, 1, 2, 3, 4


def _income(t):
    """Exact cumulative regen from tick 0 to tick `t`, integrating the phases.

    THIS FUNCTION IS THE 2026-09-02 CHANGE. Before the elixir phases landed,
    cumulative income was `rate * t` -- one term, one slope -- which is why the
    original reconstruction needed only `[t, spent, 1]`. It is now a
    piecewise-linear function of t with breakpoints at the two phase
    boundaries, and no single coefficient on t can represent it.

    Both boundaries are read from the engine, so a schedule change carries this
    with it instead of silently invalidating every fit below.
    """
    d, tr = _CE.DOUBLE_ELIXIR_TICK, _CE.TRIPLE_ELIXIR_TICK
    r = _REGEN_PER_TICK
    return (r * min(t, d)
            + 2.0 * r * max(0.0, min(t, tr) - d)
            + 3.0 * r * max(0.0, t - tr))


def _collect(episodes=6, seed=0):
    """Design matrix and target for reconstructing the opponent's hidden elixir.

    Columns: `[t_ticks, opp_spent, own_spent, 1, income(t)]`. The last is the
    phase-integrated income above; `t_ticks` is kept alongside it so the tests
    can contrast the OLD single-slope basis against the correct one.

    Also returns, per sample, whether the opponent's bar has hit the 10.0 cap
    at any point in that episode up to and including this sample. That flag is
    load-bearing, and it is NOT the same as "this sample reads 10.0": once the
    cap has discarded income, every LATER sample in the episode carries the
    loss as a permanent offset. Counting only samples currently at the cap
    undercounts the contamination by more than an order of magnitude --
    measured 2026-09-02, 0.3% of samples sit at the cap while 3 of 6 EPISODES
    are affected by it.
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
    """`_REGEN_PER_TICK` is a restated private constant, so measure it.

    Read off the BAR over a window entirely inside single elixir, with nobody
    spending. If `GameManager::ELIXIR_REGEN_RATE` moves, `_income` goes wrong
    and every reconstruction below degrades for a reason that has nothing to do
    with what those tests are about.
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
    """THE ORIGINAL FINDING, still true -- but only once income is integrated.

    No recurrence required: the 'hidden' target is reconstructed from scalars
    the observation already carries, by ordinary least squares. What changed on
    2026-09-02 is the BASIS, not the conclusion -- `income(t)` replaces a bare
    `t` because the elixir phases made cumulative regen piecewise-linear.

    Restricted to overflow-free samples, which is the regime the original
    measurement was taken in and exactly the caveat the module docstring
    already stated. See `test_overflow_is_what_makes_this_task_non_trivial`
    for the other regime, which the phases turned from hypothetical to common.
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
    """The 2026-09-02 regression guard, and it asserts a FAILURE.

    Before the elixir phases, `[t, spent, 1]` reconstructed the target exactly
    (MAE 0.0000, measured 2026-08-27 over 2,606 samples). It cannot any more,
    because cumulative income has two breakpoints and one coefficient on t
    cannot bend. Measured 2026-09-02: 1.4265 on overflow-free samples, against
    a predict-the-mean baseline of the same order -- i.e. the naive basis is
    now worth nothing at all.

    Pinned deliberately rather than deleted. If someone reverts the phases or
    makes them continuous, this fails and says so; the naive basis silently
    starting to work again is exactly the kind of change that should not pass
    unnoticed.
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


def test_overflow_is_what_makes_this_task_non_trivial(samples):
    """What the phases actually changed, and it is not the arithmetic.

    The module docstring's closing caveat -- written 2026-08-27, before the
    phases existed -- called this exactly: "a policy that baits the opponent
    into overflowing would make the task genuinely non-trivial, and that, not
    memory, is what a rising MAE would actually be detecting."

    Nothing baits anyone. Tripling the income was enough: the opponent now
    overflows unaided, and the cap DISCARDS elixir that no scalar records, so
    the discarded amount is unrecoverable by any function of the present
    observation. Measured 2026-09-02 per episode against the exact analytic
    model: the three that never capped reconstruct to a residual of -0.035
    (one tick of regen, i.e. noise), while the three that did carry +0.76,
    +1.63 and +5.38.

    So a rising `Aux/OppElixir_MAE` under phases is an overflow detector, and
    still not a memory diagnostic.
    """
    X, y, tainted = samples
    if not tainted.any():
        # NOT a silent pass, and not a failure either: the regime this test is
        # defined on stopped occurring on 2026-09-06, for a reason that is
        # correct.
        #
        # Overflow needs a long match. The match-end rules added that day end a
        # match at 3:00 whenever the crowns differ, and TRIPLE elixir starts at
        # exactly 3:00 -- so the opponent now reaches triple elixir only in
        # OVERTIME, i.e. only when the score is level at regulation. That is the
        # real game's behaviour and it makes overflow rare rather than routine.
        # The 2026-09-02 calibration behind "roughly half of episodes overflow"
        # was measured against the old rules and does not survive it.
        #
        # The FINDING is unaffected -- a cap still destroys information no
        # scalar carries, and `Aux/OppElixir_MAE` would still be detecting that
        # rather than memory. What is gone is this sampler's ability to reach
        # the regime by playing ordinary matches. See TODO 0c: the fix is to
        # CONSTRUCT the overflow state with the state setters instead of hoping
        # the episode distribution supplies it, which would also make the test
        # deterministic.
        pytest.skip(
            "no sampled episode overflowed: since the 2026-09-06 match-end "
            "rules a match ends at 3:00 on a crown lead, so triple elixir is "
            "reached only in overtime. Regime absent, not refuted -- TODO 0c.")
    clean = _mae(X[~tainted], y[~tainted], [C_INCOME, C_OPP_SPENT, C_BIAS])
    dirty = _mae(X[tainted], y[tainted], [C_INCOME, C_OPP_SPENT, C_BIAS])
    assert dirty > 5 * max(clean, 1e-3), (
        f"overflow-free samples reconstruct at MAE {clean:.4f} and "
        f"overflow-contaminated ones at {dirty:.4f}. The gap IS the finding: "
        "the cap destroys information the observation does not carry. If "
        "these have converged, either the cap stopped binding or the "
        "contamination flag is no longer tracking it.")


def test_neither_scalar_alone_is_enough(samples):
    """The control that makes the reconstruction test mean something.

    Without it, "a linear model fits" could be an artifact of the target
    barely varying. Each scalar ALONE must score no better than
    predict-the-mean, so the reconstruction is genuinely using both -- the
    same shape of control CLAUDE.md applied when it ruled out 'read your own
    elixir', carried through to the confound that was missed.
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
    """Stated as the operational warning, so a reader who only runs the suite
    still gets the point: the threshold CLAUDE.md gave (~1.3) is cleared by an
    affine map with no state at all, by more than an order of magnitude."""
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
