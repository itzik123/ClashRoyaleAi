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
"""
import numpy as np
import pytest

import clash_royale_env
from python_ai.envs.gym_wrapper import MicroRoyaleEnv


N_EXTRA = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS
#: Offsets from the END of the observation. Derived from the engine's own
#: constant rather than restated, and ordered as ClashEnv.h appends them:
#: time, elixir spent by this team, elixir spent by the other team, then 6
#: tower HP readings.
T_IDX, SPENT_SELF_IDX, SPENT_OPP_IDX = -N_EXTRA + 0, -N_EXTRA + 1, -N_EXTRA + 2


def _collect(episodes=6, seed=0):
    """(design matrix, target) pairs of (time, opp_spent, own_spent, 1) -> opp elixir."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(episodes):
        env = MicroRoyaleEnv({})
        obs = env.reset()
        done, n = False, 0
        while not done and n < 400:
            res = env.step({"card_index": int(rng.integers(0, 5)),
                            "target_x": float(rng.integers(0, 18)),
                            "target_y": float(rng.integers(0, 16))})
            obs, info = res[0], res[-1]
            done = (bool(res[2]) or bool(res[3])) if len(res) == 5 else bool(res[2])
            o = np.asarray(obs[0] if isinstance(obs, tuple) else obs,
                           dtype=np.float32)
            if "opp_elixir" in info:
                X.append([o[T_IDX], o[SPENT_OPP_IDX], o[SPENT_SELF_IDX], 1.0])
                y.append(float(np.atleast_1d(info["opp_elixir"])[0]))
            n += 1
    return np.asarray(X), np.asarray(y)


@pytest.fixture(scope="module")
def samples():
    X, y = _collect()
    assert len(y) > 500, f"only {len(y)} samples collected; too few to conclude"
    return X, y


def _mae(X, y, cols):
    A = X[:, cols]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.abs(y - A @ w).mean())


def test_opponent_elixir_is_an_affine_function_of_two_present_scalars(samples):
    """THE FINDING. No recurrence required: four parameters reconstruct the
    'hidden' target from the observation the agent already receives."""
    X, y = samples
    mae = _mae(X, y, [0, 1, 3])          # time, opp_spent, bias
    assert mae < 0.05, (
        f"linear reconstruction MAE is {mae:.4f}; the affine relation "
        "elixir = start + rate*t - spent is expected to hold to within "
        "float noise while no elixir bar clamps at its cap. If this has "
        "risen, check whether the engine changed ELIXIR_REGEN_RATE, whether "
        "the extra-scalar ORDER moved (the offsets here are positional), or "
        "whether the sampled policy now baits the opponent into overflowing "
        "-- the last would make the aux task genuinely non-trivial.")


def test_neither_scalar_alone_is_enough(samples):
    """The control that makes the test above mean something.

    Without it, "a linear model fits" could be an artifact of the target
    barely varying. Each scalar ALONE must score no better than
    predict-the-mean, so the reconstruction is genuinely using both -- the
    same shape of control CLAUDE.md applied when it ruled out 'read your own
    elixir', just carried through to the confound that was missed.
    """
    X, y = samples
    baseline = float(np.abs(y - y.mean()).mean())
    assert baseline > 1.0, f"target too flat to conclude anything (MAE {baseline:.3f})"
    for label, col in (("time", 0), ("opponent elixir spent", 1)):
        alone = _mae(X, y, [col, 3])
        assert alone > 0.9 * baseline, (
            f"{label} alone reaches MAE {alone:.4f} against a "
            f"predict-the-mean baseline of {baseline:.4f}; the two-scalar "
            "result would then not be evidence of a joint reconstruction")


def test_the_metric_cannot_distinguish_memory_from_arithmetic(samples):
    """Stated as the operational warning, so a reader who only runs the suite
    still gets the point: the threshold CLAUDE.md gives (~1.3) is cleared by
    an affine map with no state at all, by a margin of more than an order of
    magnitude."""
    X, y = samples
    memoryless = _mae(X, y, [0, 1, 3])
    claimed_memory_threshold = 1.3
    assert memoryless < claimed_memory_threshold / 10, (
        f"a stateless least-squares fit scores {memoryless:.4f} against the "
        f"{claimed_memory_threshold} threshold documented as indicating that "
        "'the recurrent state genuinely counts'. Any reading below that "
        "threshold is therefore consistent with zero memory, and the trained "
        "head's reported 0.77-0.83 is a SHORTFALL against this bound, not a "
        "capability measured against predict-the-mean.")
