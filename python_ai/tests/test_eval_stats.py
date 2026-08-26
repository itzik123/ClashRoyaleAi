"""eval/stats.py -- the measurement code, which has to be the most trustworthy.

Four harnesses each carried their own copy of paired-bootstrap CI plus an exact
sign test. These tests exist because a statistics helper is exactly the kind of
code that "looks right" and is not: a one-sided test, a 90% interval, or a sign
test that counts ties as agreement would all read plausibly and would all
silently inflate every result in CLAUDE.md.
"""
import numpy as np
import pytest

from python_ai.eval import stats


# ------------------------------------------------------------- sign test --
def test_the_sign_test_is_TWO_sided():
    """One-sided would halve every p in the project's history."""
    better, worse, tied, p = stats.sign_test([1, 1, 1, 1, 1, -1])
    assert (better, worse, tied) == (5, 1, 0)
    # Two-sided exact binomial on 6 trials with k<=1: 2 * (1 + 6) / 64.
    assert p == pytest.approx(2 * 7 / 64)


def test_ties_are_reported_but_excluded_from_the_test():
    """Ties carry no directional information. Counting them as agreement would
    make a result that is mostly ties look decisive."""
    better, worse, tied, p = stats.sign_test([1, -1, 0, 0, 0, 0])
    assert (better, worse, tied) == (1, 1, 4)
    assert p == 1.0


def test_an_all_tied_comparison_is_p_equals_one_not_a_division_by_zero():
    assert stats.sign_test([0, 0, 0]) == (0, 0, 3, 1.0)


def test_a_unanimous_result_is_the_smallest_p_the_sample_allows():
    _b, _w, _t, p = stats.sign_test([1] * 10)
    assert p == pytest.approx(2 / 2 ** 10)


# ------------------------------------------------------------- bootstrap --
def test_the_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 400)
    mean, lo, hi = stats.bootstrap_ci(x, rng=np.random.default_rng(1))
    assert lo < mean < hi
    assert mean == pytest.approx(float(x.mean()))


def test_the_ci_is_95_percent_by_default_and_narrows_with_n():
    rng = np.random.default_rng(2)
    _m, lo_small, hi_small = stats.bootstrap_ci(
        rng.normal(0, 1, 30), rng=np.random.default_rng(3))
    _m, lo_big, hi_big = stats.bootstrap_ci(
        rng.normal(0, 1, 3000), rng=np.random.default_rng(3))
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_the_default_rng_is_SEEDED_so_a_rerun_reproduces_its_own_ci():
    """An unseeded default would make two runs of the same measurement disagree
    in the third decimal for no reason anyone could trace."""
    x = np.arange(50, dtype=float)
    assert stats.bootstrap_ci(x) == stats.bootstrap_ci(x)


def test_an_empty_sample_yields_nan_not_a_crash():
    mean, lo, hi = stats.bootstrap_ci([])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------- paired --
def test_delta_is_b_minus_a():
    r = stats.paired([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
    assert r.delta == pytest.approx(2.0)
    assert r.mean_a == pytest.approx(1.0) and r.mean_b == pytest.approx(3.0)


def test_mismatched_arms_are_refused():
    """Two arms of different lengths are not paired, and silently truncating
    them would compare state i of one run against state i of another."""
    with pytest.raises(ValueError, match="same length"):
        stats.paired([1.0, 2.0], [1.0])


def test_a_clear_effect_excludes_zero_and_both_verdicts_agree():
    a = np.zeros(60)
    b = np.ones(60)
    r = stats.paired(a, b)
    assert r.ci_excludes_zero and r.p < 0.05 and r.agrees


def test_no_effect_includes_zero():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 200)
    r = stats.paired(x, x + rng.normal(0, 1, 200) * 0.001)
    assert not r.ci_excludes_zero


def test_the_two_verdicts_can_disagree_and_that_is_reported_not_hidden():
    """The Fireball result at ep 78,270: the bootstrap CI excluded zero while
    the exact sign test did not (223 better / 252 worse, p = 0.199), because
    the advisor won more PAIRS while the net won bigger ones. `agrees` exists so
    that situation is legible rather than resolved by whichever test was
    reported.
    """
    # The exact shape of that result: one arm wins slightly MORE PAIRS, the
    # other wins by far more each time. The sign test sees a coin flip; the
    # bootstrap sees a large positive mean.
    diffs = np.array([-1.0] * 26 + [50.0] * 24)
    r = stats.paired_from_diffs(diffs)
    assert r.better < r.worse, "the losing side must win more pairs"
    assert r.p > 0.05, "the sign test must NOT resolve it"
    assert r.ci_excludes_zero, "the bootstrap must resolve it"
    assert not r.agrees


def test_the_formatted_report_names_both_arms_and_both_verdicts():
    r = stats.paired([0.0, 0.0], [1.0, 1.0])
    text = r.format("cannon", "control", "treatment")
    assert "control" in text and "treatment" in text
    assert "95% CI" in text and "sign test p" in text


def test_report_paired_handles_an_empty_comparison(capsys):
    assert stats.report_paired("nothing", [], []) is None
    assert "no paired states" in capsys.readouterr().out


# -------------------------------------------------------------- unpaired --
def test_the_unpaired_diff_returns_a_bootstrap_p_as_well():
    rng = np.random.default_rng(9)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(3.0, 1.0, 200)
    delta, lo, hi, p = stats.unpaired_bootstrap_diff(a, b,
                                                     rng=np.random.default_rng(1))
    assert delta == pytest.approx(float(b.mean() - a.mean()))
    assert lo > 0 and hi > lo and p < 0.05


def test_pairing_is_worth_far_more_power_than_the_same_n_unpaired():
    """The reason `env.snapshot()` exists and the reason CLAUDE.md records
    ~1,568 episodes per arm being needed WITHOUT it."""
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 10.0, 120)     # the shared per-state variation
    a = common + rng.normal(0.0, 0.1, 120)
    b = common + rng.normal(0.5, 0.1, 120)  # a small, real, consistent effect
    paired = stats.paired(a, b, rng=np.random.default_rng(1))
    _d, lo, hi, _p = stats.unpaired_bootstrap_diff(
        a, b, rng=np.random.default_rng(1))
    assert paired.ci_excludes_zero, "pairing should resolve this effect"
    assert (hi - lo) > (paired.hi - paired.lo) * 5


# --------------------------------------------------------- win-rate report --
def test_the_win_rate_report_prints_the_power_line(capsys):
    """The line that stopped an exploratory +0.105 at p=0.044 being believed --
    a confirmatory run at 4x the power collapsed it to +0.016."""
    rng = np.random.default_rng(6)
    a = rng.integers(0, 2, 200).astype(float)
    b = a.copy()
    b[:3] = 1.0
    r = stats.report_paired_winrate(b - a, a, b, "original", "distilled")
    out = capsys.readouterr().out
    assert "smallest effect this n could resolve" in out
    assert "exact McNemar p" in out
    assert r.n == 200


def test_the_win_rate_report_survives_a_zero_trial_call(capsys):
    assert stats.report_paired_winrate(np.array([]), np.array([]),
                                       np.array([]), "a", "b") is None
    assert "no trials completed" in capsys.readouterr().out


def test_the_resample_count_is_the_one_every_recorded_ci_used():
    """Changing it would make new CIs not directly comparable with the numbers
    in CLAUDE.md."""
    assert stats.N_RESAMPLES == 10000


# --- the fifth copy -------------------------------------------------------
#
# `eval/stats.py` exists because "four harnesses each had their own copy" of
# the paired bootstrap. `prove_hog.py` was a fifth that the consolidation
# missed, and it was the one that mattered most: it hand-rolled
#
#     boot = np.array([np.mean(np.random.choice(d, len(d))) for _ in range(5000)])
#
# on the UNSEEDED global RNG, so its 95% CI -- the number the script's own
# verdict branches on ("BETTER than chance" / "WORSE than chance") -- moved
# between two runs of the identical measurement. `stats._rng` is seeded by
# default for exactly this reason: "An unseeded default would make two runs of
# the same measurement disagree in the third decimal for no reason anyone could
# trace." Near a boundary the disagreement is not in the third decimal, it is
# in the conclusion.

def test_no_eval_harness_hand_rolls_its_own_bootstrap():
    """The drift guard. A harness resampling with the global RNG is both a
    second copy of a consolidated function and an irreproducible statistic."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "eval").glob("*.py")):
        if path.name == "stats.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"np\.random\.(choice|randint|integers)\(", code):
                offenders.append(f"eval/{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "hand-rolled resampling on the global RNG:\n" + "\n".join(offenders))


def test_the_shared_bootstrap_is_reproducible_by_default():
    """The property the harnesses are being routed to. Two calls with no
    explicit rng must agree exactly, or consolidating them buys nothing."""
    import numpy as np
    from python_ai.eval.stats import bootstrap_ci
    d = np.linspace(-3.0, 5.0, 64)
    assert bootstrap_ci(d, n=200) == bootstrap_ci(d, n=200)


def test_an_explicit_rng_still_overrides_the_default_seed():
    """A caller that WANTS independent resamples must still be able to say so."""
    import numpy as np
    from python_ai.eval.stats import bootstrap_ci
    d = np.linspace(-3.0, 5.0, 64)
    a = bootstrap_ci(d, rng=np.random.default_rng(1), n=200)
    b = bootstrap_ci(d, rng=np.random.default_rng(2), n=200)
    assert a != b
