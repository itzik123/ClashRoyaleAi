"""Paired bootstrap CIs and exact sign tests, shared by every measurement harness.

The two answer different questions and can disagree: the CI is about the
average size of an effect, the sign test about how often it points the same way
(with no distributional assumption). One arm can win more pairs while the other
wins bigger ones.

Bootstrap rather than a t-test because these scores are heavily zero-inflated.
"""
from dataclasses import dataclass
from math import comb

import numpy as np

#: Resamples. Recorded CIs were measured with 10,000; keep it for
#: comparability.
N_RESAMPLES = 10000


def _rng(rng):
    # Seeded by default so a harness re-run reproduces its own CI.
    return np.random.default_rng(0) if rng is None else rng


def bootstrap_ci(values, rng=None, n=N_RESAMPLES, alpha=0.05):
    """(mean, lo, hi) percentile bootstrap CI of the mean."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    r = _rng(rng)
    boot = np.array([x[r.integers(0, x.size, x.size)].mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def unpaired_bootstrap_diff(a, b, rng=None, n=N_RESAMPLES, alpha=0.05):
    """(mean(b) - mean(a), lo, hi, p), resampling the two samples independently.

    `p` is the two-sided bootstrap p. For unpaired arms only: prefer `paired()`
    whenever both arms saw the same states, which `env.snapshot()` makes
    possible almost everywhere.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nan = float("nan")
    if a.size == 0 or b.size == 0:
        return nan, nan, nan, 1.0
    r = _rng(rng)
    d = np.array([b[r.integers(0, b.size, b.size)].mean()
                  - a[r.integers(0, a.size, a.size)].mean() for _ in range(n)])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(b.mean() - a.mean()), float(lo), float(hi), float(min(1.0, p))


def sign_test(diffs):
    """(better, worse, tied, two-sided exact p) over discordant pairs.

    Ties carry no directional information: excluded from the test but still
    reported, so a result that is mostly ties cannot look decisive.
    """
    d = np.asarray(diffs, dtype=np.float64)
    better = int((d > 0).sum())
    worse = int((d < 0).sum())
    tied = int(d.size) - better - worse
    m = better + worse
    if m == 0:
        return better, worse, tied, 1.0
    k = min(better, worse)
    p = 2.0 * sum(comb(m, i) for i in range(k + 1)) / (2.0 ** m)
    return better, worse, tied, min(1.0, p)


@dataclass(frozen=True)
class PairedResult:
    """One paired comparison, with both verdicts attached."""
    n: int
    mean_a: float
    mean_b: float
    delta: float
    lo: float
    hi: float
    better: int
    worse: int
    tied: int
    p: float

    @property
    def ci_excludes_zero(self):
        return (self.lo > 0.0) or (self.hi < 0.0)

    @property
    def agrees(self):
        """True when the CI and the sign test point the same way.

        When they disagree, trust the sign test; the disagreement means one arm
        wins more pairs while the other wins bigger ones.
        """
        return self.ci_excludes_zero == (self.p < 0.05)

    def format(self, name, label_a="a", label_b="b"):
        return (f"  {name}: {label_a} {self.mean_a:.3f}  ->  "
                f"{label_b} {self.mean_b:.3f}\n"
                f"    paired delta {self.delta:+.3f}  "
                f"95% CI [{self.lo:+.3f}, {self.hi:+.3f}]   n={self.n}\n"
                f"    {self.better} better / {self.worse} worse / "
                f"{self.tied} tied   sign test p = {self.p:.3g}")


def paired(a, b, rng=None, n=N_RESAMPLES):
    """Compare two arms measured on the SAME states. Delta is b - a."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arms must be the same length: "
                         f"{a.shape} vs {b.shape}")
    return paired_from_diffs(b - a, rng=rng, n=n,
                             mean_a=float(a.mean()) if a.size else float("nan"),
                             mean_b=float(b.mean()) if b.size else float("nan"))


def paired_from_diffs(diffs, rng=None, n=N_RESAMPLES, mean_a=float("nan"),
                      mean_b=float("nan")):
    """Same, for a harness that already holds per-pair differences."""
    d = np.asarray(diffs, dtype=np.float64)
    mean, lo, hi = bootstrap_ci(d, rng=rng, n=n)
    better, worse, tied, p = sign_test(d)
    return PairedResult(n=int(d.size), mean_a=mean_a, mean_b=mean_b,
                        delta=mean, lo=lo, hi=hi, better=better, worse=worse,
                        tied=tied, p=p)


def report_paired(name, a, b, label_a="a", label_b="b", rng=None):
    """Compare and PRINT. Returns the result so a caller can also assert on it."""
    if len(a) == 0:
        print(f"  {name}: no paired states")
        return None
    result = paired(a, b, rng=rng)
    print(result.format(name, label_a, label_b))
    return result


def report_paired_winrate(diffs, a_scores, b_scores, label_a, label_b):
    """The win-rate report: normal-approximation CI, exact McNemar, and power.

    Not the bootstrap: a paired win-rate delta is a mean of values in {-1,
    -0.5, 0, +0.5, +1}, where the normal approximation behaves and its
    closed-form SE makes the power line computable. That line prints the
    smallest resolvable effect next to the observed one, so a result inside the
    noise floor is visible as such.
    """
    import math

    diffs = np.asarray(diffs, dtype=np.float64)
    a_scores = np.asarray(a_scores, dtype=np.float64)
    b_scores = np.asarray(b_scores, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        print("no trials completed")
        return None

    mean_d = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean_d - 1.96 * se, mean_d + 1.96 * se
    print("\n" + "=" * 68)
    print(f"trials (paired)        : {n}")
    print(f"{label_a:22s} : {a_scores.mean():.4f}")
    print(f"{label_b:22s} : {b_scores.mean():.4f}")
    print(f"paired delta           : {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  significant?         : "
          f"{'YES' if (lo > 0 or hi < 0) else 'no -- CI includes 0'}")
    better, worse, tied, p = sign_test(diffs)
    print(f"  {label_b} better/worse/same : {better} / {worse} / {tied}")
    if better + worse:
        print(f"  exact McNemar p        : {p:.3e} "
              f"({better + worse} discordant pairs)")
    if n > 1 and se > 0:
        detectable = 1.96 * se
        print(f"\nsmallest effect this n could resolve: +/-{detectable:.4f} "
              f"({detectable * 100:.1f} win-rate points)")
        if abs(mean_d) < detectable:
            needed = int(math.ceil(
                (1.96 * diffs.std(ddof=1) / max(1e-6, abs(mean_d))) ** 2))
            print("observed effect is INSIDE the noise floor; resolving it "
                  f"would need ~{needed} paired trials.")
    return PairedResult(n=n, mean_a=float(a_scores.mean()),
                        mean_b=float(b_scores.mean()), delta=mean_d, lo=lo,
                        hi=hi, better=better, worse=worse, tied=tied, p=p)
