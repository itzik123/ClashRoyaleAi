"""The statistics every measurement harness in this project needs, once.

Four near-identical implementations of "paired bootstrap CI plus an exact sign
test" existed here -- `prove_environment.paired`, `prove_placement.paired_report`,
`prove_solvency.boot_ci`/`boot_diff`, `expert_iteration.report_paired` -- with
the same 10,000 resamples and the same `math.comb` sign test written out four
times. For measurement code that is worse than ordinary duplication: this is the
part of the tree whose whole job is to be trustworthy, and four copies means
four chances for one of them to quietly use a different tail, a different
confidence level, or a one-sided test.

WHY BOOTSTRAP AND SIGN TEST TOGETHER, AND WHY BOTH ARE REPORTED. They answer
different questions and this project has had them disagree -- the Fireball
result at ep 78,270 had the bootstrap CI exclude zero while the exact sign test
did not (223 better / 252 worse, p = 0.199), because the advisor won more pairs
while the net won bigger ones. The CI is about the average SIZE of the effect;
the sign test is about how often it points the same way, and it makes no
distributional assumption at all.

Bootstrap rather than a t-test because these scores are heavily zero-inflated --
most cells catch nothing -- so normality is a bad assumption.
"""
from dataclasses import dataclass
from math import comb

import numpy as np

#: Resamples. 10,000 is what every number in CLAUDE.md was measured with;
#: changing it would make new CIs not directly comparable to recorded ones.
N_RESAMPLES = 10000


def _rng(rng):
    # Seeded by default so a harness re-run reproduces its own CI. An unseeded
    # default would make two runs of the same measurement disagree in the third
    # decimal for no reason anyone could trace.
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

    `p` is the two-sided bootstrap p: how often the resampled difference crosses
    zero.

    For arms that are NOT paired. Prefer `paired()` whenever the two arms saw
    the same states -- `env.snapshot()` makes pairing available almost
    everywhere here, and CLAUDE.md records it being worth roughly an order of
    magnitude in the sample size needed to resolve a few win-rate points
    (~1,568 episodes per arm unpaired).
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

    Exact binomial on the discordant pairs only -- ties carry no directional
    information and are excluded from the test while still being reported, so a
    result that is mostly ties cannot look decisive.
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

        When they DISAGREE, believe the sign test: it assumes nothing about the
        distribution, and the disagreement itself is informative -- it means one
        arm wins more pairs while the other wins bigger ones.
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
    """The WIN-RATE report: normal-approximation CI, exact McNemar, and power.

    Deliberately NOT the bootstrap above, and the difference is not an
    oversight. A win-rate delta over paired trials is a mean of values in
    {-1, -0.5, 0, +0.5, +1}, where the normal approximation is well behaved and
    the closed-form SE is what makes the POWER line below computable at all --
    and that line is the one that stopped this project over-reading a marginal
    result. Every delta quoted in CLAUDE.md for expert iteration was produced by
    exactly this arithmetic; changing it would make new numbers incomparable
    with recorded ones.

    The power line exists because of a specific mistake: an exploratory n=200
    arm gave +0.105 at p=0.044 and it was NOISE -- a confirmatory run at 4x the
    power collapsed it to +0.016. Printing the smallest resolvable effect next
    to the observed one makes "this is inside the noise floor" visible at the
    moment of reading rather than three days later.
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
