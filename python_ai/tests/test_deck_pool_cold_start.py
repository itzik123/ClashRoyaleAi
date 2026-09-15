"""The deck pool at a FROM-SCRATCH start, with the production defaults.

Audit 04, C2/C3 (2026-09-15). Two facts, both measured:

* **The cold-start branch was unreachable.** `pfsp_weights`' "nothing clears the
  floor -> easiest first" fallback tested `wr >= floor`, and the floor has been
  0.0 since 2026-09-06. Rates are clamped to [0, 1], so `wr >= 0.0` is always
  true and the branch could never fire. The tests that covered it all passed an
  explicit `floor=0.20` -- i.e. they tested a configuration production does not
  run. The fix separates the retired GATE (`POOL_WINRATE_FLOOR`, removed because
  a win rate confounds deck difficulty with teacher competence) from a
  TIE-BREAK that only applies when there is no ranking signal at all.

* **The shipped priors were wrong for this run.** `prior_win_rate` was measured
  against a trained 2.6 Hog Cycle policy. Seeded into a random-init net with a
  different deck, PFSP read them as confident and handed out the four hardest
  decks 59.9% of the time and the mirror 1.0%. Simulated over 8 workers, the
  priors bought no episode-share benefit over uniform and a 3.2x larger estimate
  error at episode 3,000 -- and a wrongly-HIGH prior is self-sealing, because a
  high estimate buys a low weight and so no samples to correct it.
"""
import pytest

from python_ai.opponents import deck_pool


def test_the_cold_start_tie_break_is_reachable_with_production_defaults():
    """No `floor=` argument: this is the call gym_wrapper actually makes."""
    w = deck_pool.pfsp_weights({"awful": 0.00, "bad": 0.02, "least_bad": 0.04})
    # Below the signal floor 0.00 vs 0.04 is one to four wins per hundred:
    # noise. The honest encoding of "no signal" is equal shares -- not the
    # (1-wr)^2 curve, and not the old easiest-first rule either.
    assert max(w.values()) == pytest.approx(min(w.values()))
    assert sum(w.values()) == pytest.approx(1.0)


def test_the_tie_break_never_parks_a_deck_out_of_rotation():
    """Hard decks are the point. The old fallback's bare `wr**2` gave the
    hardest deck 0.1% of episodes -- an exclusion in all but name."""
    rates = {f"d{i}": r for i, r in enumerate([0.0, 0.0, 0.01, 0.02, 0.03,
                                               0.04, 0.04, 0.04])}
    w = deck_pool.pfsp_weights(rates)
    assert min(w.values()) > 0.02, f"a deck fell to {min(w.values()):.3%}"


def test_a_single_deck_with_signal_restores_ordinary_pfsp_exactly():
    """Only the no-signal regime moves. With any deck at or above the signal
    floor the weights are bit-identical to the ordinary (1-wr)^2 curve, so a
    resumed run and every existing measurement are unchanged."""
    rates = {"awful": 0.00, "bad": 0.02, "ok": 0.30}
    w = deck_pool.pfsp_weights(rates)
    raw = {n: max(deck_pool.POOL_MIN_WEIGHT, (1.0 - r) ** 2) for n, r in rates.items()}
    total = sum(raw.values())
    for n in rates:
        assert w[n] == pytest.approx(raw[n] / total, abs=1e-12)


def test_the_shipped_pool_starts_from_neutral_priors():
    """For a new agent deck the 2.6-measured priors describe a different
    matchup table. Every deck starts at NEUTRAL_PRIOR and the count-weighted
    estimator takes it from there within ~10 matches per deck."""
    decks = deck_pool.load_pool()
    priors = {d.name: d.prior_win_rate for d in decks}
    assert set(priors.values()) == {deck_pool.NEUTRAL_PRIOR}, priors


def test_a_neutral_start_is_uniform_not_hardest_first():
    decks = deck_pool.load_pool()
    w = deck_pool.pfsp_weights({d.name: d.prior_win_rate for d in decks})
    assert max(w.values()) == pytest.approx(min(w.values()))
