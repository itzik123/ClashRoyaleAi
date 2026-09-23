"""The deck pool at a from-scratch start, with production defaults.

* With no deck showing a ranking signal, the weights are uniform (a tie-break
  separate from the retired POOL_WINRATE_FLOOR gate).
* The shipped pool starts from neutral priors: priors measured against another
  deck's trained policy would be read as confident, and a wrongly high prior is
  self-sealing (a high estimate buys a low weight and no samples to correct
  it).
"""
import pytest

from python_ai.opponents import deck_pool


def test_the_cold_start_tie_break_is_reachable_with_production_defaults():
    """No `floor=` argument: the call gym_wrapper actually makes."""
    w = deck_pool.pfsp_weights({"awful": 0.00, "bad": 0.02, "least_bad": 0.04})
    # Below the signal floor, 0.00 vs 0.04 is noise; "no signal" means equal
    # shares.
    assert max(w.values()) == pytest.approx(min(w.values()))
    assert sum(w.values()) == pytest.approx(1.0)


def test_the_tie_break_never_parks_a_deck_out_of_rotation():
    """Hard decks are the point: no deck may be parked out of rotation."""
    rates = {f"d{i}": r for i, r in enumerate([0.0, 0.0, 0.01, 0.02, 0.03,
                                               0.04, 0.04, 0.04])}
    w = deck_pool.pfsp_weights(rates)
    assert min(w.values()) > 0.02, f"a deck fell to {min(w.values()):.3%}"


def test_a_single_deck_with_signal_restores_ordinary_pfsp_exactly():
    """Only the no-signal regime moves; with any deck above the signal floor the
    weights are the ordinary (1-wr)^2 curve exactly.
    """
    rates = {"awful": 0.00, "bad": 0.02, "ok": 0.30}
    w = deck_pool.pfsp_weights(rates)
    raw = {n: max(deck_pool.POOL_MIN_WEIGHT, (1.0 - r) ** 2) for n, r in rates.items()}
    total = sum(raw.values())
    for n in rates:
        assert w[n] == pytest.approx(raw[n] / total, abs=1e-12)


def test_the_shipped_pool_starts_from_neutral_priors():
    """Every deck starts at NEUTRAL_PRIOR; the count-weighted estimator takes it
    from there within ~10 matches.
    """
    decks = deck_pool.load_pool()
    priors = {d.name: d.prior_win_rate for d in decks}
    assert set(priors.values()) == {deck_pool.NEUTRAL_PRIOR}, priors


def test_a_neutral_start_is_uniform_not_hardest_first():
    decks = deck_pool.load_pool()
    w = deck_pool.pfsp_weights({d.name: d.prior_win_rate for d in decks})
    assert max(w.values()) == pytest.approx(min(w.values()))
