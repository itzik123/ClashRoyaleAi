"""Capacity and temperature for expert collection under the WIDENED search.

WHY THESE TWO ARE ONE PROBLEM
-----------------------------
Collection pads each row's ranked candidate set to a fixed schema width
`K_MAX` and, on overflow, kept the K_MAX HIGHEST-scoring candidates. Under the
widened search that is not a capacity nuisance, it silently breaks the
temperature calibration the whole distillation depends on.

Measured over 100 decisions with >=2 candidates:

    candidates/decision   median 77   p90 139   max 150
    rows over K_MAX=8     87.0%

    value spread (max-min)   full set  mean 0.5786  median 0.6186
                             top-8     mean 0.2111  median 0.1055
                             overflow rows: 0.6371 -> 0.1036  (84% compressed)

The top 8 are the candidates most similar to each other, so keeping them throws
away 84% of the spread. And the target is `softmax(values / T)`, so:

    T      full set   top-8
    0.02      0.288   0.538
    0.05      0.592   0.858     <- calibrate here, collect truncated, and the
    0.25      0.959   0.994        target is near-uniform without any symptom

Calibrating T=0.05 on the full distribution and then collecting truncated puts
the real target at 0.858 of maximum entropy -- the near-uniform region CLAUDE.md
records as having produced a null "while looking like it was training".
"""
import numpy as np
import pytest


# --- 1. the widened proposal set must be analytically bounded ---------------
def test_the_widened_set_is_capped():
    """A schema width must rest on a proof, not on the largest sample seen.

    Before the cap the worst case was 1 + k_cards * 153 = 460 (the full stride
    grid for every expanded card) while the observed max was 150 -- so sizing
    K_MAX on observation would have been sizing it on luck.
    """
    import torch
    from python_ai.search.config import WIDE_PROPOSAL_MAX_CELLS
    from python_ai.search.search import propose_cells

    n = 612
    flat = torch.log(torch.full((n,), 1.0 / n))
    cells = propose_cells(flat, k_cells=2)
    assert len(cells) <= WIDE_PROPOSAL_MAX_CELLS


def test_the_cap_keeps_the_board_spread():
    """Capping by truncating the grid would collapse it into one corner."""
    import torch
    from python_ai.engine_constants import BOARD_H, BOARD_W
    from python_ai.search.search import propose_cells

    n = 612
    cells = propose_cells(torch.log(torch.full((n,), 1.0 / n)), k_cells=2)
    xs = [c % BOARD_W for c in cells]
    ys = [c // BOARD_W for c in cells]
    assert max(xs) - min(xs) > BOARD_W // 2, "x range collapsed"
    assert max(ys) - min(ys) > BOARD_H // 2, "y range collapsed"


def test_max_candidates_accounts_for_widening():
    """THE GUARD THAT HAD SILENTLY STOPPED GUARDING.

    `max_candidates` returned 1 + k_cards*k_cells = 7 while the real search
    emitted up to 150, so the padding-width test passed for a schema that could
    not hold a row. Same failure mode as the receptive-field test that kept
    measuring 10x10 across the change that invalidated its own docstring.
    """
    from python_ai.search.config import SearchCfg, WIDE_PROPOSAL_MAX_CELLS

    cfg = SearchCfg(k_cards=3, k_cells=2)
    assert cfg.max_candidates >= 1 + 3 * WIDE_PROPOSAL_MAX_CELLS


def test_the_padding_width_holds_the_widened_search():
    from python_ai import shipping
    from python_ai.trainers.expert_collect import K_MAX

    assert shipping.search_cfg().max_candidates <= K_MAX


# --- 2. overflow must preserve the SPREAD, not the top ---------------------
def test_overflow_keeps_the_min_and_the_max():
    """Value-sorted UNIFORM STRIDE, not top-k.

    Top-k keeps the candidates most similar to each other. Striding the
    value-sorted list keeps the endpoints, so the target's dynamic range -- the
    only thing distinguishing "marginally better" from "a blunder" -- survives.
    """
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.linspace(0.0, 1.0, 100)
    keep = select_candidate_indices(scores, k_max=8)

    assert len(keep) == 8
    assert scores[keep].max() == pytest.approx(1.0)
    assert scores[keep].min() == pytest.approx(0.0)


def test_overflow_retains_the_best_candidate_first():
    """The argmax is the expert's actual choice and must never be dropped."""
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.array([0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05, 0.8])
    keep = select_candidate_indices(scores, k_max=4)
    assert 1 in keep, "the highest-scoring candidate must survive"


def test_no_overflow_keeps_everything_ranked():
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.array([0.3, 0.1, 0.2])
    keep = select_candidate_indices(scores, k_max=8)
    assert len(keep) == 3
    assert list(scores[keep]) == sorted(scores, reverse=True)


def test_striding_preserves_far_more_spread_than_top_k():
    """The measured claim, as an executable contrast.

    Real overflow rows compressed 0.6371 -> 0.1036 under top-k. Striding must
    do materially better on the same shape of input.
    """
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.concatenate([np.linspace(0.60, 0.64, 40),   # a tight top cluster
                             np.linspace(0.00, 0.59, 60)])
    topk = np.sort(scores)[::-1][:8]
    strided = scores[select_candidate_indices(scores, k_max=8)]

    assert (strided.max() - strided.min()) > 5 * (topk.max() - topk.min())


# --- 3. temperature must be recalibrated per DAgger round ------------------
def test_temperature_is_solved_from_the_data_not_fixed():
    """T is a property of the CRITIC's value spread, and expert iteration
    changes the critic -- so a T fixed once is correct only for round 0."""
    from python_ai.trainers.expert_distill import calibrate_temperature

    rng = np.random.default_rng(0)
    wide = rng.normal(0.0, 0.30, size=(200, 16))
    narrow = rng.normal(0.0, 0.03, size=(200, 16))
    n = np.full(200, 16, dtype=np.int64)

    t_wide = calibrate_temperature(wide, n, target_frac=0.55)
    t_narrow = calibrate_temperature(narrow, n, target_frac=0.55)

    assert t_narrow < t_wide, "a tighter spread needs a LOWER temperature"


def test_the_calibrated_temperature_hits_its_target():
    from python_ai.trainers.expert_distill import (calibrate_temperature,
                                                   target_entropy_frac)

    rng = np.random.default_rng(1)
    vals = rng.normal(0.0, 0.25, size=(300, 12))
    n = np.full(300, 12, dtype=np.int64)

    T = calibrate_temperature(vals, n, target_frac=0.55)
    assert abs(target_entropy_frac(vals, n, T) - 0.55) < 0.05


def test_calibration_refuses_a_degenerate_dataset():
    """All-equal values carry no preference. Returning some default T would
    hand the trainer a uniform target that looks like a real one."""
    from python_ai.trainers.expert_distill import calibrate_temperature

    vals = np.zeros((50, 8), dtype=np.float32)
    n = np.full(50, 8, dtype=np.int64)
    with pytest.raises(ValueError, match="spread"):
        calibrate_temperature(vals, n, target_frac=0.55)


def test_the_cap_holds_even_when_the_argmax_is_OFF_the_grid():
    """The case that escaped the first cap test and reached the bound check.

    `propose_cells` appends the head's argmax so search can never do worse than
    the head it is helping. Appending it AFTER the cap returns
    WIDE_PROPOSAL_MAX_CELLS + 1 cells, and with k_cards=3 that put the real
    emission at 147 against an analytic bound of 145 -- caught only by an
    end-to-end assert, because the first cap test used a uniform distribution
    whose argmax lands on the grid by luck.
    """
    import torch
    from python_ai.search.config import WIDE_PROPOSAL_MAX_CELLS
    from python_ai.search.search import propose_cells

    n = 612
    v = torch.full((n,), 1.0 / n)
    v[7] = v[7] * 1.5              # argmax at an ODD index, off a (2,2) grid
    cells = propose_cells(torch.log(v), k_cells=2)

    assert len(cells) <= WIDE_PROPOSAL_MAX_CELLS, (
        f"cap exceeded: {len(cells)} > {WIDE_PROPOSAL_MAX_CELLS}")
    assert 7 in cells, "the head's argmax must still survive the cap"


def test_the_analytic_bound_is_not_merely_the_observed_maximum():
    """The bound must cover the worst case propose_cells can actually emit."""
    from python_ai.search.config import SearchCfg, WIDE_PROPOSAL_MAX_CELLS

    cfg = SearchCfg(k_cards=3, k_cells=2)
    worst = 1 + cfg.k_cards * WIDE_PROPOSAL_MAX_CELLS
    assert cfg.max_candidates >= worst
