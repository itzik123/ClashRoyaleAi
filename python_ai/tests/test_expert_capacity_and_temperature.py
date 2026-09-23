"""Capacity and temperature for expert collection under the widened search.

Collection pads each row's candidate set to K_MAX. Under the widened search
most rows overflow, and keeping the top-scoring K_MAX discards most of the
value spread (the top candidates are the most similar), which pushes the
`softmax(values / T)` target toward uniform with no symptom. Overflow keeps a
value-sorted stride instead, and T is recalibrated from the data.
"""
import numpy as np
import pytest


# --- 1. the widened proposal set is analytically bounded ---
def test_the_widened_set_is_capped():
    """A schema width must rest on a proof, not on the largest sample seen."""
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
    """`max_candidates` must include the widened set, or the padding-width test
    passes for a schema that cannot hold a row.
    """
    from python_ai.search.config import SearchCfg, WIDE_PROPOSAL_MAX_CELLS

    cfg = SearchCfg(k_cards=3, k_cells=2)
    assert cfg.max_candidates >= 1 + 3 * WIDE_PROPOSAL_MAX_CELLS


def test_the_padding_width_holds_the_widened_search():
    from python_ai import shipping
    from python_ai.trainers.expert_collect import K_MAX

    assert shipping.search_cfg().max_candidates <= K_MAX


# --- 2. overflow preserves the spread, not the top ---
def test_overflow_keeps_the_min_and_the_max():
    """A value-sorted uniform stride keeps the endpoints, so the target's dynamic
    range, which separates "marginally better" from "a blunder", survives.
    """
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.linspace(0.0, 1.0, 100)
    keep = select_candidate_indices(scores, k_max=8)

    assert len(keep) == 8
    assert scores[keep].max() == pytest.approx(1.0)
    assert scores[keep].min() == pytest.approx(0.0)


def test_overflow_retains_the_best_candidate_first():
    """The argmax is the expert's actual choice and is never dropped."""
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
    """Striding keeps far more spread than top-k on the same input shape."""
    from python_ai.trainers.expert_collect import select_candidate_indices

    scores = np.concatenate([np.linspace(0.60, 0.64, 40),   # a tight top cluster
                             np.linspace(0.00, 0.59, 60)])
    topk = np.sort(scores)[::-1][:8]
    strided = scores[select_candidate_indices(scores, k_max=8)]

    assert (strided.max() - strided.min()) > 5 * (topk.max() - topk.min())


# --- 3. temperature is recalibrated per DAgger round ---
def test_temperature_is_solved_from_the_data_not_fixed():
    """T is a property of the critic's value spread, which expert iteration
    changes.
    """
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
    """All-equal values carry no preference; a default T would hand the trainer a
    uniform target that looks real.
    """
    from python_ai.trainers.expert_distill import calibrate_temperature

    vals = np.zeros((50, 8), dtype=np.float32)
    n = np.full(50, 8, dtype=np.int64)
    with pytest.raises(ValueError, match="spread"):
        calibrate_temperature(vals, n, target_frac=0.55)


def test_the_cap_holds_even_when_the_argmax_is_OFF_the_grid():
    """`propose_cells` appends the head's argmax so search never does worse than
    the head; it must be counted inside the cap. A uniform input puts the
    argmax on the grid by luck, so this uses an off-grid one.
    """
    import torch
    from python_ai.search.config import WIDE_PROPOSAL_MAX_CELLS
    from python_ai.search.search import propose_cells

    n = 612
    v = torch.full((n,), 1.0 / n)
    v[7] = v[7] * 1.5              # argmax at an odd index, off a (2,2) grid
    cells = propose_cells(torch.log(v), k_cells=2)

    assert len(cells) <= WIDE_PROPOSAL_MAX_CELLS, (
        f"cap exceeded: {len(cells)} > {WIDE_PROPOSAL_MAX_CELLS}")
    assert 7 in cells, "the head's argmax must still survive the cap"


def test_the_analytic_bound_is_not_merely_the_observed_maximum():
    """The bound covers the worst case propose_cells can emit."""
    from python_ai.search.config import SearchCfg, WIDE_PROPOSAL_MAX_CELLS

    cfg = SearchCfg(k_cards=3, k_cells=2)
    worst = 1 + cfg.k_cards * WIDE_PROPOSAL_MAX_CELLS
    assert cfg.max_candidates >= worst
