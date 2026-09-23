"""The entropy controller. Pins three properties:

  * the coefficient cannot walk out of its bounds;
  * a per-update step cap binds on a large excursion;
  * the placement target anneals, on a clock the caller supplies.
"""
import math

import pytest

from python_ai.rl.config import (
    PHASE1_ENTROPY, PHASE2_ENTROPY, EntropyConfig, log_reachable,
)
from python_ai.rl.entropy import EntropyController


def test_the_target_anneals_from_start_to_final_and_then_stays():
    c = EntropyController(PHASE2_ENTROPY)
    cfg = PHASE2_ENTROPY
    assert c.placement_target(0) == pytest.approx(cfg.target_placement_start)
    mid = c.placement_target(cfg.anneal_episodes // 2)
    assert cfg.target_placement_final < mid < cfg.target_placement_start
    assert c.placement_target(cfg.anneal_episodes) == pytest.approx(
        cfg.target_placement_final)
    # Past the horizon the target holds rather than overshooting.
    assert c.placement_target(10 * cfg.anneal_episodes) == pytest.approx(
        cfg.target_placement_final)


def test_the_card_target_is_deliberately_NOT_annealed():
    """For the card head the danger runs the other way (narrowing card choice
    collapses the deck), so only placement anneals.
    """
    c = EntropyController(PHASE1_ENTROPY)
    assert c.card_target == PHASE1_ENTROPY.target_card
    assert not hasattr(EntropyConfig(), "target_card_final")


def test_measuring_below_target_raises_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_card
    c.update(card_frac=0.0, placement_frac=0.5, anneal_episodes_done=0)
    assert c.coef_card > before


def test_measuring_above_target_lowers_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_card
    c.update(card_frac=1.0, placement_frac=0.5, anneal_episodes_done=0)
    assert c.coef_card < before


def test_a_fresh_policy_is_pushed_DOWN_not_up():
    """A fresh net is near-uniform over its reachable arms, far above the target,
    so the controller must push down.
    """
    c = EntropyController(PHASE1_ENTROPY)
    start = c.coef_card
    c.update(card_frac=0.958, placement_frac=0.996, anneal_episodes_done=0)
    assert c.coef_card < start


def test_the_coefficient_cannot_leave_its_bounds():
    """The floor must stay a floor; too low a value becomes an off switch for the
    entropy term.
    """
    cfg = PHASE1_ENTROPY
    c = EntropyController(cfg)
    for _ in range(200):
        c.update(card_frac=1.0, placement_frac=1.0, anneal_episodes_done=0)
    assert c.coef_card == pytest.approx(cfg.coef_floor)
    assert c.coef_placement == pytest.approx(cfg.coef_floor)

    c = EntropyController(cfg)
    for _ in range(200):
        c.update(card_frac=0.0, placement_frac=0.0, anneal_episodes_done=0)
    assert c.coef_card <= cfg.coef_ceil_card
    assert c.coef_placement <= cfg.coef_ceil_placement


def test_the_step_cap_binds_on_a_large_excursion_where_it_is_set():
    """The cap bounds the multiplicative walk at (1 +/- STEP_MAX)^n however wrong
    the error.
    """
    cfg = PHASE2_ENTROPY
    assert cfg.coef_step_max is not None
    c = EntropyController(cfg)
    before = c.coef_placement
    # A target/measured gap far larger than anything real.
    c.update(card_frac=0.35, placement_frac=-50.0, anneal_episodes_done=0)
    assert c.coef_placement <= before * (1.0 + cfg.coef_step_max) + 1e-12


def test_phase1_has_no_step_cap_so_its_behaviour_is_unchanged():
    """Pipeline 1's controller has no step cap: the general form reduces exactly
    to `clip(coef * exp(rate * error), floor, ceil)`.
    """
    cfg = PHASE1_ENTROPY
    assert cfg.coef_step_max is None
    c = EntropyController(cfg)
    start = c.coef_card
    c.update(card_frac=0.10, placement_frac=0.65, anneal_episodes_done=0)
    expected = min(cfg.coef_ceil_card,
                   start * math.exp(cfg.adapt_rate_card * (cfg.target_card - 0.10)))
    assert c.coef_card == pytest.approx(expected)


def test_the_anneal_clock_is_the_callers_not_the_raw_episode_count():
    """The anneal clock is supplied by the caller, so pipeline 2's stall re-boost
    can hand back a smaller number and widen the target.
    """
    c = EntropyController(PHASE2_ENTROPY)
    late = PHASE2_ENTROPY.anneal_episodes
    assert c.placement_target(late) < c.placement_target(0)
    # A re-boost hands back a small number, and the target widens again.
    assert c.placement_target(0) > c.placement_target(late // 4)


def test_state_survives_a_checkpoint_roundtrip():
    """A resume must not silently reset the controller."""
    c = EntropyController(PHASE2_ENTROPY)
    c.update(card_frac=0.1, placement_frac=0.1, anneal_episodes_done=0)
    saved = c.state_dict()
    restored = EntropyController(PHASE2_ENTROPY)
    restored.load_state_dict(saved)
    assert restored.coef_card == c.coef_card
    assert restored.coef_placement == c.coef_placement


def test_a_checkpoint_without_the_keys_keeps_the_seeded_defaults():
    """An older checkpoint resumes rather than raising KeyError."""
    c = EntropyController(PHASE1_ENTROPY)
    c.load_state_dict({"episodes_completed": 5})
    assert c.coef_card == PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == PHASE1_ENTROPY.initial_coef_placement


def test_the_two_pipelines_keep_their_measured_differences():
    """Not unified on purpose: changing either is gameplay-affecting in that
    pipeline.
    """
    assert PHASE1_ENTROPY.target_placement_start == 0.65
    assert PHASE2_ENTROPY.target_placement_start == 0.50
    assert PHASE1_ENTROPY.adapt_rate_placement == 0.5
    assert PHASE2_ENTROPY.adapt_rate_placement == 0.10
    assert PHASE1_ENTROPY.coef_ceil_placement == 0.5
    assert PHASE2_ENTROPY.coef_ceil_placement == 0.20
    # ...and everything shared really is shared.
    assert PHASE1_ENTROPY.target_card == PHASE2_ENTROPY.target_card
    assert PHASE1_ENTROPY.coef_floor == PHASE2_ENTROPY.coef_floor


def test_log_reachable_never_returns_zero():
    """log(1) = 0 would divide by zero; a single-arm row carries no entropy and is
    masked anyway, so the floor only keeps the arithmetic finite.
    """
    assert log_reachable(1) == math.log(2)
    assert log_reachable(0) == math.log(2)
    assert log_reachable(5) == math.log(5)


# --- non-finite measurements ---
# One non-finite gradient drops every minibatch, so every reported mean is NaN;
# `exp(NaN)` would make the coefficient NaN, every later loss NaN, and the NaN
# would be checkpointed. The controller's state must never take a non-finite
# value. Also, `math.exp` raises OverflowError rather than saturating, and with
# no step cap nothing bounds its argument.

def test_a_nan_measurement_does_not_poison_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_placement
    c.update(0.35, float("nan"), 0)
    assert math.isfinite(c.coef_placement), c.coef_placement
    assert c.coef_placement == before, "a NaN reading must not move the coefficient"


def test_a_nan_on_ONE_head_leaves_the_other_head_working():
    """The two heads are independent: one unreadable measurement must not freeze
    the other.
    """
    c = EntropyController(PHASE1_ENTROPY)
    before_place = c.coef_placement
    c.update(0.99, float("nan"), 0)          # card far above target, placement unreadable
    assert c.coef_card < PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == before_place


def test_an_infinite_measurement_does_not_poison_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    for bad in (float("inf"), float("-inf")):
        c.coef_placement = 0.06
        c.update(0.35, bad, 0)
        assert math.isfinite(c.coef_placement), (bad, c.coef_placement)


def test_a_wild_measurement_does_not_raise_overflowerror():
    """With coef_step_max=None nothing bounds exp()'s argument; math.exp(1000.0)
    raises and would end the run.
    """
    c = EntropyController(PHASE1_ENTROPY)
    c.update(0.35, -1e6, 0)
    assert math.isfinite(c.coef_placement)
    assert PHASE1_ENTROPY.coef_floor <= c.coef_placement <= PHASE1_ENTROPY.coef_ceil_placement


def test_a_nan_coefficient_is_never_written_to_a_checkpoint():
    """Belt and braces: a NaN coefficient is never what a resume restores."""
    c = EntropyController(PHASE1_ENTROPY)
    c.coef_placement = float("nan")
    c.coef_card = float("nan")
    state = c.state_dict()
    assert math.isfinite(state["ent_coef_place"]), state
    assert math.isfinite(state["ent_coef_card"]), state


def test_a_poisoned_legacy_checkpoint_is_not_loaded():
    """A checkpoint written before this guard may carry a NaN; restoring it would
    look like the resume itself broke.
    """
    c = EntropyController(PHASE1_ENTROPY)
    c.load_state_dict({"ent_coef_card": float("nan"),
                       "ent_coef_place": float("nan")})
    assert math.isfinite(c.coef_card)
    assert math.isfinite(c.coef_placement)
    assert c.coef_card == PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == PHASE1_ENTROPY.initial_coef_placement


def test_a_healthy_measurement_still_moves_the_coefficient():
    """The guard must not freeze a working controller."""
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_placement
    c.update(0.35, 0.10, 0)      # far below target: push up
    assert c.coef_placement > before
