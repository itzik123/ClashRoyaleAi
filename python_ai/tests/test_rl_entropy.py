"""The entropy controller, and the guards its history says it needs.

This project has SEVEN recorded instances of an entropy normalizer that did not
hold in the regime it was measured in, three of them in this controller. The
tests here pin the three properties that would have caught them:

  * a coefficient the controller cannot walk out of its bounds
  * a per-update step cap that binds on a large excursion
  * a placement target that ANNEALS, on a clock the caller supplies
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
    # Past the horizon it must hold, not overshoot into negative entropy.
    assert c.placement_target(10 * cfg.anneal_episodes) == pytest.approx(
        cfg.target_placement_final)


def test_the_card_target_is_deliberately_NOT_annealed():
    """The measured failure mode for the card head is the opposite one: card
    entropy 0.10 collapsed the policy to 5 of 8 cards. Narrowing card choice is
    the known danger, so only placement anneals."""
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
    """Pre-registered prediction 4 of the 2026-08-16 fix, as a test.

    A fresh net is near-uniform over its legal arms -- ~1.0 of REACHABLE -- so
    it is far ABOVE the 0.35 target and the controller must push down. Under the
    old log(total-arms) divisor the same policy read 0.413 and the controller
    pushed UP, which is the whole bug in one number.
    """
    c = EntropyController(PHASE1_ENTROPY)
    start = c.coef_card
    c.update(card_frac=0.958, placement_frac=0.996, anneal_episodes_done=0)
    assert c.coef_card < start


def test_the_coefficient_cannot_leave_its_bounds():
    """0.002 was measurably an OFF SWITCH rather than a floor: once there the
    entropy term stopped opposing the policy gradient at all."""
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
    """The 2026-08-11 dissolution: gain 0.5 against an error of 0.16 multiplied
    the coefficient by 1.083 per update and compounded ~55x over 50 updates,
    reaching 0.433 and dissolving the policy. The cap bounds that walk at
    (1 +/- STEP_MAX)^n regardless of how wrong the error is."""
    cfg = PHASE2_ENTROPY
    assert cfg.coef_step_max is not None
    c = EntropyController(cfg)
    before = c.coef_placement
    # A target/measured gap far larger than anything real.
    c.update(card_frac=0.35, placement_frac=-50.0, anneal_episodes_done=0)
    assert c.coef_placement <= before * (1.0 + cfg.coef_step_max) + 1e-12


def test_phase1_has_no_step_cap_so_its_behaviour_is_unchanged():
    """Pipeline 1's controller is historical and was NOT retuned by this
    refactor. With `coef_step_max=None` the general form must reduce exactly to
    `clip(coef * exp(rate * error), floor, ceil)`."""
    cfg = PHASE1_ENTROPY
    assert cfg.coef_step_max is None
    c = EntropyController(cfg)
    start = c.coef_card
    c.update(card_frac=0.10, placement_frac=0.65, anneal_episodes_done=0)
    expected = min(cfg.coef_ceil_card,
                   start * math.exp(cfg.adapt_rate_card * (cfg.target_card - 0.10)))
    assert c.coef_card == pytest.approx(expected)


def test_the_anneal_clock_is_the_callers_not_the_raw_episode_count():
    """Until 2026-08-09 pipeline 2 passed the raw counter here, which made its
    stall re-boost dead code: it printed a message and changed nothing, fired
    twice at a measured pool win rate of 0.49, and the target carried on
    annealing straight through both."""
    c = EntropyController(PHASE2_ENTROPY)
    late = PHASE2_ENTROPY.anneal_episodes
    assert c.placement_target(late) < c.placement_target(0)
    # A re-boost hands back a small number, and the target must widen again.
    assert c.placement_target(0) > c.placement_target(late // 4)


def test_state_survives_a_checkpoint_roundtrip():
    """A phase-2 resume that silently reset the controller was observed on
    2026-07-30: placement went from a converged 0.0132 back to 0.06 and took
    ~5,600 episodes to walk back, with nothing warning."""
    c = EntropyController(PHASE2_ENTROPY)
    c.update(card_frac=0.1, placement_frac=0.1, anneal_episodes_done=0)
    saved = c.state_dict()
    restored = EntropyController(PHASE2_ENTROPY)
    restored.load_state_dict(saved)
    assert restored.coef_card == c.coef_card
    assert restored.coef_placement == c.coef_placement


def test_a_checkpoint_without_the_keys_keeps_the_seeded_defaults():
    """An older checkpoint must resume, not KeyError."""
    c = EntropyController(PHASE1_ENTROPY)
    c.load_state_dict({"episodes_completed": 5})
    assert c.coef_card == PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == PHASE1_ENTROPY.initial_coef_placement


def test_the_two_pipelines_keep_their_measured_differences():
    """These are NOT unified, and the test says so out loud: changing either
    would be gameplay-affecting in whichever pipeline moved."""
    assert PHASE1_ENTROPY.target_placement_start == 0.65
    assert PHASE2_ENTROPY.target_placement_start == 0.50
    assert PHASE1_ENTROPY.adapt_rate_placement == 0.5
    assert PHASE2_ENTROPY.adapt_rate_placement == 0.10
    assert PHASE1_ENTROPY.coef_ceil_placement == 0.5
    assert PHASE2_ENTROPY.coef_ceil_placement == 0.20
    # ...and everything they share really is shared.
    assert PHASE1_ENTROPY.target_card == PHASE2_ENTROPY.target_card
    assert PHASE1_ENTROPY.coef_floor == PHASE2_ENTROPY.coef_floor


def test_log_reachable_never_returns_zero():
    """log(1) = 0 would divide the entropy fraction by zero. A single-arm row
    carries zero entropy and is masked out anyway, so the floor only has to
    keep the arithmetic finite."""
    assert log_reachable(1) == math.log(2)
    assert log_reachable(0) == math.log(2)
    assert log_reachable(5) == math.log(5)
