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


# --- non-finite measurements ----------------------------------------------
#
# THE CASCADE THIS PREVENTS, end to end:
#
#   one non-finite gradient
#     -> PPOUpdater drops every minibatch in the update
#     -> every reported mean is NaN (there is nothing to average)
#     -> EntropyController.update(NaN) sets coef = exp(NaN) = NaN
#     -> the entropy bonus is NaN, so EVERY future loss is NaN
#     -> every future update is dropped too
#     -> and the NaN coefficient is CHECKPOINTED, so a resume reloads it
#
# The controller is the link that makes a transient fault permanent: the PPO
# guard protects the weights, but nothing protected the controller's own state,
# and its state is written to disk. A coefficient is a number the run cannot
# recover from on its own.
#
# `math.exp` is also the one call here that RAISES rather than saturating:
# math.exp(1000.0) is an OverflowError, not inf. With coef_step_max=None --
# which is the PHASE1 default, "None disables it" -- nothing bounds the
# argument, so a wild measurement crashes the process outright.

def test_a_nan_measurement_does_not_poison_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_placement
    c.update(0.35, float("nan"), 0)
    assert math.isfinite(c.coef_placement), c.coef_placement
    assert c.coef_placement == before, "a NaN reading must not move the coefficient"


def test_a_nan_on_ONE_head_leaves_the_other_head_working():
    """The two heads are independent measurements. Freezing both because one
    is unreadable would silently disable the card controller too."""
    c = EntropyController(PHASE1_ENTROPY)
    before_place = c.coef_placement
    c.update(0.99, float("nan"), 0)          # card way above target, placement unreadable
    assert c.coef_card < PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == before_place


def test_an_infinite_measurement_does_not_poison_the_coefficient():
    c = EntropyController(PHASE1_ENTROPY)
    for bad in (float("inf"), float("-inf")):
        c.coef_placement = 0.06
        c.update(0.35, bad, 0)
        assert math.isfinite(c.coef_placement), (bad, c.coef_placement)


def test_a_wild_measurement_does_not_raise_overflowerror():
    """PHASE1 has coef_step_max=None, so nothing bounds exp()'s argument.
    math.exp(1000.0) RAISES -- it does not saturate -- and an uncaught
    OverflowError in the controller ends the run."""
    c = EntropyController(PHASE1_ENTROPY)
    c.update(0.35, -1e6, 0)
    assert math.isfinite(c.coef_placement)
    assert PHASE1_ENTROPY.coef_floor <= c.coef_placement <= PHASE1_ENTROPY.coef_ceil_placement


def test_a_nan_coefficient_is_never_written_to_a_checkpoint():
    """Belt and braces: even if one were reached some other way, it must not
    be the thing a resume restores."""
    c = EntropyController(PHASE1_ENTROPY)
    c.coef_placement = float("nan")
    c.coef_card = float("nan")
    state = c.state_dict()
    assert math.isfinite(state["ent_coef_place"]), state
    assert math.isfinite(state["ent_coef_card"]), state


def test_a_poisoned_legacy_checkpoint_is_not_loaded():
    """A checkpoint written before this guard existed can already carry a NaN.
    Restoring it would reinstate the dead run on resume, which is the failure
    mode that is hardest to attribute -- it looks like the resume itself broke.
    """
    c = EntropyController(PHASE1_ENTROPY)
    c.load_state_dict({"ent_coef_card": float("nan"),
                       "ent_coef_place": float("nan")})
    assert math.isfinite(c.coef_card)
    assert math.isfinite(c.coef_placement)
    assert c.coef_card == PHASE1_ENTROPY.initial_coef_card
    assert c.coef_placement == PHASE1_ENTROPY.initial_coef_placement


def test_a_healthy_measurement_still_moves_the_coefficient():
    """The guard must not freeze a working controller -- that would disable
    the only defence against mode collapse."""
    c = EntropyController(PHASE1_ENTROPY)
    before = c.coef_placement
    c.update(0.35, 0.10, 0)      # far below target -> push UP
    assert c.coef_placement > before
