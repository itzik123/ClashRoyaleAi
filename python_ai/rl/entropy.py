"""The adaptive per-head entropy controller.

One controller step per PPO update (~27 s of wall clock), so it moves far slower
than training and cannot fight the policy gradient step for step.

THE NORMALIZATION IS THE LOAD-BEARING PART, and getting it wrong is this
project's single most repeated defect -- seven instances on record, three of
them in this exact controller:

  2026-07-31  the exploiter used RAW NATS where the main loop used fractions,
              making its placement coefficient effectively 73x too large. The
              policy dissolved to 86.7% of maximum placement entropy and went
              166-839-2 against the agent it was a bit-exact copy of.
  2026-08-11  placement entropy was averaged over steps where a card was
              AFFORDABLE, not steps where one was PLAYED. Reported 0.462 of max
              decomposed into 0.850 on no-op steps against 0.090 on real
              placements. The controller saw a surplus and pinned the
              coefficient to its floor for 57% of updates while the
              distribution that actually places cards collapsed.
  2026-08-16  both heads divided by log(TOTAL arms) instead of log(REACHABLE
              arms). The affordability mask leaves ~2 legal card arms on 54.1%
              of decisions, where the reachable maximum is log(2) = 0.693 --
              so a 0.35 * log(5) target was 81.3% of it, and the controller
              held the play/wait choice near a coin flip. Downstream: elixir
              spent on sight, mean elixir 2.25/10, nothing affordable on 73.9%
              of steps, P(play) FLAT against threat.

The invariant that prevents all three: **a uniform distribution over the legal
arms must read exactly 1.0, at any number of arms.** `rl/ppo.py` does that
division per step; this class only ever sees fractions.
"""
import math

import numpy as np


class EntropyController:
    """Holds the two coefficients and moves them toward their targets.

    State is (coef_card, coef_placement) and it is CHECKPOINTED. It has to be:
    a phase-2 resume that silently threw the controller back to 0.05/0.06 was
    observed on 2026-07-30 to reset placement from a converged 0.0132 and take
    ~5,600 episodes to walk back to 0.0205, with nothing warning.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.coef_card = cfg.initial_coef_card
        self.coef_placement = cfg.initial_coef_placement
        #: Updates whose measurement was unreadable and whose step was skipped.
        #: Not checkpointed -- it describes THIS process's health, not the
        #: run's state. Non-zero means the PPO update upstream produced nothing
        #: to average, which is a numerical fault worth chasing, not a quirk.
        self.frozen_updates = 0

    # -- targets ------------------------------------------------------------
    def placement_target(self, episodes_done):
        """The annealed placement target.

        `episodes_done` is measured on the ANNEAL clock, which is not always the
        raw episode counter: pipeline 2 passes `episodes_completed -
        entropy_reboost_episode`, so a stall re-boost actually stops the target
        sharpening. Until 2026-08-09 it passed the raw counter, which made the
        re-boost dead code -- it printed a message and changed nothing, fired
        twice at a measured pool win rate of 0.49 both times, and the target
        carried on annealing straight through.
        """
        c = self.cfg
        frac = min(1.0, max(0.0, episodes_done / c.anneal_episodes))
        return (c.target_placement_start
                + frac * (c.target_placement_final - c.target_placement_start))

    @property
    def card_target(self):
        return self.cfg.target_card

    # -- the controller step ------------------------------------------------
    #: Largest exponent handed to `math.exp`. exp() RAISES OverflowError past
    #: ~709 rather than saturating, and with `coef_step_max=None` -- the PHASE1
    #: default, "None disables it" -- nothing else bounds the argument. exp(50)
    #: is already ~5e21, so anything beyond it is clipped to `ceil` regardless
    #: and the bound costs no reachable behaviour.
    _MAX_EXPONENT = 50.0

    def _step_one(self, coef, rate, target, measured, ceil):
        # A NON-FINITE MEASUREMENT FREEZES THIS HEAD, and only this head.
        #
        # `measured` is a mean over the PPO update's minibatches, and it is NaN
        # whenever there were none to average -- which is exactly what happens
        # when the non-finite guard in `rl/ppo.py` drops them all. Without this
        # check the chain runs: exp(NaN) = NaN -> coef = NaN -> the entropy
        # bonus is NaN -> every future loss is NaN -> every future update is
        # dropped -> and `state_dict` writes the NaN to the checkpoint, so a
        # resume reloads the dead run. The controller is the link that turns a
        # transient numerical fault into a permanent one.
        #
        # Holding the coefficient is the right response, not resetting it: one
        # unreadable update is no evidence the coefficient is wrong, and the
        # next healthy update moves it normally.
        if not math.isfinite(measured):
            self.frozen_updates += 1
            return coef
        exponent = float(np.clip(rate * (target - measured),
                                 -self._MAX_EXPONENT, self._MAX_EXPONENT))
        step = math.exp(exponent)
        if self.cfg.coef_step_max is not None:
            # Cap the RATIO, not the coefficient. Any multiplicative controller
            # compounds, so a signal that suddenly reads far from target walks
            # the coefficient exponentially before the policy can respond;
            # capping the ratio bounds that walk at (1 +/- STEP_MAX)^n_updates
            # regardless of how wrong the error is. Binds only on large
            # excursions -- exactly when it should.
            lo, hi = 1.0 - self.cfg.coef_step_max, 1.0 + self.cfg.coef_step_max
            step = min(hi, max(lo, step))
        return float(np.clip(coef * step, self.cfg.coef_floor, ceil))

    def update(self, card_frac, placement_frac, anneal_episodes_done):
        """Move both coefficients one step. Returns the placement target used,
        which the caller logs next to the measured value -- that comparison is
        exactly what diagnosed the fixed-target pathology in the first place.

        `card_frac` / `placement_frac` are ALREADY fractions of each head's
        REACHABLE maximum (see the module docstring). Passing raw nats here is
        the 2026-07-31 failure.
        """
        c = self.cfg
        target_placement = self.placement_target(anneal_episodes_done)
        self.coef_card = self._step_one(
            self.coef_card, c.adapt_rate_card, c.target_card, card_frac,
            c.coef_ceil_card)
        self.coef_placement = self._step_one(
            self.coef_placement, c.adapt_rate_placement, target_placement,
            placement_frac, c.coef_ceil_placement)
        return target_placement

    # -- persistence --------------------------------------------------------
    def _safe(self, value, fallback):
        """`value` unless it is non-finite, in which case the seeded default.

        Applied on the way OUT and on the way IN. A coefficient is a number the
        run cannot recover from by itself, so a poisoned one must not survive a
        checkpoint in either direction -- including in a checkpoint written
        before this guard existed.
        """
        return float(value) if math.isfinite(value) else float(fallback)

    def state_dict(self):
        return {"ent_coef_card": self._safe(self.coef_card,
                                            self.cfg.initial_coef_card),
                "ent_coef_place": self._safe(self.coef_placement,
                                             self.cfg.initial_coef_placement)}

    def load_state_dict(self, state):
        """Restore from a checkpoint, keeping the seeded default for a key an
        older checkpoint does not carry."""
        self.coef_card = self._safe(
            state.get("ent_coef_card", self.coef_card), self.cfg.initial_coef_card)
        self.coef_placement = self._safe(
            state.get("ent_coef_place", self.coef_placement),
            self.cfg.initial_coef_placement)
