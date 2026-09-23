"""The adaptive per-head entropy controller.

One step per PPO update, so it moves far slower than training.

It only ever sees entropies as fractions of each head's reachable maximum,
log(n legal arms), computed per step in rl/ppo.py. The invariant: a uniform
distribution over the legal arms reads exactly 1.0 at any number of arms.
Normalizing any other way (raw nats, log of all arms, averaging over steps that
placed nothing) has broken this controller three times.
"""
import math

import numpy as np


class EntropyController:
    """Holds the two coefficients and moves them toward their targets.

    Checkpointed: a resume that reset them once took ~5,600 episodes to
    recover, silently.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.coef_card = cfg.initial_coef_card
        self.coef_placement = cfg.initial_coef_placement
        #: Updates skipped for an unreadable measurement. Not checkpointed;
        #: non-zero means the PPO update upstream produced nothing to average.
        self.frozen_updates = 0

    def placement_target(self, episodes_done):
        """The annealed placement target.

        `episodes_done` is on the anneal clock: pipeline 2 subtracts its last
        stall re-boost, which is what lets a re-boost stop the target
        sharpening.
        """
        c = self.cfg
        frac = min(1.0, max(0.0, episodes_done / c.anneal_episodes))
        return (c.target_placement_start
                + frac * (c.target_placement_final - c.target_placement_start))

    @property
    def card_target(self):
        return self.cfg.target_card

    # exp() raises past ~709 instead of saturating, and with coef_step_max None
    # nothing else bounds the exponent. Anything past exp(50) clips to the
    # ceiling anyway.
    _MAX_EXPONENT = 50.0

    def _step_one(self, coef, rate, target, measured, ceil):
        # A non-finite measurement (the non-finite guard in rl/ppo.py dropped
        # every minibatch) holds this head's coefficient. Otherwise NaN would
        # spread to the coefficient, every later loss, and the checkpoint.
        if not math.isfinite(measured):
            self.frozen_updates += 1
            return coef
        exponent = float(np.clip(rate * (target - measured),
                                 -self._MAX_EXPONENT, self._MAX_EXPONENT))
        step = math.exp(exponent)
        if self.cfg.coef_step_max is not None:
            # Cap the ratio, not the coefficient: a multiplicative controller
            # compounds, and this bounds the walk at (1 +/- STEP_MAX) per
            # update however large the error.
            lo, hi = 1.0 - self.cfg.coef_step_max, 1.0 + self.cfg.coef_step_max
            step = min(hi, max(lo, step))
        return float(np.clip(coef * step, self.cfg.coef_floor, ceil))

    def update(self, card_frac, placement_frac, anneal_episodes_done):
        """Move both coefficients one step and return the placement target used.

        `card_frac` / `placement_frac` must already be fractions of each head's
        reachable maximum.
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

    def _safe(self, value, fallback):
        """`value` unless it is non-finite, else the seeded default. Applied on
        save and on load, so a poisoned coefficient cannot survive a
        checkpoint.
        """
        return float(value) if math.isfinite(value) else float(fallback)

    def state_dict(self):
        return {"ent_coef_card": self._safe(self.coef_card,
                                            self.cfg.initial_coef_card),
                "ent_coef_place": self._safe(self.coef_placement,
                                             self.cfg.initial_coef_placement)}

    def load_state_dict(self, state):
        """Restore from a checkpoint; a missing key keeps the seeded default.
        """
        self.coef_card = self._safe(
            state.get("ent_coef_card", self.coef_card), self.cfg.initial_coef_card)
        self.coef_placement = self._safe(
            state.get("ent_coef_place", self.coef_placement),
            self.cfg.initial_coef_placement)
