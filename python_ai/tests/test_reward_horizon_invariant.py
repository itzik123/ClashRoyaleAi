"""The discount horizon must be long enough that WINNING still outranks the
proxies we pay for along the way.

WHY THIS FILE EXISTS. `rewards/weights.py` sets `W_TOWER_DESTROYED = 0.6` with
the comment "Set above the discounted value of a win (~0.28 at these episode
lengths)". That 0.28 was measured over episodes averaging **112 decisions**.
The 2026-08-07 movement-speed fix made matches roughly three times longer --
CLAUDE.md records "no win rate below survives it" -- and the reward constants
were never re-derived against the new length.

Measured 2026-08-27 with `model_weights_phase1_v5.pth`, greedy, 10 episodes:

    decisions/episode          mean 297, median 360 (== maxTicks bound)
    terminal contribution to
      the discounted return    0.0195
    win discounted to t=0      0.99^360 = 0.0268
    W_TOWER_DESTROYED          0.6000, undiscounted, fires mid-episode
    ratio                      22.4x

So the agent is paid twenty-two times more for taking one crown than for
winning the game that crown is supposed to serve, and ~98% of its objective is
shaping. That is not a bug in any single line; it is the horizon and the
constants having drifted apart. It caps strategic depth directly: a sacrifice
-- give up tower HP now, win later -- costs 0.5 immediately and pays 0.027 at
the end, so no such policy can ever be found by gradient ascent on this
objective.

WHAT THIS TEST PINS, and what it deliberately does NOT. It does not assert a
particular gamma; that is a tuning decision with a real variance cost and it
belongs to whoever runs the training. It asserts the RELATIONSHIP the reward
design already claims to hold in its own comments -- that the terminal outcome
is not dwarfed by a single mid-episode bonus -- so the pair can never silently
rot apart again the way it did across the speed fix.

Everything here is DERIVED from the engine and the config rather than restated,
per CLAUDE.md's no-second-copies rule: episode length comes from the env's own
`max_ticks` default and `step`'s own `skip_frames` default, not from a literal.
"""
import inspect
import math

import pytest

from python_ai.envs import gym_wrapper
from python_ai.rl.config import PPOConfig
from python_ai.rewards import weights


def episode_decisions():
    """Decisions in a full-length match, derived from the two engine defaults.

    `max_ticks` is the env's own default and `skip_frames` is `step`'s own
    default, both read by reflection. Hardcoding 360 here would be exactly the
    second copy that let the reward constants drift out of date in the first
    place -- if either default moves, this must move with it.
    """
    src = inspect.getsource(gym_wrapper.MicroRoyaleEnv.__init__)
    # env_config.get("max_ticks", <default>)
    import re
    m = re.search(r'max_ticks["\']\s*,\s*(\d+)', src)
    assert m, "could not derive max_ticks from MicroRoyaleEnv.__init__"
    max_ticks = int(m.group(1))
    skip = inspect.signature(gym_wrapper.MicroRoyaleEnv.step).parameters[
        "skip_frames"].default
    return max_ticks // skip


def test_episode_length_is_derived_and_matches_the_engine():
    """Guards the derivation itself: a silent change to either default that
    this helper stopped tracking would make every assertion below meaningless."""
    n = episode_decisions()
    assert n == 360, (
        f"full-length match is {n} decisions; if the engine's max_ticks or "
        "skip_frames changed, the reward-horizon numbers in this file and in "
        "rewards/weights.py must both be re-derived")


def test_terminal_win_is_not_dwarfed_by_a_single_crown_bonus():
    """THE INVARIANT. `weights.py` states its own intent: W_TOWER_DESTROYED is
    "set above the discounted value of a win" so that "taking a crown is never
    worth less than the trade that led to it". That is a sane goal at a small
    multiple. At a LARGE multiple it inverts the objective -- the sub-goal
    stops serving the goal and replaces it.

    Bound of 3x: a crown may outrank a win somewhat (that is the deliberate,
    documented bias), but a full match's actual outcome must stay the same
    ORDER OF MAGNITUDE as one intermediate bonus, or the terminal signal is
    not participating in learning at all.
    """
    n = episode_decisions()
    gamma = PPOConfig().gamma
    win_value = gamma ** n
    ratio = weights.W_TOWER_DESTROYED / win_value
    assert ratio <= 3.0, (
        f"gamma={gamma} over a {n}-decision match discounts a win to "
        f"{win_value:.4f}, while W_TOWER_DESTROYED pays "
        f"{weights.W_TOWER_DESTROYED} undiscounted -- a crown is worth "
        f"{ratio:.1f}x winning the game. The reward function ranks the "
        "sub-goal above the goal. Either raise gamma so the horizon covers "
        "the match, or lower W_TOWER_DESTROYED to match the horizon; the two "
        "are one decision and must be made together.")


def test_discount_horizon_covers_a_full_length_match():
    """The effective horizon 1/(1-gamma) must reach the end of the episode.

    This is the same fact as the test above stated in the units people
    actually tune in, and it is the one that generalizes past this particular
    bonus: with a horizon of 100 decisions on a 360-decision match, EVERY
    terminal quantity -- the win, the loss, DRAW_PENALTY, the flawless-defense
    bonus -- is discounted into irrelevance together, not just the crown.
    """
    n = episode_decisions()
    gamma = PPOConfig().gamma
    horizon = 1.0 / (1.0 - gamma)
    assert horizon >= n, (
        f"effective horizon is {horizon:.0f} decisions but a full match is "
        f"{n}; terminal rewards decay to {gamma ** n:.4f} of their face value. "
        "DRAW_PENALTY=1.0 was raised from 0.2 specifically to stop stalling "
        f"and is currently worth {gamma ** n:.4f} at episode start.")


def test_draw_penalty_retains_meaningful_weight_at_episode_end():
    """DRAW_PENALTY was raised 0.2 -> 1.0 to make a draw "as costly as an
    outright loss". Discounted, it must still be able to outweigh the
    per-step incentives it was raised to beat -- here, one step of the
    elixir-overflow penalty, the cheapest continuous term on the board.
    """
    n = episode_decisions()
    gamma = PPOConfig().gamma
    discounted_draw = weights.DRAW_PENALTY * gamma ** n
    assert discounted_draw >= weights.W_ELIXIR_OVERFLOW, (
        f"DRAW_PENALTY discounts to {discounted_draw:.4f} at episode start, "
        f"below a single step of W_ELIXIR_OVERFLOW ({weights.W_ELIXIR_OVERFLOW}). "
        "A terminal penalty that cannot outweigh one step of a continuous one "
        "is not shaping behaviour at the timescale it was written for.")
