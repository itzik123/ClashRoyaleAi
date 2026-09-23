"""The discount horizon is long enough that winning still outranks the proxies
paid along the way.

Asserts relationships, not a particular gamma (a tuning decision with a real
variance cost): the terminal outcome must not be dwarfed by a single
mid-episode bonus. Episode length is derived from the env's own `max_ticks`
default and `step`'s `skip_frames` default.
"""
import inspect
import math

import pytest

from python_ai.envs import gym_wrapper
from python_ai.rl.config import PPOConfig
from python_ai.rewards import weights


def episode_decisions():
    """Decisions in a full-length match, from the two engine defaults read by
    reflection.
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
    """Guards the derivation: if either default moves, every assertion below must
    be re-derived.
    """
    n = episode_decisions()
    assert n == 360, (
        f"full-length match is {n} decisions; if the engine's max_ticks or "
        "skip_frames changed, the reward-horizon numbers in this file and in "
        "rewards/weights.py must both be re-derived")


def test_terminal_win_is_not_dwarfed_by_a_single_crown_bonus():
    """The invariant: W_TOWER_DESTROYED is meant to sit above the discounted value
    of a win, but at a large multiple the sub-goal replaces the goal. At most
    3x.
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
    """The same fact in tuning units: the effective horizon 1/(1-gamma) must reach
    the end of the episode, or every terminal quantity is discounted into
    irrelevance together.
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
    """Discounted, DRAW_PENALTY must still outweigh one step of the
    elixir-overflow penalty, the cheapest continuous term.
    """
    n = episode_decisions()
    gamma = PPOConfig().gamma
    discounted_draw = weights.DRAW_PENALTY * gamma ** n
    assert discounted_draw >= weights.W_ELIXIR_OVERFLOW, (
        f"DRAW_PENALTY discounts to {discounted_draw:.4f} at episode start, "
        f"below a single step of W_ELIXIR_OVERFLOW ({weights.W_ELIXIR_OVERFLOW}). "
        "A terminal penalty that cannot outweigh one step of a continuous one "
        "is not shaping behaviour at the timescale it was written for.")
