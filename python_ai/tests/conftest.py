"""Fixtures shared by the whole python_ai suite.

These two moved here VERBATIM from the old single `test_python_ai.py` when it was
split on 2026-08-20. Note `fresh_obs` returns `(env, obs)` and not a tensor:
several tests step that env afterwards, and "improving" the shape on the way out
is what broke the first attempt at the split.

`python_ai/conftest.py` (one level up) is what puts the repo root and the
compiled engine's directory on sys.path; this file only holds fixtures.
"""
import pytest

from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet

import clash_royale_env

CE = clash_royale_env.ClashRoyaleEnv


@pytest.fixture(scope="module")
def net():
    return MicroRoyaleNet(num_ability_slots=0)


@pytest.fixture(scope="module")
def fresh_obs():
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    return env, env.get_observation_for_team(0)
