"""Fixtures shared by the python_ai suite.

`fresh_obs` returns `(env, obs)`, not a tensor: several tests step that env
afterwards. `python_ai/conftest.py` puts the repo root and the engine on
sys.path.
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
