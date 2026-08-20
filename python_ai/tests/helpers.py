"""Builders shared by more than one test file.

Both of these were defined in one section of the old `test_python_ai.py` and
used from another -- which is exactly why splitting the monolith surfaced them:
in one file a cross-section reference is invisible, across files it is an
ImportError. They are here rather than in `conftest.py` because they are plain
functions, not fixtures, and the tests call them with arguments.
"""
import numpy as np
import torch

import clash_royale_env
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN

CE = clash_royale_env.ClashRoyaleEnv

#: A deliberately tiny truncated-BPTT chunk: 3 timesteps x 2 envs. Small enough
#: that an exact-zero gradient assertion is cheap, large enough that the (L, B)
#: reshaping in the update is actually exercised.
L, B = 3, 2


def chunk_fixture():
    """A tiny (L,B) chunk built from real observations, with a real hand."""
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    obs = torch.tensor(env.get_observation_for_team(0), dtype=torch.float32)
    obs_seq = obs.view(1, 1, -1).expand(L, B, -1).contiguous()

    flat = obs_seq.view(L * B, -1)
    feats, embeds, spatial = net.extract_features(flat)
    feats_seq = feats.view(L, B, -1)
    embeds_seq = embeds.view(L, B, *embeds.shape[1:])
    spatial_seq = spatial.view(L, B, *spatial.shape[1:])
    # Everything affordable, so "unchosen" is a real choice and not a mask
    # artifact -- otherwise the test could pass for the wrong reason.
    card_mask = torch.ones(L, B, net.hand_size + 1, dtype=torch.bool)
    resets = torch.ones(L, B)
    hidden = (torch.zeros(B, LSTM_HIDDEN), torch.zeros(B, LSTM_HIDDEN))
    hand = net.hand_card_ids(flat)[0].tolist()
    return (net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask,
            resets, hidden, hand)


def shaping_stats(fireball_killed):
    """Minimal stats/prev pair where a Fireball has just killed some value."""
    z = np.zeros(1, dtype=np.float32)
    base = {
        "team0_troop_damage": z.copy(), "team1_troop_damage": z.copy(),
        "team0_building_damage": z.copy(), "team1_building_damage": z.copy(),
        "team0_tower_damage": z.copy(), "team1_tower_damage": z.copy(),
        "team0_elixir_spent": z.copy(), "team1_elixir_spent": z.copy(),
        "team0_elixir_current": np.array([7.0], dtype=np.float32),
        "team0_towers_alive": np.array([3]), "team1_towers_alive": np.array([3]),
        "enemy_tower_hp": np.zeros((1, 3), dtype=np.float32),
        "fireball_in_hand": z.copy(),
        "fireball_value_killed": z.copy(), "fireball_elixir_spent": z.copy(),
    }
    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["fireball_value_killed"] = np.array([fireball_killed], dtype=np.float32)
    cur["fireball_elixir_spent"] = np.array([4.0], dtype=np.float32)
    return cur, prev
