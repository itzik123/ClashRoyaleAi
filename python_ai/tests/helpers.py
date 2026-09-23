"""Plain builder functions shared by more than one test file (fixtures live in
conftest.py).
"""
import numpy as np
import torch

import clash_royale_env
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN

CE = clash_royale_env.ClashRoyaleEnv

#: A tiny truncated-BPTT chunk, 3 timesteps x 2 envs: cheap for exact-zero
#: gradient assertions, large enough to exercise the update's (L, B) reshaping.
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
    # artifact.
    card_mask = torch.ones(L, B, net.hand_size + 1, dtype=torch.bool)
    resets = torch.ones(L, B)
    hidden = (torch.zeros(B, LSTM_HIDDEN), torch.zeros(B, LSTM_HIDDEN))
    hand = net.hand_card_ids(flat)[0].tolist()
    return (net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask,
            resets, hidden, hand)


def shaping_stats(fireball_killed):
    """Minimal stats/prev pair where a Fireball has just killed some value.

    The spell is supplied through `spell_damage` / `spell_cost` as the envs
    publish it (689 / 4, pinned by test_damage_spell_is_deck_derived). A
    fixture, since these tests pin the formula and must not move with
    CLASH_DECK.
    """
    z = np.zeros(1, dtype=np.float32)
    base = {
        "team0_troop_damage": z.copy(), "team1_troop_damage": z.copy(),
        "team0_building_damage": z.copy(), "team1_building_damage": z.copy(),
        "team0_tower_damage": z.copy(), "team1_tower_damage": z.copy(),
        "team0_elixir_spent": z.copy(), "team1_elixir_spent": z.copy(),
        "team0_elixir_current": np.array([7.0], dtype=np.float32),
        "team0_towers_alive": np.array([3]), "team1_towers_alive": np.array([3]),
        "enemy_tower_hp": np.zeros((1, 3), dtype=np.float32),
        "spell_in_hand": z.copy(),
        "spell_value_killed": z.copy(), "spell_elixir_spent": z.copy(),
        "spell_damage": np.array([689.0], dtype=np.float32),
        "spell_cost": np.array([4.0], dtype=np.float32),
    }
    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["spell_value_killed"] = np.array([fireball_killed], dtype=np.float32)
    cur["spell_elixir_spent"] = np.array([4.0], dtype=np.float32)
    return cur, prev
