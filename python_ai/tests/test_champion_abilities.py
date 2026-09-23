"""Champion / Hero ability training. Three facts, each silent if wrong:

* Engine slots are deck indices 1 and 2, not "the first and second Champion": a
  lone Champion at index 2 builds one head, which must drive engine slot 2.
* The activate arm is masked by readiness. The engine refuses an unready
  activation, so an unmasked arm would put noise log-probs into the PPO ratio.
* The log-prob stored at rollout and the one the update recomputes must agree,
  or the ratio is not 1 at epoch 0.
"""
import numpy as np
import pytest
import torch

from python_ai.rl import abilities as A


def test_a_single_champion_in_deck_slot_two_drives_engine_slot_two():
    #          0          1            2 (Golden Knight)  ...
    deck = [6, 40, 116, 24, 72, 33, 7, 0]
    assert A.ability_engine_slots(deck) == [2]
    acts = A.action_dict(torch.tensor([[1], [0]]), [2], num_envs=2)
    assert acts["activate_ability_slot1"].tolist() == [0, 0]
    assert acts["activate_ability_slot2"].tolist() == [1, 0]


def test_no_champion_means_no_slots_and_all_zero_actions():
    assert A.ability_engine_slots([15, 6, 25, 40, 24, 72, 33, 7]) == []
    acts = A.action_dict(torch.zeros((3, 0), dtype=torch.long), [], num_envs=3)
    assert acts["activate_ability_slot1"].tolist() == [0, 0, 0]
    assert acts["activate_ability_slot2"].tolist() == [0, 0, 0]


def test_two_champions_map_in_slot_order():
    assert A.ability_engine_slots([6, 116, 118, 24, 72, 33, 7, 0]) == [1, 2]


def test_readiness_is_read_per_engine_slot_and_defaults_to_not_ready():
    infos = {"champion_ability_slot2_ready": np.array([True, False])}
    ready = A.ready_from_infos(infos, num_envs=2, engine_slots=[2])
    assert ready.tolist() == [[True], [False]]
    assert A.ready_from_infos({}, 2, [1, 2]).tolist() == [[False, False]] * 2


def test_a_not_ready_ability_can_only_be_declined_with_probability_one():
    logits = [torch.tensor([[0.0, 5.0], [0.0, 5.0]])]
    ready = torch.tensor([[True], [False]])
    actions, logprob, entropy = A.sample(logits, ready)
    assert int(actions[1, 0]) == 0
    assert float(logprob[1]) == pytest.approx(0.0)
    assert float(entropy[1, 0]) == pytest.approx(0.0)
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()


def test_log_prob_recomputation_matches_sampling():
    torch.manual_seed(0)
    logits = [torch.randn(16, 2), torch.randn(16, 2)]
    ready = torch.rand(16, 2) > 0.3
    actions, logprob, _ = A.sample(logits, ready)
    again, _ = A.log_prob_and_entropy(logits, ready, actions)
    assert torch.allclose(again, logprob)
