"""The auxiliary task predicts the opponent's next card.

Unlike opponent elixir (an affine function of two observed scalars, see
test_aux_task_is_not_a_memory_probe.py), the next card needs the opponent's
play history, so it asks something of the recurrent state. Most coverage goes
to the label transform: an off-by-one or a missed episode boundary would
silently train the net to predict the next match's opening from this match's
end, visible only as a plausible loss curve.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
from python_ai.rl.config import PPOConfig  # noqa: E402
from python_ai.rl.engine_stats import (next_card_labels,  # noqa: E402
                                       opponent_played_card)


# --- labels ---

def test_label_is_the_next_play_at_or_after_each_step():
    # T=4, N=1: nothing, Hog, nothing, Log; one episode.
    played = torch.tensor([[-1], [15], [-1], [33]])
    masks = torch.ones(4, 1)
    labels, has = next_card_labels(played, masks, torch.ones(4, 1))
    assert labels.squeeze(-1).tolist() == [15, 15, 33, 33]
    assert has.squeeze(-1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_a_step_that_plays_is_labelled_with_its_own_card():
    """The play during step t is in t's future: the agent acted on the observation
    at the top of the step, before the opponent played.
    """
    played = torch.tensor([[15], [33]])
    labels, has = next_card_labels(played, torch.ones(2, 1), torch.ones(2, 1))
    assert labels.squeeze(-1).tolist() == [15, 33]


def test_the_carry_does_not_leak_across_an_episode_boundary():
    """The regression this file exists for: masks[t] == 0 marks the step an
    episode ended on, and the backward carry must clear there, or step 2 below
    would be labelled with a card from a different match.
    """
    played = torch.tensor([[-1], [15], [-1], [-1], [33], [-1]])
    masks = torch.ones(6, 1)
    masks[3, 0] = 0.0                     # episode ends at t=3
    labels, has = next_card_labels(played, masks, torch.ones(6, 1))

    assert has.squeeze(-1).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert labels[0].item() == 15 and labels[1].item() == 15
    assert labels[4].item() == 33         # its own play, same episode
    # t=2 and t=3 have no play left in their episode: dropped, not 33.
    assert has[2].item() == 0.0 and has[3].item() == 0.0


def test_steps_with_no_future_play_are_dropped_not_given_a_class():
    """"They never played again" is an artifact of where the episode stopped; as a
    class it would be the majority label.
    """
    played = torch.full((5, 2), -1)
    labels, has = next_card_labels(played, torch.ones(5, 2), torch.ones(5, 2))
    assert has.sum().item() == 0.0


def test_unlabelled_rows_still_carry_a_legal_index():
    """cross_entropy indexes with the label tensor even on masked rows, so -1
    would raise before the mask discards it.
    """
    played = torch.full((3, 2), -1)
    labels, _ = next_card_labels(played, torch.ones(3, 2), torch.ones(3, 2))
    assert int(labels.min()) >= 0


def test_valid_gates_the_label_mask():
    """Phantom auto-reset steps are excluded even when a label exists."""
    played = torch.tensor([[15], [33]])
    valid = torch.tensor([[0.0], [1.0]])
    _, has = next_card_labels(played, torch.ones(2, 1), valid)
    assert has.squeeze(-1).tolist() == [0.0, 1.0]


def test_missing_info_key_defaults_to_no_play():
    """The all-envs-auto-reset step drops the key; the default must contribute
    exactly zero, as in engine_stats.
    """
    assert opponent_played_card({}, 4).tolist() == [-1, -1, -1, -1]


# --- head ---

def test_the_head_is_a_card_classifier_and_the_elixir_head_is_gone():
    import clash_royale_env
    from python_ai.models.net import MicroRoyaleNet
    net = MicroRoyaleNet(num_ability_slots=0)
    assert not hasattr(net, "aux_elixir_head")
    assert not hasattr(net, "predict_opp_elixir")
    n_cards = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS
    out = net.predict_opp_next_card(torch.zeros(3, net.LSTM_HIDDEN))
    assert out.shape == (3, n_cards)


def test_the_coefficients_moved_with_the_task():
    cfg = PPOConfig()
    assert not hasattr(cfg, "aux_elixir_coef")
    assert cfg.aux_card_coef == 0.5
    assert cfg.aux_card_scale == 0.02


# --- env ---

def test_the_env_reports_which_card_the_opponent_played():
    """End to end: the signal arrives, and is -1 on the first step rather than a
    phantom play from an empty baseline.
    """
    from python_ai.envs.gym_wrapper import MicroRoyaleEnv
    env = MicroRoyaleEnv()
    env.reset()
    seen = []
    for _ in range(120):
        obs, _, term, trunc, info = env.step(
            {"card_index": 4, "target_x": 8, "target_y": 5})   # no-op
        seen.append(int(info["opp_played_card"]))
        if term or trunc:
            break
    assert seen[0] == -1, "first poll only seeds the baseline"
    played = [c for c in seen if c >= 0]
    assert played, "the opponent never played -- no supervision would exist"
    assert all(c in set(env.current_opp_deck) for c in played)
