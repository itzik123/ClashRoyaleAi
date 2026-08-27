"""The auxiliary task predicts the opponent's NEXT CARD, not their elixir.

WHY THE TASK WAS SWAPPED (2026-08-28). The elixir head was measurably not a
task: `test_aux_task_is_not_a_memory_probe.py` shows opponent elixir is an
affine function of two scalars already in the observation, solved exactly
(MAE 0.0000) by four parameters and no recurrence, while the trained head sat
at 0.77. It therefore exerted no pressure on the recurrent state.

Next-card cannot be solved that way -- it needs the opponent's play history,
which is exactly the information item 24 put in the observation and
`eval/probe_card_counting.py` then measured the policy DISCARDING (trained hx
decoded next-card +0.013 over a random projection at ep 1522 and -0.025 at
ep 2054, i.e. below the floor). Nothing in the objective asked for the cycle;
this head is the ask.

WHAT IS PINNED HERE. The label transform is the subtle part and gets most of
the coverage: it is the one place where an off-by-one or a missing episode
boundary would silently teach the net to predict the NEXT MATCH's opening play
from this match's final state -- a corruption that would show up as a
plausible-looking loss curve and nothing else.
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


# ---------------------------------------------------------------- labels ----

def test_label_is_the_next_play_at_or_after_each_step():
    # T=4, N=1: nothing, HOG, nothing, LOG -- all one episode.
    played = torch.tensor([[-1], [15], [-1], [33]])
    masks = torch.ones(4, 1)
    labels, has = next_card_labels(played, masks, torch.ones(4, 1))
    assert labels.squeeze(-1).tolist() == [15, 15, 33, 33]
    assert has.squeeze(-1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_a_step_that_plays_is_labelled_with_its_own_card():
    """The play during step t is in t's FUTURE: the agent chose its action
    from the observation at the top of the step, before the opponent acted."""
    played = torch.tensor([[15], [33]])
    labels, has = next_card_labels(played, torch.ones(2, 1), torch.ones(2, 1))
    assert labels.squeeze(-1).tolist() == [15, 33]


def test_the_carry_does_not_leak_across_an_episode_boundary():
    """The regression this file exists for.

    masks[t] == 0 marks the step an episode ENDED on. Without clearing the
    backward carry there, step 2 below would be labelled 33 -- a card played
    in a DIFFERENT match -- and the net would be trained to predict the next
    episode's opening from this one's final state.
    """
    played = torch.tensor([[-1], [15], [-1], [-1], [33], [-1]])
    masks = torch.ones(6, 1)
    masks[3, 0] = 0.0                     # episode ends at t=3
    labels, has = next_card_labels(played, masks, torch.ones(6, 1))

    assert has.squeeze(-1).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert labels[0].item() == 15 and labels[1].item() == 15
    assert labels[4].item() == 33         # its own play, same episode
    # t=2 and t=3 have no play left in THEIR episode -> dropped, not 33.
    assert has[2].item() == 0.0 and has[3].item() == 0.0


def test_steps_with_no_future_play_are_dropped_not_given_a_class():
    """'They never played again' is an artifact of where the episode stopped.

    Given a class it would be the majority label and the head would learn to
    emit it.
    """
    played = torch.full((5, 2), -1)
    labels, has = next_card_labels(played, torch.ones(5, 2), torch.ones(5, 2))
    assert has.sum().item() == 0.0


def test_unlabelled_rows_still_carry_a_legal_index():
    """cross_entropy indexes with the label tensor even on masked rows, so -1
    would raise before the mask could discard it."""
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
    """The all-envs-auto-reset step drops the key entirely; the default has to
    be the value that contributes exactly zero, matching engine_stats' rule."""
    assert opponent_played_card({}, 4).tolist() == [-1, -1, -1, -1]


# ------------------------------------------------------------------ head ----

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


# ------------------------------------------------------------------- env ----

def test_the_env_reports_which_card_the_opponent_played():
    """End to end: the signal has to actually arrive, and it has to be -1 on
    the first step rather than a phantom play from the baseline being empty."""
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
