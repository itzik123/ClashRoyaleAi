"""Entropy normalisation by reachable arms, and the perfect-defense bonus."""
import os
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
import clash_royale_env as E  # noqa: E402
from python_ai import engine_constants as EC  # noqa: E402
from python_ai.advisors import advisor_target as AT  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.envs import scenario_offense  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.rewards import shaping as T  # noqa: E402
from python_ai.rewards import shaping as train_shaping  # noqa: E402
from python_ai.rewards import weights as TW  # noqa: E402
from python_ai.rewards import weights as train_weights  # noqa: E402
from python_ai.rewards.elixir_shaping import (  # noqa: E402
    SOLVENCY_RESERVE, W_SOLVENCY, bankruptcy_rate, solvency_potential,
    solvency_shaping,
)
from python_ai.rl.coverage import (  # noqa: E402
    PLACEMENT_COVERAGE_COEF, placement_coverage_slots,
)
from python_ai.trainers.distill_tactics import masked_kl  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv

from python_ai.tests.helpers import (  # noqa: E402
    B, L, chunk_fixture, shaping_stats,
)


def test_card_entropy_must_be_normalized_by_the_REACHABLE_maximum():
    """Card entropy is normalised by the reachable maximum, log(#affordable + 1),
    not log(hand_size + 1).

    With the fixed divisor a two-arm step (common) could carry only a fraction
    of the target, so the controller held play/wait near a coin flip and the
    agent spent elixir on sight: imposed bankruptcy that reads as defensive
    apathy. A distribution uniform over its legal arms must read exactly 1.0
    whatever the arm count.
    """
    logits = torch.zeros(4, 5)
    for row, n_legal in enumerate((2, 3, 4, 5)):
        logits[row, n_legal:] = float("-inf")
    mask = torch.isfinite(logits)
    ent = Categorical(logits=logits).entropy()

    n_legal = mask.sum(-1).clamp(min=2).float()
    reachable = ent / torch.log(n_legal)
    old_way = ent / float(np.log(5))

    print("\n  uniform-over-legal rows: reachable-normalized "
          f"{[round(v, 4) for v in reachable.tolist()]}  vs old/log(5) "
          f"{[round(v, 4) for v in old_way.tolist()]}")

    assert torch.allclose(reachable, torch.ones(4), atol=1e-6), (
        f"uniform over legal arms must read 1.0, got {reachable.tolist()}")
    assert old_way[0] < 0.45, (
        "a uniform 2-arm row used to read ~0.43 of 'maximum' -- if this no "
        "longer holds the normalizer changed and this test needs rereading")
    assert bool((old_way[:3] < 1.0).all()), "only the unmasked row can reach 1.0 the old way"


def test_flawless_defense_pays_only_on_a_win_and_scales_with_tower_hp():
    """The perfect-defense bonus cannot reorder win/loss/draw: it is gated on the
    win, so a turtle stalling into a timeout collects nothing (policy-invariant
    tower shaping measurably left pure defence as the optimum).
    """
    from python_ai.rewards import shaping as T, weights as TW

    dones = np.array([True, True, True, False])
    #                 clean win, scraped win, loss, mid-episode
    step_rewards = np.array([1.0, 1.0, -1.0, 0.0], dtype=np.float32)
    stats = {"team1_tower_damage": np.array(
        [0.0, EC.OWN_TOWER_HP_TOTAL * 0.5, 0.0, 0.0], dtype=np.float32),
        # A crown taken in every row, so this isolates the HP scaling; the
        # crown gate is tested below.
        "team1_towers_alive": np.array([2, 2, 2, 2], dtype=np.int64)}

    b = T.flawless_defense_bonus(dones, step_rewards, stats, None)

    print(f"\n  flawless bonus: clean={b[0]:.3f} scraped={b[1]:.3f} "
          f"loss={b[2]:.3f} mid={b[3]:.3f}  (W={TW.W_FLAWLESS_DEFENSE})")

    assert b[0] == pytest.approx(TW.W_FLAWLESS_DEFENSE), "a flawless win pays in full"
    assert b[1] == pytest.approx(TW.W_FLAWLESS_DEFENSE * 0.5), "half the HP, half the bonus"
    assert b[2] == 0.0, "a LOSS must never collect -- it could reorder outcomes"
    assert b[3] == 0.0, "mid-episode steps must never collect"
    draw = T.flawless_defense_bonus(
        np.array([True]), np.array([0.0], dtype=np.float32),
        {"team1_tower_damage": np.array([0.0], dtype=np.float32),
         "team1_towers_alive": np.array([2], dtype=np.int64)}, None)
    assert draw[0] == 0.0, "a DRAW must collect nothing, however clean"


def test_flawless_defense_reads_a_counter_that_already_auto_reset():
    """On a done step the counter may already read 0 for the next episode; the
    running max must recover the finished episode's damage, or a hard-fought
    win would be paid as flawless.
    """
    from python_ai.rewards import shaping as T, weights as TW

    took_half = EC.OWN_TOWER_HP_TOTAL * 0.5
    crown = {"team1_towers_alive": np.array([2], dtype=np.int64)}
    prev = {"team1_tower_damage": np.array([took_half], dtype=np.float32), **crown}
    post_reset = {"team1_tower_damage": np.array([0.0], dtype=np.float32), **crown}

    b = T.flawless_defense_bonus(np.array([True]), np.array([1.0], dtype=np.float32),
                                 post_reset, prev)
    assert b[0] == pytest.approx(TW.W_FLAWLESS_DEFENSE * 0.5), (
        "the post-autoreset zero was taken at face value; a scraped win would "
        f"be paid as flawless (got {b[0]}, expected {TW.W_FLAWLESS_DEFENSE * 0.5})")


def test_default_deck_is_the_26_hog_cycle_and_every_card_is_cheap_enough():
    """Nothing in the 2.6 deck costs more than 4, so a starved win condition (a
    5-cost card legal only after many non-spending steps) is structurally
    unavailable.
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    info = [clash_royale_env.get_card_info(c) for c in deck]
    names = [i["name"] for i in info]
    costs = [i["cost"] for i in info]

    print(f"\n  deck: {list(zip(names, costs))}  avg={sum(costs)/len(costs):.3f}")

    assert set(names) == {"Hog Rider", "Musketeer", "Cannon", "Ice Golem",
                          "Skeletons", "Ice Spirit", "The Log", "Fireball"}
    assert max(costs) <= 4.0, f"nothing may cost more than 4 in this deck, got {costs}"
    assert sum(costs) / len(costs) == pytest.approx(2.625), "2.6 Hog Cycle average"
    assert clash_royale_env.validate_deck_slots(deck) == "", "deck must be legal"


def test_win_condition_is_derived_from_the_engine_not_hardcoded():
    """The win condition follows the deck, not a literal id."""
    assert gym_wrapper.WIN_CONDITION_ID == 15, "2.6 Hog Cycle's win condition is the Hog"
    assert gym_wrapper._find_win_condition([10, 1, 41, 25, 7, 2, 6, 5]) == 2, "Giant deck -> Giant"
    assert gym_wrapper._find_win_condition([1, 6, 12, 24, 72, 33, 7, 29]) is None


def test_win_condition_damage_term_responds_to_its_weight():
    """The win-condition weight actually reaches the reward (a weight never passed
    would sit dead).
    """
    from python_ai.rewards import shaping as T, weights as TW

    # Reuse the shared fixture: a parallel copy of the key list can silently
    # exercise a different code path from the trainer.
    cur, prev = shaping_stats(0.0)
    prev["team0_wincon_damage"] = np.zeros(1, dtype=np.float32)
    cur["team0_wincon_damage"] = np.array([400.0], dtype=np.float32)
    on = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=1.0)[0])
    off = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=0.0)[0])
    print(f"\n  wincon term: w=1.0 -> {on:.5f}   w=0.0 -> {off:.5f}")
    assert on > off, "the win-condition weight does not reach the reward"
    assert on - off == pytest.approx(400.0 / EC.MAX_BUILDING_HP, rel=1e-4)


def test_flawless_bonus_refuses_a_win_with_every_enemy_tower_standing():
    """The turtle gate: a win with every enemy tower standing is a timeout win on
    tower HP, pure defence, and pays nothing.
    """
    from python_ai.rewards import shaping as T, weights as TW

    assert TW.FLAWLESS_REQUIRES_CROWN, "this test describes the crown-gated behaviour"
    clean = {"team1_tower_damage": np.zeros(1, dtype=np.float32),
             "team1_towers_alive": np.full(1, 3, dtype=np.int64)}
    crowned = {"team1_tower_damage": np.zeros(1, dtype=np.float32),
               "team1_towers_alive": np.full(1, 2, dtype=np.int64)}
    dones, rew = np.array([True]), np.array([1.0], dtype=np.float32)

    turtle = T.flawless_defense_bonus(dones, rew, clean, None)[0]
    decisive = T.flawless_defense_bonus(dones, rew, crowned, None)[0]
    print(f"\n  flawless bonus: turtle-win={turtle:.3f}  crowned-win={decisive:.3f}")
    assert turtle == 0.0, "a win with all 3 enemy towers up must pay NOTHING"
    assert decisive == pytest.approx(TW.W_FLAWLESS_DEFENSE), (
        "a flawless win that took a crown must still pay in full")
