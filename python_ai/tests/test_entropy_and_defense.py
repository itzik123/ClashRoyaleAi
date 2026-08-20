"""The 2026-08-16 audit: entropy normalization and perfect defense.

Split out of the old single `test_python_ai.py` on 2026-08-20. The bodies are
unchanged -- only the shared header moved into `tests/conftest.py`, so the set of
test node ids is the same modulo the file name.

    python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
"""
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


# ===========================================================================
# 2026-08-16 audit: the two defects the from-scratch 2.6 Hog run was gated on
# ===========================================================================

def test_card_entropy_must_be_normalized_by_the_REACHABLE_maximum():
    """The regression for the bankruptcy bug. Read the numbers.

    Both trainers used to divide the card head's entropy by
    LOG_N_CARD = log(hand_size + 1) = log(5). But the affordability mask leaves
    only (#affordable + 1) legal arms, and MEASURED on model_weights_cured.pth
    over 706 decision steps, 54.1% of them leave exactly TWO. On such a step
    the reachable maximum is log(2) = 0.693 nats, so a 0.35 target expressed
    against log(5) is 0.5633 nats -- 81.3% of what the step can carry.

    The controller therefore held the play/wait choice near a coin flip, the
    agent spent elixir on sight, mean elixir sat at 2.25/10, nothing was
    affordable on 73.9% of steps (78.5% under a big push), and P(play) was FLAT
    against threat (0.1008 none vs 0.1016 largest) -- which reads as defensive
    apathy and is really an imposed bankruptcy.

    This test pins the invariant the fix restores: a distribution that is
    UNIFORM over its legal arms must normalize to exactly 1.0, whatever the
    number of arms. Under the old divisor a uniform 2-arm distribution scored
    ~0.43 and the controller read that as "not random enough".
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
    """The 'perfect defense' term must not be able to reorder win/loss/draw.

    That is the whole safety argument for it being non-potential-based: this
    project already measured that policy-invariant tower shaping left PURE
    DEFENCE as the optimum and win-condition usage decayed to 0.7%. Gating on
    the win is what stops this term walking back into that -- a turtle that
    stalls into a timeout must collect exactly nothing.
    """
    from python_ai.rewards import shaping as T, weights as TW

    dones = np.array([True, True, True, False])
    #                 clean win, scraped win, loss, mid-episode
    step_rewards = np.array([1.0, 1.0, -1.0, 0.0], dtype=np.float32)
    stats = {"team1_tower_damage": np.array(
        [0.0, EC.OWN_TOWER_HP_TOTAL * 0.5, 0.0, 0.0], dtype=np.float32),
        # A crown taken in every row, so this test isolates the HP SCALING.
        # The crown gate itself is covered by
        # test_flawless_bonus_refuses_a_win_with_every_enemy_tower_standing.
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
    """On a done step the vector env has auto-reset, so the cumulative counter
    may already read 0 for the NEXT episode. The running max must recover the
    finished episode's real damage, or a hard-fought win would be paid as if it
    were flawless -- the bonus would then reward exactly the wrong games."""
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
    """The deck switch, pinned by the property that motivated it.

    The Giant deck's measured failure was a STARVED win condition: at ~0.35
    elixir per decision a 5-cost card is legal only after ~14 consecutive
    non-spending steps, and Giant was never played once across four full runs.
    Nothing in 2.6 costs more than 4, which makes that failure mode
    structurally unavailable rather than merely unlikely.
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
    """The win condition must follow the deck, not a literal.

    A hardcoded id would mean "Hog Rider" forever and silently credit the wrong
    card the next time the deck moves -- the same drift that had a test
    injecting a Musketeer under a comment saying Archers.
    """
    assert gym_wrapper.WIN_CONDITION_ID == 15, "2.6 Hog Cycle's win condition is the Hog"
    assert gym_wrapper._find_win_condition([10, 1, 41, 25, 7, 2, 6, 5]) == 2, "Giant deck -> Giant"
    assert gym_wrapper._find_win_condition([1, 6, 12, 24, 72, 33, 7, 29]) is None


def test_win_condition_damage_term_responds_to_its_weight():
    """The regression `spell_value_weight` did not have.

    W_SPELL_VALUE's anneal sat DEAD for an entire training era because no test
    ever varied the argument -- the weight was simply never passed. This asserts
    the win-condition term actually reaches the reward.
    """
    from python_ai.rewards import shaping as T, weights as TW

    # Reuse the module's own fixture rather than a parallel copy of the key
    # list -- a second copy is how a shaping test ends up silently exercising
    # a different code path from the trainer.
    cur, prev = shaping_stats(0.0)
    prev["team0_wincon_damage"] = np.zeros(1, dtype=np.float32)
    cur["team0_wincon_damage"] = np.array([400.0], dtype=np.float32)
    on = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=1.0)[0])
    off = float(T.compute_shaping(cur, prev, gamma=0.99, w_wincon=0.0)[0])
    print(f"\n  wincon term: w=1.0 -> {on:.5f}   w=0.0 -> {off:.5f}")
    assert on > off, "the win-condition weight does not reach the reward"
    assert on - off == pytest.approx(400.0 / EC.MAX_BUILDING_HP, rel=1e-4)


def test_flawless_bonus_refuses_a_win_with_every_enemy_tower_standing():
    """The turtle gate. A win where no enemy tower fell is a timeout win on
    tower HP -- pure defence, which is the local optimum this term must not
    pay for. Measured at ep 6,053: Hog usage 0.8% while the agent won ~100% of
    games at 1.0x by defending."""
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
