"""teacher.py: the utility-search opponent, and the curriculum."""
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


# Difficulty is competence (lookahead, width, epsilon) at a symmetric 1.0x
# economy, never an elixir handicap. These pin the parts that fail silently: a
# wrong team-1 frame conversion makes every team-1 placement illegal, which
# reads as "the bot is weak".


def test_card_roles_derives_roles_from_the_engine_not_a_hardcoded_list():
    from python_ai.opponents.teacher import card_roles
    roles = card_roles(gym_wrapper.DEFAULT_DECK)
    assert roles[15] == "wincon", "Hog Rider is the deck's building-targeter"
    assert roles[7] == "spell"      # Fireball
    assert roles[33] == "spell"     # The Log
    assert roles[25] == "building"  # Cannon
    assert roles[6] == "ranged"     # Musketeer
    # Ice Golem also targets buildings but is a 2-cost tank, not the win
    # condition.
    assert roles[40] != "wincon"
    assert set(roles) == set(gym_wrapper.DEFAULT_DECK)


def _teacher_env(ticks=40):
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    for _ in range(ticks):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    return env


def test_teacher_candidates_are_all_legal_for_either_team():
    """is_valid_placement takes absolute y for both teams while step_self_play
    takes team 1's y mirrored; backwards, team 1 silently never places
    anything.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    for team in (0, 1):
        env = _teacher_env()
        t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=team, horizon_ticks=0)
        t.reset()
        obs = np.asarray(env.get_observation_for_team(team), np.float32)
        cands = t.candidates(env, obs)
        assert len(cands) >= 1, "the no-op candidate is always present"
        for c in cands:
            if c.slot == CE.HAND_SIZE:
                continue
            assert env.is_valid_placement(c.card_id, c.x, t.to_absolute_y(c.y), team), (
                f"team {team} proposed an illegal cell: {c}")


def test_teacher_never_proposes_an_unaffordable_card():
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env(ticks=0)
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=0)
    t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    cands = t.candidates(env, obs, elixir=1.0)
    for c in cands:
        if c.slot == CE.HAND_SIZE:
            continue
        assert E.get_card_info(c.card_id)["cost"] <= 1.0


def test_noop_scores_exactly_zero_so_the_teacher_can_hold_elixir():
    """The no-op baseline makes every other score a marginal value; if it drifts,
    the bot dumps elixir on sight or freezes.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=30)
    t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(env, obs) if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base, obs) == 0.0


def test_rollout_does_not_touch_the_live_match():
    """snapshot() deep-copies the stats collectors; otherwise hypothetical hits
    land in the real match's statistics, which feed the reward.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    before = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
              env.get_elixir_spent(0), env.get_elixir_for_team(0))
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=30)
    t.reset()
    t.act(env, np.asarray(env.get_observation_for_team(0), np.float32))
    after = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
             env.get_elixir_spent(0), env.get_elixir_for_team(0))
    assert before == after


def test_teacher_is_side_agnostic():
    """One class plays both sides; a broken team-1 frame shows up as a bot that
    never lands a card, not an exception.
    """
    from python_ai.opponents.teacher import UtilityTeacher

    # Seeded: an unseeded reset() draws team 1's hand from OS entropy, and the
    # landed-card count would vary per invocation.
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.seed(0)
    t1 = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=1, horizon_ticks=30, seed=0)
    t1.reset()
    played = 0
    for _ in range(120):
        obs1 = np.asarray(env.get_observation_for_team(1), np.float32)
        slot, x, y = t1.act(env, obs1)
        spent_before = env.get_elixir_spent(1)
        env.step_self_play(-1, 0, 0, slot, x, y, 10)
        if env.get_elixir_spent(1) > spent_before:
            played += 1
        if env.is_game_over():
            break
    print(f"\n  team-1 teacher landed {played} cards in 120 decisions")
    assert played >= 5, f"team-1 teacher only landed {played} cards"


def test_cycle_tracker_order_logic_against_an_exact_simulated_cycle():
    """The order half of CycleTracker, against an exact model of the engine's rule
    (the played card goes to the back of the queue, the front fills the vacated
    slot), with no engine.

    Engine-driven versions pass vacuously: playing slot 0 only rotates slot 0,
    playing the cheapest card never plays a 4-cost Hog, and a match can end
    before the Hog is affordable. Engine agreement is checked separately below.
    """
    from python_ai.opponents.teacher import CycleTracker

    deck = list(gym_wrapper.DEFAULT_DECK)
    hand, queue = deck[:4], deck[4:]
    ct = CycleTracker(deck)
    ct.reset()
    ct.observe(list(hand))

    rng = np.random.default_rng(0)
    for _ in range(200):
        slot = int(rng.integers(len(hand)))
        played = hand[slot]
        hand[slot] = queue.pop(0)
        queue.append(played)
        ct.observe(list(hand))

        # A card just played sits at the back of a 4-long queue, exactly 4
        # plays from returning. Membership alone is satisfied by any ordering.
        assert ct.distance_to(played) == 4, (
            f"{played} was just played; expected 4, got {ct.distance_to(played)}")
        for i, card in enumerate(queue):
            assert ct.distance_to(card) == i + 1, (
                f"queue {queue} but distance_to({card}) said "
                f"{ct.distance_to(card)}, expected {i + 1}")
        for card in hand:
            assert ct.distance_to(card) == 0


def test_cycle_tracker_agrees_with_the_engines_own_hand():
    """Against the real engine: distance is 0 exactly when the card is in hand.
    Several matches, asserting at the end that a rotation was actually
    observed.
    """
    from python_ai.opponents.teacher import CycleTracker

    costs = {c: E.get_card_info(c)["cost"] for c in gym_wrapper.DEFAULT_DECK}
    driver = np.random.default_rng(0)
    seen_far = False
    for _ in range(6):
        env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
        env.reset()
        ct = CycleTracker(gym_wrapper.DEFAULT_DECK)
        ct.reset()
        for _ in range(200):
            hand = list(env.get_hand_for_team(0))
            ct.observe(hand)
            d = ct.distance_to(15)
            if 15 in hand:
                assert d == 0, f"Hog is in hand {hand} but distance_to said {d}"
            else:
                assert d > 0, f"Hog absent from {hand} but distance_to said 0"
                seen_far = True
            elixir = env.get_elixir_for_team(0)
            playable = [i for i, c in enumerate(hand)
                        if costs[c] <= elixir + 1e-6]
            if not playable:
                env.step_self_play(-1, 0.0, 0.0, -1, 0, 0, 10)
                continue
            env.step_self_play(int(driver.choice(playable)), 9.0, 10.0,
                               -1, 0, 0, 10)
            if env.is_game_over():
                break
        if seen_far:
            break
    assert seen_far, "the win condition never left hand in 6 matches"


def test_teacher_stages_are_competence_not_economy():
    from python_ai.opponents.teacher import TEACHER_STAGES

    # The count is asserted so the table cannot shrink silently; the properties
    # below hold at any length.
    assert len(TEACHER_STAGES) == 11
    for s in TEACHER_STAGES:
        assert set(s) == {"horizon_ticks", "epsilon", "k_cells", "max_combos",
                          "reactive"}, (
            "a stage must never carry an elixir multiplier -- that is the whole "
            "point of this curriculum")
        # The claim is about economy, so check every key directly; a future
        # competence axis must pass this too.
        for key in s:
            assert "elixir" not in key and "multiplier" not in key, key
    eps = [s["epsilon"] for s in TEACHER_STAGES]
    hor = [s["horizon_ticks"] for s in TEACHER_STAGES]
    combos = [s["max_combos"] for s in TEACHER_STAGES]
    assert eps == sorted(eps, reverse=True), "noise must fall monotonically"
    assert hor == sorted(hor), "lookahead must rise monotonically"
    assert combos == sorted(combos), "combo width must rise monotonically"
    # A combo's follow-up lands 10 ticks in, so a rung whose rollout stops
    # sooner would charge two cards and simulate one.
    from python_ai.opponents.teacher import COMBO_MIN_HORIZON_TICKS
    for s in TEACHER_STAGES:
        if s["max_combos"] > 0:
            assert s["horizon_ticks"] >= COMBO_MIN_HORIZON_TICKS


def test_epsilon_one_still_only_emits_legal_actions():
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=0,
                       epsilon=1.0, seed=0)
    t.reset()
    for _ in range(50):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t.act(env, obs)
        if slot != CE.HAND_SIZE:
            cid = env.get_hand_for_team(0)[slot]
            assert env.is_valid_placement(cid, x, t.to_absolute_y(y), 0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
        if env.is_game_over():
            break


def test_default_env_is_unchanged_and_still_uses_the_cpp_heuristic():
    """The teacher is opt-in: harnesses constructing MicroRoyaleEnv with no
    opponent key keep the C++ HeuristicOpponent.
    """
    env = gym_wrapper.MicroRoyaleEnv()
    assert env.teacher is None
    assert env.opponent_kind == "builtin"


def test_teacher_env_produces_the_same_info_keys_as_the_builtin_env():
    """train.py reads many info keys, and stepSelfPlay returns a different result
    type than step, so the routing is where they could diverge.
    """
    a = gym_wrapper.MicroRoyaleEnv()
    a.reset()
    b = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 2})
    b.reset()
    assert b.teacher is not None
    act = {"card_index": np.array([CE.HAND_SIZE]),
           "target_x": np.array([9.0]), "target_y": np.array([10.0])}
    _, _, _, _, ia = a.step(act)
    _, _, _, _, ib = b.step(act)
    assert set(ia) == set(ib)


def test_teacher_opponent_actually_plays_cards():
    """A teacher that silently never plays looks like a weak opponent: the
    end-to-end frame check.
    """
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 3})
    env.reset()
    act = {"card_index": np.array([CE.HAND_SIZE]),
           "target_x": np.array([9.0]), "target_y": np.array([10.0])}
    info = None
    for _ in range(150):
        _, _, term, _, info = env.step(act)
        if term:
            break
    assert info is not None and info["team1_elixir_spent"] > 0.0, (
        "the teacher opponent never spent a single elixir")


def test_set_teacher_stage_is_a_noop_on_a_builtin_env():
    """train.py calls envs.call('set_teacher_stage', n) unconditionally; on a
    heuristic env it must not raise.
    """
    env = gym_wrapper.MicroRoyaleEnv()
    env.set_teacher_stage(4)          # must not raise
    assert env.teacher is None


#: Pipeline 1's training path: trainers/train.py plus the loop and curriculum
#: in rl/. Not pipeline 2, whose PFSP pool deliberately includes heuristic
#: anchors at 1.35x / 1.50x as their own identity.
def _phase1_training_sources():
    import pathlib
    root = pathlib.Path(python_ai.PACKAGE_DIR)
    yield root / "trainers" / "train.py"
    yield from sorted((root / "rl").glob("*.py"))


def _curriculum_stages_literal():
    """The real object, imported from rl/curriculum.py."""
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    return CURRICULUM_STAGES


def test_training_never_raises_the_opponent_elixir_multiplier():
    """Phase 1's training path never sets an opponent elixir multiplier.

    The handicap priced the win condition negatively. The API remains for the
    measurement harnesses that sweep it. AST-based rather than line-based, so a
    mention cannot hide in a docstring.
    """
    import ast
    offenders = []
    for path in _phase1_training_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "set_opponent_elixir_multiplier"):
                offenders.append(f"{path.name}:{node.lineno}: direct call")
            elif isinstance(node, ast.Constant) and node.value in (
                    "set_opponent_elixir_multiplier", "opp_elixir_multiplier"):
                # `envs.call("set_opponent_elixir_multiplier", m)` and the
                # env-config key are the two indirect routes.
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "phase 1's training path sets an opponent elixir multiplier:\n  "
        + "\n  ".join(offenders))


def test_curriculum_lives_where_a_test_can_import_it():
    """CURRICULUM_STAGES is importable at module scope."""
    from python_ai.rl import curriculum
    assert curriculum.CURRICULUM_STAGES is _curriculum_stages_literal()


def test_curriculum_stages_are_competence_not_economy():
    from python_ai.rl.curriculum import STAGE_WIN_RATE_GATE

    stages = _curriculum_stages_literal()
    assert len(stages) == 11          # eleven rungs
    for s in stages:
        assert "opp_elixir_multiplier" not in s, (
            "a curriculum stage must never carry an elixir multiplier again")
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in stages] == list(range(11))
    # The gate and the rung split go together (see test_rl_curriculum.py); read
    # from the constant, not restated.
    assert ([s["win_rate_threshold"] for s in stages]
            == [STAGE_WIN_RATE_GATE] * 10 + [None])


def test_curriculum_stage_count_matches_the_teacher_ladder():
    """Drift would index past the end of TEACHER_STAGES, or stop a rung short of
    the top.
    """
    from python_ai.opponents.teacher import TEACHER_STAGES
    stages = _curriculum_stages_literal()
    assert len(stages) == len(TEACHER_STAGES)
    for s in stages:
        assert 0 <= s["teacher_stage"] < len(TEACHER_STAGES)


def test_phase1_opponent_defaults_to_the_teacher():
    from python_ai.trainers import train
    assert train.PHASE1_OPPONENT == "teacher", (
        "the C++ HeuristicOpponent is an EVAL ANCHOR now, not a training "
        "opponent -- see the 2026-08-19 pivot")
