"""teacher.py -- the utility-search sparring partner, and the curriculum.

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


# ==========================================================================
# teacher.py -- the utility-search sparring partner (2026-08-19)
# ==========================================================================
# The elixir-multiplier curriculum priced the win condition negatively (see
# CLAUDE.md's 1.5x hypothesis and its monotone dose-response). It is replaced
# by a COMPETENCE curriculum: a deterministic utility-search bot at a symmetric
# 1.0x economy, whose difficulty is dialed on lookahead/width/epsilon.
#
# These tests pin the parts that fail SILENTLY. In particular a wrong team-1
# frame conversion makes every team-1 placement illegal, which reads as "the
# bot is weak" rather than "the bot is broken" -- exactly the class of bug this
# project has paid for three times.


def test_card_roles_derives_roles_from_the_engine_not_a_hardcoded_list():
    from python_ai.opponents.teacher import card_roles
    roles = card_roles(gym_wrapper.DEFAULT_DECK)
    assert roles[15] == "wincon", "Hog Rider is the deck's building-targeter"
    assert roles[7] == "spell"      # Fireball
    assert roles[33] == "spell"     # The Log
    assert roles[25] == "building"  # Cannon
    assert roles[6] == "ranged"     # Musketeer
    # Ice Golem also targets buildings but is a 2-cost shield, not a win
    # condition -- highest-cost building-targeter wins, the same tiebreak
    # gym_wrapper._find_win_condition uses.
    assert roles[40] != "wincon"
    assert set(roles) == set(gym_wrapper.DEFAULT_DECK)


def _teacher_env(ticks=40):
    env = CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    for _ in range(ticks):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    return env


def test_teacher_candidates_are_all_legal_for_either_team():
    """is_valid_placement takes ABSOLUTE y for both teams while step_self_play
    takes team 1's y MIRRORED. Get that backwards and team 1 silently never
    places anything."""
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
    """The no-op baseline is what makes every other score a MARGINAL value. If
    it drifts, the bot either dumps elixir on sight or freezes forever."""
    from python_ai.opponents.teacher import UtilityTeacher

    env = _teacher_env()
    t = UtilityTeacher(gym_wrapper.DEFAULT_DECK, team=0, horizon_ticks=30)
    t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(env, obs) if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base, obs) == 0.0


def test_rollout_does_not_touch_the_live_match():
    """snapshot() deep-copies the stats collectors. If that ever regresses,
    every hypothetical hit lands in the REAL match's statistics -- and those
    feed the reward shaping, so a search would corrupt its own returns."""
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
    """One class plays both sides. A broken team-1 frame shows up here as a
    bot that never lands a card, not as an exception."""
    from python_ai.opponents.teacher import UtilityTeacher

    # SEEDED. A bare reset() drew team 1's opening hand from
    # std::random_device, so "how many cards land in 120 decisions" varied per
    # invocation and this occasionally tripped its own >= 5 floor. That was
    # latent for as long as the scorer was lenient. `ClashRoyaleEnv.seed()`
    # seeds both engine generators and re-deals (2026-08-21).
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
    """The ORDER half of CycleTracker, tested deterministically with no engine.

    WHY NOT DRIVE THE ENGINE. Two engine-driven versions of this test passed
    VACUOUSLY and one stayed flaky. Playing hand slot 0 every step rotates only
    slot 0, so a Hog dealt into slots 1-3 never leaves the hand; playing the
    CHEAPEST affordable card fails the same way for the opposite reason, since
    2.6 has two 1-cost cards and a 4-cost Hog is never cheapest; and a uniform
    draw still misses when the match ends before the Hog is ever affordable.
    A vacuous pass on a cycle tracker is worse than no test at all -- 2.6 is
    DEFINED by cycling -- so the logic is pinned here against an exact model and
    the engine agreement is checked separately below.

    The model is the engine's own rule: the played card goes to the BACK of the
    queue, the front of the queue fills the vacated hand slot.
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

        # A card just played sits at the back of a 4-long queue, so it is
        # exactly 4 plays from returning. Membership alone is satisfied by ANY
        # ordering; this is the part that can actually be wrong.
        assert ct.distance_to(played) == 4, (
            f"{played} was just played; expected 4, got {ct.distance_to(played)}")
        for i, card in enumerate(queue):
            assert ct.distance_to(card) == i + 1, (
                f"queue {queue} but distance_to({card}) said "
                f"{ct.distance_to(card)}, expected {i + 1}")
        for card in hand:
            assert ct.distance_to(card) == 0


def test_cycle_tracker_agrees_with_the_engines_own_hand():
    """The identity invariant, against the real engine: distance is 0 exactly
    when the card is in hand. Runs several matches because one can end before
    the win condition is ever affordable, and asserts at the end that at least
    one rotation was actually observed -- otherwise this passes vacuously too.
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

    # ELEVEN rungs since 2026-09-03 (was 6). The count is asserted so the
    # table cannot shrink back silently, but every property below is what
    # actually matters and holds at any length.
    assert len(TEACHER_STAGES) == 11
    for s in TEACHER_STAGES:
        assert set(s) == {"horizon_ticks", "epsilon", "k_cells", "max_combos",
                          "reactive"}, (
            "a stage must never carry an elixir multiplier -- that is the whole "
            "point of this curriculum")
        # The claim above is about ECONOMY, so state it directly rather than
        # relying on the key set alone. `max_combos` was added on 2026-08-20 as
        # a third COMPETENCE axis (how many two-card sequences the bot may
        # simulate), and a future axis should have to pass this too.
        for key in s:
            assert "elixir" not in key and "multiplier" not in key, key
    eps = [s["epsilon"] for s in TEACHER_STAGES]
    hor = [s["horizon_ticks"] for s in TEACHER_STAGES]
    combos = [s["max_combos"] for s in TEACHER_STAGES]
    assert eps == sorted(eps, reverse=True), "noise must fall monotonically"
    assert hor == sorted(hor), "lookahead must rise monotonically"
    assert combos == sorted(combos), "combo width must rise monotonically"
    # A combo's follow-up lands 10 ticks in, so a rung whose rollout stops
    # before that would charge two cards and simulate one. The ladder must not
    # be able to ask for combos it cannot score.
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
    """The teacher is OPT-IN. Every existing harness, probe and eval anchor
    constructs MicroRoyaleEnv with no opponent key and must keep getting the C++
    HeuristicOpponent, or every historical number silently stops being
    comparable."""
    env = gym_wrapper.MicroRoyaleEnv()
    assert env.teacher is None
    assert env.opponent_kind == "builtin"


def test_teacher_env_produces_the_same_info_keys_as_the_builtin_env():
    """train.py reads ~20 keys out of info. stepSelfPlay returns a different
    result type than step (observation0/reward0, no observation/reward), so the
    routing is the one place those could diverge."""
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
    """A teacher that silently never plays looks exactly like a weak opponent.
    This is the end-to-end version of the frame check in
    test_teacher_candidates_are_all_legal_for_either_team."""
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
    heuristic env that must not raise."""
    env = gym_wrapper.MicroRoyaleEnv()
    env.set_teacher_stage(4)          # must not raise
    assert env.teacher is None


#: PIPELINE 1's training path -- `trainers/train.py` plus the loop and the
#: curriculum that moved out of it into `rl/`. Scanning train.py alone, which is
#: what this test used to do, would now pass while the banned call sat one
#: import away.
#:
#: Deliberately NOT pipeline 2: `envs/selfplay_env.BUILTIN_TRAINING_OPPONENTS`
#: really does put "builtin:heuristic@1.35" and "@1.50" in the PFSP pool, so a
#: multiplier there is the anchor's own identity, not a curriculum handicap.
#: The 2026-08-19 pivot was about phase 1's ladder and this test's claim is
#: about phase 1's ladder.
def _phase1_training_sources():
    import pathlib
    root = pathlib.Path(python_ai.PACKAGE_DIR)
    yield root / "trainers" / "train.py"
    yield from sorted((root / "rl").glob("*.py"))


def _curriculum_stages_literal():
    """The real object, imported.

    This used to PARSE `train.py` with `ast`, because CURRICULUM_STAGES was a
    local of `train_ppo()` and could not be imported at all. That worked, and it
    was a smell worth acting on: a constant nothing can import is a constant
    nothing can check, and a parser that finds nothing can only be told apart
    from a parser that finds the wrong thing by the assertion it raises. It now
    lives at module scope in `rl/curriculum.py`.
    """
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    return CURRICULUM_STAGES


def test_training_never_raises_the_opponent_elixir_multiplier():
    """THE PIN ON THE 2026-08-19 PIVOT.

    The 1.5x handicap is what priced the win condition negatively -- measured
    monotone across 1.0/1.25/1.5x, and the reason four separate Hog
    interventions all returned null. The API survives for the ~15 measurement
    harnesses that sweep it (including the falsifier that justified the change);
    phase 1's training path must never call it again.

    AST-based, not line-based. The line scan this replaces skipped anything
    starting with `#`, which silently exempted every mention inside a DOCSTRING
    -- and a docstring is exactly where a future author would explain the ban
    before quietly reintroducing it below.
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
                # env-config key are the two indirect routes to the same thing.
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "phase 1's training path sets an opponent elixir multiplier:\n  "
        + "\n  ".join(offenders))


def test_curriculum_lives_where_a_test_can_import_it():
    """The regression for the reason the two tests below used to parse source."""
    from python_ai.rl import curriculum
    assert curriculum.CURRICULUM_STAGES is _curriculum_stages_literal()


def test_curriculum_stages_are_competence_not_economy():
    from python_ai.rl.curriculum import STAGE_WIN_RATE_GATE

    stages = _curriculum_stages_literal()
    assert len(stages) == 11          # was 6 before 2026-09-03
    for s in stages:
        assert "opp_elixir_multiplier" not in s, (
            "a curriculum stage must never carry an elixir multiplier again")
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in stages] == list(range(11))
    # The gate moved 0.80 -> 0.65 WITH the rung split, and the two must not be
    # separated: 0.80 across eleven rungs is a strictly harder ladder than the
    # six-rung version it replaced. Read from the constant rather than
    # restated, so retuning it stays a one-line change.
    assert ([s["win_rate_threshold"] for s in stages]
            == [STAGE_WIN_RATE_GATE] * 10 + [None])


def test_curriculum_stage_count_matches_the_teacher_ladder():
    """If these ever drift, CURRICULUM_STAGES would index past the end of
    TEACHER_STAGES -- or, worse, silently stop at a rung short of the top."""
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
