"""The teacher's rollout capture: action-identity, and the recorded schema.

The first test is the load-bearing one. `UtilityTeacher` is phase 1's opponent,
so a debug hook that perturbed its play would silently invalidate every win rate
measured against it. This is the same guarantee `test_teacher_combos.py` pins
for `max_combos = 0`.
"""
import json
import os

import numpy as np
import pytest

import clash_royale_env
from python_ai.envs.gym_wrapper import DEFAULT_DECK
from python_ai.opponents.teacher import UtilityTeacher
from python_ai.rl.teacher_debug import (
    CapturingTeacher,
    attach_debug_capture,
    attach_teacher_debug,
    capturing_like,
)

CE = clash_royale_env.ClashRoyaleEnv
DECK = list(DEFAULT_DECK)


def _play(make_team1, seed, stage, n=60):
    """One match; returns team 1's action stream and the teacher that played it.

    BOTH teachers are seeded, and that is not optional plumbing.
    `UtilityTeacher.__init__` defaults `seed=None` -> `default_rng(None)`, i.e.
    OS entropy, and `reset()` then REDRAWS `profile` and `lane_bias` from it on
    purpose ("a FULLY deterministic opponent is memorizable"). Two unseeded arms
    therefore run different scoring weights and diverge for reasons that have
    nothing to do with what is being tested.
    """
    env = CE(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    t0 = UtilityTeacher(DECK, team=0, seed=seed)
    t0.set_stage(stage)
    t0.reset()
    t1 = make_team1()
    t1.set_stage(stage)
    t1.reset()
    acts = []
    for _ in range(n):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        acts.append(tuple(round(float(v), 6) for v in a1))
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    return acts, t1


# Stage 5 is the shipped rung. Stage 3 is included because its epsilon is 0.10,
# so it exercises the exploration branch the capture code must leave alone.
@pytest.mark.parametrize("stage", [5, 3])
def test_capture_is_action_identical_to_the_plain_teacher(stage):
    seed = 23
    plain, _ = _play(lambda: UtilityTeacher(DECK, team=1, seed=seed), seed, stage)
    capt, _ = _play(lambda: CapturingTeacher(DECK, team=1, seed=seed, top_k=4),
                    seed, stage)
    assert len(plain) == len(capt), "capture changed how long the match ran"
    assert plain == capt, (
        "CapturingTeacher chose a different action from UtilityTeacher -- a "
        "debug hook must never change phase 1's opponent"
    )


def test_the_harness_itself_is_deterministic():
    """Control. Without this, a passing identity test proves nothing: two plain
    arms that disagreed would mean the comparison, not the capture, is broken.
    """
    a, _ = _play(lambda: UtilityTeacher(DECK, team=1, seed=77), 77, 5)
    b, _ = _play(lambda: UtilityTeacher(DECK, team=1, seed=77), 77, 5)
    assert a == b


def test_capture_is_off_unless_a_capturing_teacher_is_built():
    """The plain teacher must carry no capture state at all, so the training
    loop cannot accidentally pay for it."""
    t = UtilityTeacher(DECK, team=1, seed=1)
    assert not hasattr(t, "debug_records")
    assert not hasattr(t, "top_k")


def test_records_have_the_schema_the_viewer_reads():
    _, t = _play(lambda: CapturingTeacher(DECK, team=1, seed=5, top_k=4), 5, 5)
    recs = t.debug_records
    assert recs, "no decisions recorded"

    for r in recs:
        assert r["kind"] in ("rollout", "rules_only", "epsilon", "no_candidates")
        assert isinstance(r["candidates"], list)
        assert isinstance(r["held"], bool)
        # top_k bounds the EXPENSIVE half: only that many candidates may carry a
        # predicted board, however many were scored.
        assert sum(1 for c in r["candidates"] if "final" in c) <= 4
        for c in r["candidates"]:
            assert set(c) >= {"steps", "kind", "score", "margin", "rejected"}
            assert isinstance(c["score"], float)
            assert c["rejected"] == (c["score"] <= c["margin"])
            for st in c["steps"]:
                assert set(st) == {"cardId", "slot", "x", "y", "delayTicks"}

    rollouts = [r for r in recs if r["kind"] == "rollout"]
    assert rollouts, "no rollout decisions in the sample"
    # A rollout decision always has the no-op comparator its scores are defined
    # against.
    assert all(r["baseline"] is not None for r in rollouts)
    # Ranked descending, so index 0 is the best-scoring candidate.
    for r in rollouts:
        scores = [c["score"] for c in r["candidates"]]
        assert scores == sorted(scores, reverse=True)


def test_chosen_index_points_at_the_action_that_was_played():
    _, t = _play(lambda: CapturingTeacher(DECK, team=1, seed=9, top_k=4), 9, 5)
    for r in t.debug_records:
        if r["kind"] != "rollout":
            continue
        if r["chosenIndex"] >= 0:
            # Something was played, so it must have beaten its own margin.
            c = r["candidates"][r["chosenIndex"]]
            assert not c["rejected"]
            assert not r["held"]
        elif r["candidates"]:
            # Nothing chosen: either it held, or every candidate was rejected.
            assert r["held"] or all(c["rejected"] for c in r["candidates"])


def test_attach_writes_one_indexed_block_not_a_copy_per_tick(tmp_path):
    """The size property that matters: a record carries several predicted
    boards, so stamping it onto every tick of its window multiplies the file."""
    replay = tmp_path / "r.json"
    replay.write_text(json.dumps({
        "boardWidth": 18, "boardHeight": 34, "totalTicks": 30,
        "ticks": [{"tick": i, "entities": []} for i in range(30)],
    }))
    records = [{"team": 1, "kind": "rollout", "candidates": [], "held": False,
                "chosenIndex": -1, "baseline": None, "horizonTicks": 100}
               for _ in range(3)]
    attach_teacher_debug(str(replay), records, skip_frames=10)

    data = json.loads(replay.read_text())
    assert data["teacherDebug"]["skipFrames"] == 10
    assert data["teacherDebug"]["team"] == 1
    assert len(data["teacherDebug"]["decisions"]) == 3
    # No per-tick copies.
    assert all("teacherDebug" not in t for t in data["ticks"])
    # Everything else in the replay is untouched.
    assert data["totalTicks"] == 30 and len(data["ticks"]) == 30


def test_attach_tolerates_no_records(tmp_path):
    replay = tmp_path / "r.json"
    replay.write_text(json.dumps({"ticks": [{"tick": 0, "entities": []}]}))
    attach_teacher_debug(str(replay), [], skip_frames=10)
    assert json.loads(replay.read_text())["teacherDebug"]["decisions"] == []


def test_capturing_like_preserves_the_live_opponent_not_a_fresh_one():
    """`reset()` redraws profile and lane bias from the RNG, so rebuilding a
    teacher from its constructor arguments alone yields a DIFFERENT opponent."""
    t = UtilityTeacher(DECK, team=1, seed=31)
    t.set_stage(5)
    t.reset()
    c = capturing_like(t, top_k=2)
    assert c.profile == t.profile
    assert c.lane_bias == t.lane_bias
    assert c.horizon_ticks == t.horizon_ticks
    assert c.epsilon == t.epsilon
    assert c.max_combos == t.max_combos
    assert c.deck == t.deck
    assert c.team == t.team
    assert c.top_k == 2


def test_attach_debug_capture_returns_none_without_a_teacher():
    """An env whose opponent is the C++ heuristic has no candidate rollouts, and
    asking for capture there must be a no-op rather than an error."""

    class NoTeacherEnv:
        teacher = None

    assert attach_debug_capture(NoTeacherEnv()) is None
