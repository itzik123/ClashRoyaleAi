"""`get_units()` reads the board the logger records, without the file.

The same entities, with the same id, position, hp, team, symbol, flying flag,
cardId and name as the last tick of `save_log`, mid-match with troops out.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402

DECK = list(gym_wrapper.DEFAULT_DECK)


def test_get_units_matches_the_logged_board(tmp_path):
    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.reset()
    env.inject(DECK[0], 3.0, 12.0, 0)
    env.inject(DECK[1], 14.0, 22.0, 1)
    for _ in range(80):
        env.step_self_play_fast(-1, 0, 0, -1, 0, 0, 1)

    path = tmp_path / "log.json"
    env.save_log(str(path))
    logged = json.loads(path.read_text())["ticks"][-1]["entities"]
    units = env.get_units()

    assert len(units) == len(logged) > 6   # the towers and some troops
    by_id = {e["id"]: e for e in logged}
    for u in units:
        e = by_id[u.id]
        assert (u.team, u.hp, u.card_id, u.name, u.symbol, u.flying) == \
            (e["team"], e["hp"], e["cardId"], e["name"], e["symbol"], e["isFlying"])
        assert abs(u.x - e["x"]) < 0.01 and abs(u.y - e["y"]) < 0.01


def test_get_units_changes_nothing():
    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.reset()
    before = env.get_observation_for_team(0)
    for _ in range(3):
        env.get_units()
    assert env.get_observation_for_team(0) == before
