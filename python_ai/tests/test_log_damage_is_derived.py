"""The Log's damage is measured, not restated (a second copy had gone stale).

`eval/measure_deck_matchups.py` held `LOG_DAMAGE = 240.0`, labelled
"CardRegistry.h -- The Log's damage", while the registry says 269. Every
Log-opportunity figure measured with it capped each body 11% low. It is now
`card_probes.roller_damage`, and the harness's fast separable Log map is pinned
against the literal double loop its own docstring calls the test oracle -- which
no test had actually run.
"""
import numpy as np

import python_ai  # noqa: F401
import clash_royale_env as E

from python_ai.advisors import card_probes

ID = {E.get_card_info(c)["name"]: c for c in E.get_all_card_ids()}


def test_the_log_deals_the_registrys_269():
    assert card_probes.roller_damage(ID["The Log"]) == 269.0


def test_only_rollers_have_a_roller_damage():
    assert card_probes.roller_damage(ID["Barbarian Barrel"]) > 0.0
    assert card_probes.roller_damage(ID["Fireball"]) == 0.0      # a disc spell
    assert card_probes.roller_damage(ID["Knight"]) == 0.0        # not a spell


def test_the_harness_uses_the_measured_value():
    from python_ai.eval import measure_deck_matchups as M
    assert M._log_damage() == card_probes.roller_damage(ID["The Log"])


def test_the_fast_log_map_matches_its_reference_loop():
    from python_ai.eval import measure_deck_matchups as M
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    env = E.ClashRoyaleEnv(deck, deck, 3600)
    env.seed(1)
    for cid, x, y in ((ID["Skeleton Army"], 6.0, 9.0), (ID["Knight"], 12.0, 11.0),
                      (ID["Goblin Gang"], 3.0, 4.0)):
        env.inject(cid, x, y, 1, -1.0, 300)
    env.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    fast, slow = M.log_catch_map(obs), M._log_catch_map_reference(obs)
    assert fast.max() > 0.0
    assert np.allclose(fast, slow, atol=1e-3)
