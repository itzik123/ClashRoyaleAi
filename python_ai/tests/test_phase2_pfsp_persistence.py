"""Phase 2's per-opponent PFSP estimates survive a resume.

Otherwise every resume resets each opponent to the 0.5 prior and PFSP samples a
mastered snapshot as often as one the agent still loses to, until every worker
re-meets every opponent. Phase 1 does the same for its deck pool.
"""
import pytest

from python_ai.envs import selfplay_env
from python_ai.trainers.train_selfplay import Phase2Trainer


@pytest.fixture(scope="module")
def env():
    e = selfplay_env.MicroRoyaleSelfPlayEnv({"scenarios_enabled": False,
                                             "scenario_seed": 3})
    yield e
    e.close()


def test_a_seeded_estimate_survives_the_pool_refresh(env):
    """`refresh_pfsp_pool` fills missing entries with the prior; it must not
    overwrite what a resume restored.
    """
    env.set_pfsp_stats({"snap_a.pth": 0.83}, {"snap_a.pth": 40})
    env.refresh_pfsp_pool(["snap_a.pth", "snap_b.pth"])
    stats, counts = env.get_pfsp_stats()
    assert stats["snap_a.pth"] == pytest.approx(0.83)
    assert counts["snap_a.pth"] == 40
    assert stats["snap_b.pth"] == 0.5 and "snap_b.pth" not in counts


def test_the_merge_is_count_weighted_and_ignores_the_prior():
    per_worker = [
        ({"a": 0.9, "b": 0.5}, {"a": 30}),          # never met b: 0.5 is a prior
        ({"a": 0.6, "b": 0.2}, {"a": 10, "b": 20}),
        ({"c": 0.5}, {}),                           # nobody has played c
    ]
    stats, counts = Phase2Trainer.merge_pfsp(per_worker)
    assert stats["a"] == pytest.approx((0.9 * 30 + 0.6 * 10) / 40)
    assert stats["b"] == pytest.approx(0.2)         # not (0.5 + 0.2) / 2
    assert "c" not in stats
    assert counts == {"a": 40, "b": 20}


class _FakeEnvs:
    def __init__(self, per_worker):
        self.per_worker = per_worker
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "get_pfsp_stats":
            return self.per_worker
        return [None] * len(self.per_worker)


def _bare_trainer(envs, num_envs=4):
    t = Phase2Trainer.__new__(Phase2Trainer)    # no nets, no workers
    t.envs = envs

    class _Cfg:
        pass
    t.cfg = _Cfg()
    t.cfg.num_envs = num_envs
    for attr in ("entropy_reboost_episode", "last_stall_reboost_episode",
                 "last_eval_ep", "last_exploiter_burst_ep",
                 "exploiter_burst_index"):
        setattr(t, attr, 0)
    t.reference_roster = []
    return t


def test_the_checkpoint_carries_the_pooled_estimates():
    t = _bare_trainer(_FakeEnvs([({"a": 0.8}, {"a": 12}), ({"a": 0.4}, {"a": 4})]))
    payload = t.checkpoint_payload()
    assert payload["pfsp_stats"]["a"] == pytest.approx((0.8 * 12 + 0.4 * 4) / 16)
    assert payload["pfsp_counts"] == {"a": 16}


def test_a_resume_seeds_every_worker_with_an_equal_share():
    envs = _FakeEnvs([])
    t = _bare_trainer(envs, num_envs=4)
    t._seed_pfsp({"pfsp_stats": {"a": 0.7}, "pfsp_counts": {"a": 16}})
    assert envs.calls == [("set_pfsp_stats", ({"a": 0.7}, {"a": 4}))]


def test_an_older_checkpoint_seeds_nothing():
    """Control: no stored estimates means the prior, as before."""
    envs = _FakeEnvs([])
    _bare_trainer(envs)._seed_pfsp({"episodes_completed": 5})
    assert envs.calls == []


def test_a_worker_that_cannot_answer_does_not_break_the_checkpoint():
    class _Broken:
        def call(self, *a):
            raise RuntimeError("worker died")
    payload = _bare_trainer(_Broken()).checkpoint_payload()
    assert "pfsp_stats" not in payload


@pytest.mark.slow
def test_a_finished_match_is_counted_against_its_opponent():
    """The live path: a finished match moves the EMA and raises the count, which
    weights the merge.
    """
    e = selfplay_env.MicroRoyaleSelfPlayEnv({"scenarios_enabled": False,
                                             "scenario_seed": 3})
    try:
        e.refresh_pfsp_pool(["scripted:Rusher"])
        e.reset(seed=4)
        e._set_opponent("scripted:Rusher")
        done = False
        for _ in range(500):
            _, _, term, trunc, _ = e.step(
                {"card_index": 4, "target_x": 0.0, "target_y": 0.0})
            if term or trunc:
                done = bool(term)
                break
        assert done, "a passive agent should lose a full match to the Rusher"
        stats, counts = e.get_pfsp_stats()
        assert counts == {"scripted:Rusher": 1}
        assert stats["scripted:Rusher"] != 0.5
    finally:
        e.close()
