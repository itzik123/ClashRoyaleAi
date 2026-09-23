"""A corrupt PFSP pool entry must not take the whole run down.

`_sample_pfsp_opponent` runs on every reset, so one unreadable checkpoint would
crash phase 2 hours in. But silently swallowing load errors is worse: after an
architecture change every checkpoint fails at once and the league quietly
becomes the scripted bots. So: skip the entry, say so loudly, and never let the
pool empty silently.
"""
import numpy as np
import pytest
import torch

from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv


def _env():
    return MicroRoyaleSelfPlayEnv({"scenarios_enabled": False,
                                   "scenario_seed": 3})


def _good_checkpoint(path, env):
    torch.save({"model": env.opponent_net.state_dict()}, path)
    return str(path)


def _corrupt_checkpoint(path):
    path.write_bytes(b"\x80\x02}q\x00")     # a truncated pickle
    return str(path)


def test_a_corrupt_pool_entry_is_skipped_not_fatal(tmp_path, capsys):
    env = _env()
    good = _good_checkpoint(tmp_path / "good.pth", env)
    bad = _corrupt_checkpoint(tmp_path / "bad.pth")
    env.refresh_pfsp_pool([bad, good])

    for _ in range(12):          # sampling is random; force the bad one up
        env._sample_pfsp_opponent()

    assert env.opponent_checkpoint_path == good
    assert bad not in env.pfsp_pool, "the unreadable entry stayed in the pool"


def test_skipping_a_corrupt_entry_is_reported(tmp_path, capsys):
    env = _env()
    good = _good_checkpoint(tmp_path / "good.pth", env)
    bad = _corrupt_checkpoint(tmp_path / "bad.pth")
    env.refresh_pfsp_pool([bad, good])
    for _ in range(12):
        env._sample_pfsp_opponent()

    out = capsys.readouterr().out
    assert "bad.pth" in out and "pool" in out.lower(), out


def test_an_entirely_unreadable_pool_raises_rather_than_going_quiet(tmp_path):
    """Every entry failing is an architecture problem, not a bad file; quietly
    continuing turns the league into four scripted bots.
    """
    env = _env()
    pool = [_corrupt_checkpoint(tmp_path / f"bad{i}.pth") for i in range(3)]
    env.refresh_pfsp_pool(pool)
    with pytest.raises(RuntimeError, match="pool"):
        env._sample_pfsp_opponent()


def test_a_healthy_pool_is_untouched(tmp_path):
    env = _env()
    pool = [_good_checkpoint(tmp_path / f"ok{i}.pth", env) for i in range(3)]
    env.refresh_pfsp_pool(pool)
    for _ in range(10):
        env._sample_pfsp_opponent()
    assert sorted(env.pfsp_pool) == sorted(pool)


def test_an_empty_pool_is_still_a_no_op(tmp_path):
    """Startup, before any snapshot exists: must not raise."""
    env = _env()
    env.refresh_pfsp_pool([])
    env._sample_pfsp_opponent()
