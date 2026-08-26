"""A corrupt PFSP pool entry must not take the whole run down.

`_sample_pfsp_opponent` runs on EVERY reset and samples uniformly-at-random
from the pool, so a single unreadable checkpoint crashes phase 2 at an
unpredictable point -- typically hours in, and with a traceback that names
torch.load rather than the pool.

The pool has been written non-atomically for this project's entire history
(`atomic_save` only landed 2026-08-26), so a truncated entry left by an OOM
kill, a full disk or a SIGKILL can already be sitting in
`historical_checkpoints/` today. Fixing the writer does not clean up what the
old writer left.

BOTH failure directions matter, and they pull opposite ways:

  * crashing on one bad file wastes a run over an opponent that could simply
    have been skipped;
  * silently swallowing load errors is worse, because an architecture change
    makes EVERY checkpoint unloadable at once, the pool empties to just the
    scripted bots, and the league quietly stops being self-play at all.
    CLAUDE.md already records that exact shape for an empty pool: "silent, and
    it degrades the opponent distribution rather than crashing."

So: skip the individual entry, say so loudly, and never let the pool empty
without saying so.
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
    """Every entry failing is an ARCHITECTURE problem, not a bad file. Quietly
    continuing turns the league into four scripted bots and reports nothing."""
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
    """Startup, before any snapshot exists. Must not raise."""
    env = _env()
    env.refresh_pfsp_pool([])
    env._sample_pfsp_opponent()
