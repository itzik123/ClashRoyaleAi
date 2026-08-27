"""The rollout buffer must not hold two copies of the observations through the
PPO update.

`stack()` returns NEW contiguous (T, N, ...) tensors -- `torch.stack` allocates
and copies, it does not view the inputs. So from the moment it returns there
are two full copies of every field alive: the per-step list, and the stacked
batch. The list is dead weight at that point; nothing reads it again until the
next rollout overwrites it.

`run()` cleared the buffer AFTER `run_update()` returned, so both copies stayed
resident for the whole update -- which is ~87% of the cycle's wall clock. For
the observation field alone, at the production shape:

    obs dim 13606 x 8 envs x 4 bytes         = 0.435 MB per step
    x 500 steps                              = 217.7 MB
    both copies                              = 435.4 MB held through the update

against a main process measured at 1194 MB private commit. Draining at the
stack point releases ~218 MB for the expensive phase, at the cost of nothing:
the batch already owns its storage.

The tests below pin BOTH halves -- that the stacked batch is genuinely
independent of the lists (or draining would corrupt the update), and that the
lists are actually released.
"""
import pytest
import torch

from python_ai.rl.buffer import RolloutBuffer

FIELDS = ("obs", "rewards")


def _filled(n=4, width=3):
    buf = RolloutBuffer(FIELDS)
    for t in range(n):
        buf.add(obs=torch.full((width, 5), float(t)),
                rewards=torch.full((width,), float(t)))
    return buf


def test_stack_does_not_alias_the_stored_tensors():
    """The premise draining depends on. If `stack` ever returned views,
    releasing the lists would pull storage out from under the update."""
    buf = _filled()
    batch = buf.stack()
    original = buf._data["obs"][0].clone()

    batch["obs"][0].add_(99.0)
    assert torch.equal(buf._data["obs"][0], original), (
        "mutating the stacked batch changed the stored tensor -- stack() is "
        "aliasing, and drain() would be unsafe")


def test_drain_returns_exactly_what_stack_would_have():
    a = _filled().stack()
    b = _filled().drain()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k])


def test_drain_empties_the_buffer():
    buf = _filled(n=4)
    assert len(buf) == 4
    buf.drain()
    assert len(buf) == 0
    for name in FIELDS:
        assert buf._data[name] == []


def test_the_drained_batch_survives_the_release():
    """The batch must remain fully usable after the lists are gone -- this is
    the whole point, and an aliasing regression would show up here as garbage
    rather than as an exception."""
    buf = _filled(n=4, width=3)
    batch = buf.drain()
    assert batch["obs"].shape == (4, 3, 5)
    for t in range(4):
        assert torch.equal(batch["obs"][t], torch.full((3, 5), float(t)))
        assert torch.equal(batch["rewards"][t], torch.full((3,), float(t)))


def test_drain_releases_the_underlying_storage():
    """Not just the list -- the tensors themselves must become unreferenced."""
    import weakref
    buf = _filled(n=3)
    ref = weakref.ref(buf._data["obs"][0])
    assert ref() is not None
    buf.drain()
    import gc
    gc.collect()
    assert ref() is None, "a stored tensor outlived drain()"


def test_drain_on_an_empty_buffer_refuses_like_stack():
    buf = RolloutBuffer(FIELDS)
    with pytest.raises(RuntimeError):
        buf.drain()


def test_a_second_drain_refuses_rather_than_returning_a_stale_batch():
    """Draining twice is a caller bug; it must not silently hand back an empty
    or half-built batch."""
    buf = _filled()
    buf.drain()
    with pytest.raises(RuntimeError):
        buf.drain()


@pytest.mark.slow
def test_run_update_leaves_the_buffer_empty(tmp_path, monkeypatch):
    """End to end: after an update the per-step lists are already released,
    without waiting for run()'s later clear()."""
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: str(tmp_path.joinpath(*name.split("/"))))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "hc"))

    cfg = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                    num_minibatches=1, ppo_epochs=1,
                    save_every_episodes=10 ** 9,
                    replay_every_episodes=10 ** 9)

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0})
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

    t = Harness()
    t.setup()
    try:
        t.collect_rollout()
        assert len(t.buffer) == cfg.update_timestep
        stats = t.run_update()
        assert len(t.buffer) == 0, (
            "the per-step lists were still resident after run_update -- both "
            "copies are alive through the most expensive phase of the cycle")
        assert stats.nonfinite_skips == 0
        # and run()'s later clear() must remain harmless
        t.buffer.clear()
        assert len(t.buffer) == 0
    finally:
        t.envs.close()
