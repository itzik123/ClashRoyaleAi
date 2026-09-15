"""The first REAL step of an episode must not be shaped against a fabricated state.

Under gymnasium's NEXT_STEP autoreset the step after `done` is a PHANTOM: the
worker resets and returns `{}` for info, so `extract_engine_stats` fills every
key with its default. That step's own shaping is correctly zeroed by
`* (1 - prev_dones)`. The bug was one step later: `base_trainer` stored the
phantom's fabricated stats as `_prev_stats`, and the first real step was then
shaped AGAINST them.

For every counter that is harmless -- `auto_reset_mask` and the delta clamps
take care of it. It is not harmless for a POTENTIAL whose value at the default is
non-zero. Measured 2026-09-15 (audit 03, R1), `probe_reset_hygiene.py`:

    t=141 PHANTOM            elixir=0.000 (default)   shaping masked -> 0
    t=142 FIRST REAL STEP    elixir=3.350
          Phi_solv(prev) = -0.10000   <- the potential at the FABRICATED elixir
          Phi_solv(cur)  = -0.01625
          solvency F = +0.08377, auto_reset_mask=False, NOT masked

+0.084 on the first step of every episode, against a solvency term whose whole
legitimate per-episode magnitude is 0.039, and varying with a scenario's banked
starting elixir -- i.e. scenario-correlated reward noise in ~30% of episodes. It
was also the entire cause of the solvency term failing to telescope.
"""
import gymnasium as gym
import numpy as np
import pytest

from python_ai.rewards.elixir_shaping import solvency_shaping
from python_ai.rewards.shaping import compute_shaping
from python_ai.rl import engine_stats
from python_ai.rl.config import PPOConfig

GAMMA = 0.999


def _stats(elixir, n=2):
    """What extract_engine_stats returns for a quiet board at `elixir`."""
    infos = {"elixir": np.full(n, elixir, dtype=np.float32)} if elixir is not None else {}
    return engine_stats.extract_engine_stats(infos, n)


def test_the_bug_exists_without_reseating():
    """CONTROL that must fire: shaped against the phantom's default elixir, the
    first real step carries the measured spurious solvency reward. If this ever
    reads ~0, the test below is vacuous."""
    phantom, first_real = _stats(None), _stats(3.35)
    f = solvency_shaping(first_real, phantom, GAMMA)
    assert f[0] == pytest.approx(0.0838, abs=2e-3)


def test_reseating_removes_the_spurious_first_step_reward():
    phantom, first_real = _stats(None), _stats(3.35)
    first = np.array([True, True])
    prev = engine_stats.reseat_prev_stats(first_real, phantom, first)
    assert np.allclose(solvency_shaping(first_real, prev, GAMMA), 0.0, atol=1e-3)
    assert np.allclose(compute_shaping(first_real, prev, gamma=GAMMA), 0.0, atol=1e-3)


def test_reseating_only_touches_the_envs_that_just_reset():
    """An env mid-episode must keep its real previous state, or its legitimate
    shaping for this step is silently deleted."""
    prev = _stats(1.0)
    cur = _stats(3.35)
    out = engine_stats.reseat_prev_stats(cur, prev, np.array([True, False]))
    assert out["team0_elixir_current"][0] == pytest.approx(3.35)
    assert out["team0_elixir_current"][1] == pytest.approx(1.0)
    f = solvency_shaping(cur, out, GAMMA)
    assert f[0] == pytest.approx(0.0, abs=1e-3)
    assert f[1] > 0.05, "the mid-episode env lost its real solvency reward"


def test_reseating_is_a_no_op_when_nothing_just_reset():
    prev, cur = _stats(1.0), _stats(3.35)
    out = engine_stats.reseat_prev_stats(cur, prev, np.array([False, False]))
    for k in prev:
        assert np.array_equal(np.asarray(out[k]), np.asarray(prev[k])), k


# --- the wiring: does the real rollout use it? -----------------------------

@pytest.mark.slow
def test_the_rollout_never_shapes_a_first_real_step_against_the_phantom(tmp_path, monkeypatch):
    """Drive the REAL collect_rollout across episode boundaries (short matches)
    and record the prev_stats compute_shaping is handed on each env's first real
    step. Its elixir must be that step's own reading, never the phantom's 0."""
    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "run_path",
                        lambda name: str(tmp_path.joinpath(*name.split("/"))))

    seen = []
    real = base_trainer.compute_shaping

    def spy(stats, prev_stats, **kw):
        cur = np.array(stats["team0_elixir_current"], dtype=np.float32)
        prev = (np.full_like(cur, np.nan) if prev_stats is None else
                np.array(prev_stats["team0_elixir_current"], dtype=np.float32))
        seen.append((cur, prev))
        return real(stats, prev_stats, **kw)

    monkeypatch.setattr(base_trainer, "compute_shaping", spy)

    cfg = PPOConfig(num_envs=2, update_timestep=60, bptt_chunk=10,
                    num_minibatches=1, ppo_epochs=1,
                    save_every_episodes=10 ** 9, replay_every_episodes=10 ** 9)

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / "runs")
            self.dones_log = []

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0,
                     "max_ticks": 150})
            return gym.vector.SyncVectorEnv([make for _ in range(cfg.num_envs)])

        def on_step(self, ctx):
            self.dones_log.append(np.array(ctx.dones))
            super().on_step(ctx)

        def should_stop(self):
            return True

        def on_finish(self):
            self.envs.close()

    trainer = Harness()
    trainer.setup()
    trainer.collect_rollout()

    dones = trainer.dones_log
    checked = 0
    for t in range(2, len(seen)):
        cur, prev = seen[t]
        first_real = dones[t - 2]           # done at t-2 -> phantom at t-1
        for i in np.flatnonzero(first_real):
            checked += 1
            assert prev[i] == pytest.approx(cur[i]), (
                f"step {t} env {i}: first real step shaped against elixir "
                f"{prev[i]} (the phantom default) instead of {cur[i]}")
    assert checked > 0, "no episode boundary was crossed; the test is vacuous"
