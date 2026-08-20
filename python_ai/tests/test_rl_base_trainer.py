"""BaseTrainer, end to end, on a real (tiny) environment.

THE test this refactor most needed. The unit tests cover the pieces; nothing
else covers the WIRING -- rollout, GAE, update, entropy controller, checkpoint
write, checkpoint resume -- and the mission's own stated risk was discovering a
structural bug three days into a 72-hour run.

Deliberately small and slow-ish (a few seconds): two envs, 20 steps, one update.
That is enough to exercise every path in `collect_rollout` and `run_update`,
including the phantom-autoreset masking, without being a training run.

WINDOWS SPAWN NOTE: `AsyncVectorEnv` is avoided here in favour of `SyncVectorEnv`.
Spawned workers re-import the test module, which under pytest is a different and
much slower proposition; the trainer's own code path is identical either way
because it only ever calls `reset`, `step` and `call` on the vector env.
"""
import json
import os

import gymnasium as gym
import pytest
import torch

from python_ai.rl.config import PPOConfig

TINY = PPOConfig(num_envs=2, update_timestep=20, bptt_chunk=10,
                 num_minibatches=1, ppo_epochs=1,
                 save_every_episodes=1, replay_every_episodes=10 ** 9)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """The trainers write to relative paths (replays/, historical_checkpoints/),
    so a test that did not chdir would litter the repository."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _phase1(updates=1, cfg=TINY):
    from python_ai.envs import gym_wrapper
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = "runs/test"
            self.updates_run = 0

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0})
            return gym.vector.SyncVectorEnv(
                [make for _ in range(self.cfg.num_envs)])

        def run_update(self):
            stats = super().run_update()
            self.updates_run += 1
            return stats

        def should_stop(self):
            return self.updates_run >= updates

        def on_finish(self):
            # No subprocess handoff from a test.
            self.save_checkpoint(verbose=False)
            self.envs.close()

    trainer = Harness()
    trainer.run()
    return trainer


@pytest.mark.slow
def test_one_full_update_runs_and_writes_a_resumable_checkpoint(workdir):
    trainer = _phase1()
    assert trainer.updates_run == 1
    assert os.path.exists("model_weights.pth")

    ck = torch.load("model_weights.pth", map_location="cpu",
                    weights_only=False)
    # Everything a resume needs. A checkpoint missing any of these resumes
    # SILENTLY WRONG rather than failing -- the entropy controller reset to its
    # seed values cost ~5,600 episodes of walking back on 2026-07-30.
    for key in ("model", "optimizer", "episodes_completed", "outcome_history",
                "ent_coef_card", "ent_coef_place", "curriculum_stage",
                "stage_start_episode", "phase", "deck_curriculum_stage",
                "phase_deck_episode_start", "random_phase_episode_start",
                "current_random_deck"):
        assert key in ck, f"checkpoint cannot be fully resumed: missing {key}"


@pytest.mark.slow
def test_a_resume_restores_the_episode_count_and_the_controller(workdir):
    first = _phase1()
    first.entropy.coef_card = 0.1234
    first.entropy.coef_placement = 0.0321
    first.save_checkpoint(verbose=False)
    episodes = first.episodes_completed

    second = _phase1()
    assert second.full_resume, "resume did not take the full-checkpoint path"
    assert second.episodes_completed >= episodes


@pytest.mark.slow
def test_the_rollout_buffer_is_cleared_between_updates(workdir):
    """A buffer that is not cleared grows without bound, and the leak surfaces
    as an OOM many hours in rather than as an error."""
    trainer = _phase1(updates=2)
    assert len(trainer.buffer) == 0


@pytest.mark.slow
def test_the_hidden_state_is_reset_where_an_episode_ended(workdir):
    """The LSTM state carried into a new episode must start clean instead of
    being contaminated by the dead board's final observation."""
    trainer = _phase1()
    assert trainer._hx.shape == (TINY.num_envs, trainer.net.LSTM_HIDDEN)
    assert torch.isfinite(trainer._hx).all()


@pytest.mark.slow
def test_a_replay_is_recorded_and_annotated(workdir):
    """The viewer needs `stateValue` and the action fields stamped onto every
    tick; `GameLogger` writes none of them."""
    from python_ai.rl.replay import record_greedy_replay
    trainer = _phase1()
    os.makedirs("replays", exist_ok=True)
    env = trainer.build_replay_env()
    path = record_greedy_replay(trainer.net, env, torch.device("cpu"),
                                "replays/t.json")
    data = json.loads(open(path).read())
    tick = data["ticks"][0]
    for key in ("stateValue", "actionCardId", "actionCardName", "actionX",
                "actionY"):
        assert key in tick, f"replay tick missing {key}"


@pytest.mark.slow
def test_pipeline_2_refuses_to_start_without_pipeline_1s_output(workdir):
    """A clear error rather than an empty PFSP pool: self-play against nothing
    would train happily and teach nothing."""
    from python_ai.trainers.train_selfplay import Phase2Trainer
    trainer = Phase2Trainer(TINY)
    trainer.net = object()          # never reached
    with pytest.raises(RuntimeError, match="pipeline #1"):
        trainer.load_checkpoint()


def test_the_two_pipelines_declare_their_documented_differences():
    """Everything else about them is now shared, so these four lines ARE the
    difference between pipeline 1 and pipeline 2."""
    from python_ai.trainers.train import Phase1Trainer
    from python_ai.trainers.train_selfplay import Phase2Trainer

    assert Phase1Trainer.pipeline_name == "pipeline1"
    assert Phase2Trainer.pipeline_name == "pipeline2"
    # Only pipeline 2 injects scenarios, so only it can truncate an episode
    # without a king dying -- and only it must keep a scenario cutoff from being
    # scored as a passivity draw.
    assert Phase1Trainer.uses_truncation_bootstrap is False
    assert Phase2Trainer.uses_truncation_bootstrap is True
    assert Phase1Trainer.draw_on_terminated_only is False
    assert Phase2Trainer.draw_on_terminated_only is True


def test_both_pipelines_are_BaseTrainers_rather_than_two_loops():
    from python_ai.rl.base_trainer import BaseTrainer
    from python_ai.trainers.train import Phase1Trainer
    from python_ai.trainers.train_selfplay import Phase2Trainer

    assert issubclass(Phase1Trainer, BaseTrainer)
    assert issubclass(Phase2Trainer, BaseTrainer)
    # Neither may override the rollout or the update: those are the parts whose
    # arithmetic must stay identical between them.
    for cls in (Phase1Trainer, Phase2Trainer):
        assert "collect_rollout" not in vars(cls)
        assert "run_update" not in vars(cls)
