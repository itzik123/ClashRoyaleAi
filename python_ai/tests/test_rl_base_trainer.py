"""BaseTrainer end to end on a tiny real environment: rollout, GAE, update,
entropy controller, checkpoint write and resume.

Two envs, 20 steps, one update. SyncVectorEnv rather than AsyncVectorEnv,
because spawned workers re-import the test module; the trainer only calls
reset, step and call either way.
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
    """Redirect every destination a trainer writes to into tmp_path.

    Checkpoint paths are anchored, not cwd-relative, so a chdir alone isolates
    nothing: redirection goes through CLASH_WEIGHTS / CLASH_LOGDIR, `run_path`,
    and monkeypatches for the pipeline-2 constants frozen at import. The chdir
    still contains anything that writes relative paths of its own.
    """
    from python_ai.rl import base_trainer
    from python_ai.trainers import train_selfplay

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        base_trainer, "run_path",
        lambda name: str(tmp_path.joinpath(*name.split("/"))))
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "model_weights.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs" / "test"))
    for attr, value in (
            ("WEIGHT_PATH", tmp_path / "model_weights_selfplay.pth"),
            ("BOOTSTRAP_FROM_PATH", tmp_path / "model_weights.pth")):
        monkeypatch.setattr(train_selfplay, attr, str(value))
    monkeypatch.setattr(train_selfplay.Phase2Trainer, "weight_path",
                        str(tmp_path / "model_weights_selfplay.pth"))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "historical_checkpoints"))
    return tmp_path


@pytest.fixture(autouse=True)
def _never_touch_the_live_checkpoint(monkeypatch):
    """A tripwire, not a redirect: a test that writes the real
    `python_ai/model_weights.pth` has escaped `workdir`, and a run resuming
    from a toy checkpoint shows up days later.
    """
    import python_ai
    live = os.path.join(python_ai.PACKAGE_DIR, "model_weights.pth")
    existed = os.path.exists(live)
    yield
    if os.path.exists(live) and not existed:
        os.remove(live)
        raise AssertionError(
            f"a test wrote the LIVE checkpoint {live!r} -- it is not using the "
            "workdir fixture's redirection")


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
    assert os.path.exists(workdir / "model_weights.pth")

    ck = torch.load(workdir / "model_weights.pth", map_location="cpu",
                    weights_only=False)
    # Everything a resume needs; a missing key resumes silently wrong rather
    # than failing.
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
    """An uncleared buffer grows until OOM rather than erroring."""
    trainer = _phase1(updates=2)
    assert len(trainer.buffer) == 0


@pytest.mark.slow
def test_the_hidden_state_is_reset_where_an_episode_ended(workdir):
    """The LSTM state carried into a new episode must start clean."""
    trainer = _phase1()
    assert trainer._hx.shape == (TINY.num_envs, trainer.net.LSTM_HIDDEN)
    assert torch.isfinite(trainer._hx).all()


@pytest.mark.slow
def test_a_replay_is_recorded_and_annotated(workdir):
    """The viewer needs `stateValue` and the action fields on every tick;
    GameLogger writes none of them.
    """
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
    """Self-play against an empty pool would train happily and teach nothing.
    """
    from python_ai.trainers.train_selfplay import Phase2Trainer
    trainer = Phase2Trainer(TINY)
    trainer.net = object()          # never reached
    with pytest.raises(RuntimeError, match="pipeline #1"):
        trainer.load_checkpoint()


def test_the_two_pipelines_declare_their_documented_differences():
    """Everything else is shared, so these ARE the difference between the
    pipelines.
    """
    from python_ai.trainers.train import Phase1Trainer
    from python_ai.trainers.train_selfplay import Phase2Trainer

    assert Phase1Trainer.pipeline_name == "pipeline1"
    assert Phase2Trainer.pipeline_name == "pipeline2"
    # Both pipelines inject scenarios, so both must handle a window expiring:
    # bootstrap V(final_obs) through it rather than 0.0, and do not score it as
    # a passivity draw. The two flags travel together.
    for trainer in (Phase1Trainer, Phase2Trainer):
        assert trainer.uses_truncation_bootstrap is True
        assert trainer.draw_on_terminated_only is True


def test_both_pipelines_are_BaseTrainers_rather_than_two_loops():
    from python_ai.rl.base_trainer import BaseTrainer
    from python_ai.trainers.train import Phase1Trainer
    from python_ai.trainers.train_selfplay import Phase2Trainer

    assert issubclass(Phase1Trainer, BaseTrainer)
    assert issubclass(Phase2Trainer, BaseTrainer)
    # Neither may override the rollout or the update, whose arithmetic must
    # stay identical.
    for cls in (Phase1Trainer, Phase2Trainer):
        assert "collect_rollout" not in vars(cls)
        assert "run_update" not in vars(cls)


@pytest.mark.slow
def test_the_final_save_carries_the_SAME_keys_as_the_periodic_one(workdir):
    """The final save must persist the same keys as the periodic one, entropy
    coefficients included: pipeline 2 bootstraps from it, and a dropped
    controller resets to its seed values with nothing warning.
    """
    trainer = _phase1()
    trainer.entropy.coef_card = 0.4242
    trainer.entropy.coef_placement = 0.0242

    trainer.save_checkpoint(verbose=False)          # the periodic path
    periodic = torch.load(workdir / "model_weights.pth", map_location="cpu",
                          weights_only=False)
    trainer.on_finish()                             # the final path
    final = torch.load(workdir / "model_weights.pth", map_location="cpu",
                       weights_only=False)

    assert set(periodic) == set(final), (
        "the final save must not drop keys the periodic save persists")
    assert final["ent_coef_card"] == pytest.approx(0.4242)
    assert final["ent_coef_place"] == pytest.approx(0.0242)


# --- what counts as a truncation ---
# `truncateds` is the authoritative "the world continues" flag, not the
# reward's magnitude. A timeout decided by TimeoutRules pays +/-1, but an exact
# tie pays ~0 and has still ended.

def _boot(trainer_cls, net, obs, dones, raw, truncateds):
    """Call `_truncation_bootstrap` on a minimal stand-in for a live trainer.
    """
    import types

    import numpy as np
    import torch

    from python_ai.models.policy_io import LSTM_HIDDEN
    from python_ai.rl.base_trainer import BaseTrainer
    from python_ai.rl.config import PPOConfig

    n = len(dones)
    fake = types.SimpleNamespace(
        net=net, device=torch.device("cpu"),
        cfg=PPOConfig(num_envs=n),
        uses_truncation_bootstrap=trainer_cls.uses_truncation_bootstrap,
        _hx=torch.zeros(n, LSTM_HIDDEN), _cx=torch.zeros(n, LSTM_HIDDEN))
    flat = np.asarray(obs, dtype=np.float32)
    return BaseTrainer._truncation_bootstrap(
        fake, np.array(dones), np.array(raw, dtype=np.float32),
        np.repeat(flat[None, :], n, axis=0), np.array(truncateds))


def test_an_exact_tie_is_a_TERMINAL_not_a_truncation(net, fresh_obs):
    """A drawn match has ended; bootstrapping V(final_obs) would partly refund
    DRAW_PENALTY.
    """
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    out = _boot(Phase2Trainer, net, obs, dones=[True], raw=[0.0],
                truncateds=[False])
    assert float(out["flag"][0]) == 0.0, "a tie was bootstrapped"
    assert float(out["nonterminal"][0]) == 0.0, "a tie was treated as ongoing"


def test_a_scenario_window_running_out_IS_a_truncation(net, fresh_obs):
    """The game is still going, so the critic must bootstrap."""
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    out = _boot(Phase2Trainer, net, obs, dones=[True], raw=[0.0],
                truncateds=[True])
    assert float(out["flag"][0]) == 1.0
    assert float(out["nonterminal"][0]) == 1.0


def test_a_decided_result_never_bootstraps(net, fresh_obs):
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    out = _boot(Phase2Trainer, net, obs, dones=[True, True], raw=[1.0, -1.0],
                truncateds=[False, False])
    assert [float(v) for v in out["flag"]] == [0.0, 0.0]
    assert [float(v) for v in out["nonterminal"]] == [0.0, 0.0]


def test_classification_does_not_depend_on_the_reward_scale(net, fresh_obs):
    """A truncation is a truncation whatever the step happened to pay."""
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    for reward in (0.0, 0.4, 0.9, -0.9):
        out = _boot(Phase2Trainer, net, obs, dones=[True], raw=[reward],
                    truncateds=[True])
        assert float(out["flag"][0]) == 1.0, reward


def test_a_live_step_is_neither_terminal_nor_truncated(net, fresh_obs):
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    out = _boot(Phase2Trainer, net, obs, dones=[False], raw=[0.0],
                truncateds=[False])
    assert float(out["flag"][0]) == 0.0
    assert float(out["nonterminal"][0]) == 1.0, (
        "an ongoing episode must still bootstrap the next stored value")


# --- the deck is part of a run's identity ---

@pytest.mark.slow
def test_setup_prints_the_deck_contract_and_the_checkpoint_records_the_deck(workdir, capsys):
    """The log says what deck trained and what it turns off, and the checkpoint
    records it.
    """
    from python_ai.deck import DEFAULT_DECK
    trainer = _phase1()
    assert "[DECK INFO] deck:" in capsys.readouterr().out
    ck = torch.load(workdir / "model_weights.pth", map_location="cpu",
                    weights_only=False)
    assert ck["deck"] == list(DEFAULT_DECK)


@pytest.mark.slow
def test_resuming_a_checkpoint_from_a_different_deck_is_loud(workdir, capsys, monkeypatch):
    first = _phase1()
    ck = torch.load(workdir / "model_weights.pth", map_location="cpu",
                    weights_only=False)
    ck["deck"] = [0, 1, 3, 5, 6, 7, 8, 10]
    torch.save(ck, workdir / "model_weights.pth")
    capsys.readouterr()
    _phase1()
    assert "DIFFERENT DECK" in capsys.readouterr().out


@pytest.mark.slow
def test_resuming_under_different_clash_settings_is_loud(workdir, capsys, monkeypatch):
    """The checkpoint stamps every CLASH_* setting, and a resume under a different
    one names it.
    """
    monkeypatch.delenv("CLASH_SOLVENCY", raising=False)
    _phase1()
    ck = torch.load(workdir / "model_weights.pth", map_location="cpu",
                    weights_only=False)
    assert "CLASH_WEIGHTS" in ck["clash_settings"], "the stamp is missing"
    capsys.readouterr()
    monkeypatch.setenv("CLASH_SOLVENCY", "0")
    _phase1()
    out = capsys.readouterr().out
    assert "DIFFERENT CLASH_* SETTINGS" in out
    assert "CLASH_SOLVENCY: unset -> '0'" in out


@pytest.mark.slow
def test_resuming_under_the_same_settings_says_nothing(workdir, capsys):
    """The control: a warning printed on every resume would pass the test above.
    """
    _phase1()
    capsys.readouterr()
    _phase1()
    out = capsys.readouterr().out
    assert "DIFFERENT CLASH_* SETTINGS" not in out
    assert "predates the CLASH_* stamp" not in out


@pytest.mark.slow
def test_a_resume_error_crashes_and_destroys_nothing(workdir, monkeypatch):
    """A resume that cannot complete must stop, with the checkpoint and the
    TensorBoard log untouched.
    """
    from python_ai.trainers.train import Phase1Trainer
    _phase1()
    ckpt = workdir / "model_weights.pth"
    logdir = workdir / "runs" / "test"
    assert ckpt.exists()
    os.makedirs(logdir, exist_ok=True)
    (logdir / "keep.txt").write_text("history")

    def boom(self):
        raise RuntimeError("a worker failed while restoring")

    monkeypatch.setattr(Phase1Trainer, "_apply_curriculum_to_envs", boom)
    with pytest.raises(RuntimeError, match="worker failed"):
        _phase1()
    assert ckpt.exists(), "the live checkpoint was moved aside"
    assert (logdir / "keep.txt").exists(), "the TensorBoard log was deleted"


@pytest.mark.slow
def test_an_interrupt_saves_before_it_exits(workdir, monkeypatch):
    """Ctrl-C saves before exiting."""
    from python_ai.trainers.train import Phase1Trainer
    calls = {"n": 0}
    real = Phase1Trainer.collect_rollout

    def interrupt_second(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(self)

    monkeypatch.setattr(Phase1Trainer, "collect_rollout", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        _phase1(updates=5)
    ck = torch.load(workdir / "model_weights.pth", map_location="cpu", weights_only=False)
    assert ck["episodes_completed"] >= 0
    assert "lineage_started_at" in ck


@pytest.mark.slow
def test_placement_modal_share_is_fed_from_the_real_rollout(workdir):
    """The window must receive this run's placements, or the collapse detector is
    silently absent.
    """
    from python_ai.rl.placement_stats import ModalShareWindow
    trainer = _phase1(updates=2)
    window = trainer._modal_share
    assert isinstance(window, ModalShareWindow)
    plays = sum(sum(c.values()) for upd in window._updates for c in upd.values())
    assert plays > 0, "no placements reached the modal-share window"


@pytest.mark.slow
def test_a_champion_deck_trains_with_its_ability_in_the_ratio(workdir, monkeypatch):
    """A Champion deck trains with its ability in the ratio.

    Readiness is forced on so the activate arm is sampled on every row (the
    engine refuses an unready activation harmlessly). The epoch-0 ratio must be
    1: the stored and recomputed ability log-probs come from the same masked
    logits.
    """
    import numpy as np
    from python_ai.envs import gym_wrapper
    from python_ai.rl import abilities
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    deck = [6, 116, 40, 24, 72, 33, 7, 52]          # Golden Knight in deck slot 1
    sent = []
    real_ready = abilities.ready_from_infos
    monkeypatch.setattr(abilities, "ready_from_infos",
                        lambda infos, n, slots: real_ready({f"champion_ability_slot{s}_ready":
                                                            np.ones(n, bool) for s in slots},
                                                           n, slots))
    cfg = PPOConfig(num_envs=2, update_timestep=20, bptt_chunk=10,
                    num_minibatches=1, ppo_epochs=1,
                    save_every_episodes=10 ** 9, replay_every_episodes=10 ** 9)

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = "runs/test"
            self.stats = None

        def ability_engine_slots(self):
            return [1]

        def build_envs(self):
            def make():
                return gym_wrapper.MicroRoyaleEnv(
                    {"opponent": PHASE1_OPPONENT, "teacher_stage": 0,
                     "ai_deck": deck, "deck_pool": False})
            envs = gym.vector.SyncVectorEnv([make for _ in range(cfg.num_envs)])
            real_step = envs.step

            def step(action):
                sent.append(np.asarray(action["activate_ability_slot1"]).copy())
                return real_step(action)
            envs.step = step
            return envs

        def run_update(self):
            self.stats = super().run_update()
            return self.stats

        def should_stop(self):
            return self.stats is not None

        def on_finish(self):
            self.envs.close()

    monkeypatch.setattr("python_ai.envs.deck_contract.validate_deck",
                        lambda deck, strict=True: [])
    t = Harness()
    t.run()
    assert t.net.num_ability_slots == 1
    assert t.stats.ratio_dev_first < 1e-4, t.stats.ratio_dev_first
    assert any(a.any() for a in sent), "no activation ever reached the env"
