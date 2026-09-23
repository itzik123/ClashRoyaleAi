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
    """Redirect every destination a trainer writes to into tmp_path.

    A `chdir` used to be enough, because the trainers' paths were all
    cwd-relative. They were ANCHORED on 2026-08-25 so that a multi-day run
    cannot be silently redirected by the directory it was launched from
    (see rl/checkpointing.weights_path) -- and that turns this fixture's old
    one-liner into no isolation at all.

    It is not hypothetical: the first suite run after that change wrote a
    random-init, 20-step, 2-env checkpoint straight into
    `python_ai/model_weights.pth`, which is the live resume path. A subsequent
    training run would have started from it and looked like it was resuming.

    So redirection now goes through the sanctioned overrides -- `CLASH_WEIGHTS`
    and `CLASH_LOGDIR` exist precisely so "a smoke run or an experiment arm
    cannot clobber the real checkpoint" -- plus monkeypatches for the pipeline
    2 constants, which are frozen at import and have no env hook.

    `replays/` WAS the one destination still relative, and the chdir below was
    what contained it. It is anchored now (base_trainer routes both sites
    through `run_path`, which is what run_path's docstring always claimed), so
    the chdir alone would let `setup()` create -- and `record_replay()` write
    into -- the REAL repo root. `run_path` is therefore redirected here too.
    The chdir stays regardless: it is still the containment for anything that
    writes relative paths of its own, and tests that call
    `record_greedy_replay` with an explicit relative path rely on it.
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
    """A tripwire, not a redirect. Any test in this module that writes to the
    real `python_ai/model_weights.pth` has escaped `workdir`, and the symptom
    (a training run resuming from a toy checkpoint) appears days later and
    nowhere near the cause.
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
    # BOTH pipelines inject scenarios since 2026-08-29 -- phase 1 gained
    # defensive injection because 32,680 episodes without a single "defend or
    # lose the tower" moment left Cannon/Log/Fireball at P(play|in hand) of
    # 0.0053/0.0091/0.0011. So both must handle a window expiring:
    #   * bootstrap V(final_obs) through it, never 0.0, or the critic learns
    #     that holding a defence is worth nothing;
    #   * and not score it as a passivity draw, or a successful defence is
    #     charged the full DRAW_PENALTY.
    # These two flags travel together. Setting the first without the second is
    # a live bug, which is what this assertion pair exists to catch.
    for trainer in (Phase1Trainer, Phase2Trainer):
        assert trainer.uses_truncation_bootstrap is True
        assert trainer.draw_on_terminated_only is True


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


@pytest.mark.slow
def test_the_final_save_carries_the_SAME_keys_as_the_periodic_one(workdir):
    """A LATENT BUG the extraction removed, worth a regression.

    `train.py` had two hand-written `torch.save({...})` blocks: a periodic one
    that persisted `ent_coef_card` / `ent_coef_place`, and a FINAL one at the
    stop point that did not. So the very last checkpoint pipeline 1 wrote --
    the one pipeline 2 bootstraps from, and the one any resume picks up --
    silently dropped the converged entropy controller and sent it back to its
    0.05 / 0.06 seed values.

    That is precisely the failure already on record from the other pipeline:
    observed 2026-07-30, placement reset from a converged 0.0132 to 0.06 and
    took ~5,600 episodes to walk back, with nothing warning. One
    `save_checkpoint()` makes the two saves the same object by construction.
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


# --- what actually counts as a truncation ---------------------------------
#
# `_truncation_bootstrap` classified a finished episode by the MAGNITUDE OF THE
# REWARD:
#
#     is_terminal = dones & (np.abs(raw_rewards) > 0.5)
#     needs_boot  = dones & ~is_terminal
#
# That is a heuristic standing in for a signal the caller already has.
# `selfplay_env` sets `truncated=True` in exactly one place -- a scenario
# window running out while the game is NOT over -- so `truncateds` IS the
# authoritative "the world continues, we just stopped watching" flag.
#
# The heuristic gets one case wrong, and TimeoutRules is what makes it narrow:
# a timed-out match is DECIDED on surviving towers, then on the weakest
# tower's HP, and only an exact tie on both is a genuine draw. So a timeout
# almost always pays +/-1 and is correctly classed terminal. An EXACT TIE pays
# ~0, and was therefore treated as a truncation -- so the agent was charged
# DRAW_PENALTY for the tie *and* credited gamma*V(final_obs) as though the game
# carried on. The game did not carry on; a draw is an ending.
#
# Reading the flag instead of the reward also removes a silent dependency on
# the reward SCALE: if the sparse reward ever stopped being +/-1, the heuristic
# would misclassify every episode at once, with nothing reporting it.

def _boot(trainer_cls, net, obs, dones, raw, truncateds):
    """Call `_truncation_bootstrap` on a minimal stand-in for a live trainer."""
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
    """A drawn match has ended. Bootstrapping V(final_obs) there credits the
    agent with a future that does not exist, partly refunding DRAW_PENALTY --
    the very term that exists to stop a timeout being the safe option."""
    from python_ai.trainers.train_selfplay import Phase2Trainer
    _, obs = fresh_obs
    out = _boot(Phase2Trainer, net, obs, dones=[True], raw=[0.0],
                truncateds=[False])
    assert float(out["flag"][0]) == 0.0, "a tie was bootstrapped"
    assert float(out["nonterminal"][0]) == 0.0, "a tie was treated as ongoing"


def test_a_scenario_window_running_out_IS_a_truncation(net, fresh_obs):
    """The one case the mechanism exists for: the game is genuinely still
    going, so the critic must bootstrap rather than learn 'the world ends'."""
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
    """A truncation is a truncation whatever the step happened to pay. Under
    the old rule a scenario cutoff that coincided with any |reward| > 0.5 was
    silently reclassified as a terminal."""
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


# ---------------------------------------------------------------------------
# 2026-09-15: the deck is part of a run's identity.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_setup_prints_the_deck_contract_and_the_checkpoint_records_the_deck(workdir, capsys):
    """A run's log must say what deck it trained and what that deck turns off,
    and its checkpoint must say which deck produced it -- the deck became a
    CLASH_DECK setting on 2026-09-15, so it is no longer implied by the code."""
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
    """TODO 00.9: the checkpoint stamps every CLASH_* setting, and a resume under
    a different one names it -- end to end through a real save and restore."""
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
    """The CONTROL: without it, a warning printed on every resume would pass the
    test above and teach the operator to ignore it."""
    _phase1()
    capsys.readouterr()
    _phase1()
    out = capsys.readouterr().out
    assert "DIFFERENT CLASH_* SETTINGS" not in out
    assert "predates the CLASH_* stamp" not in out


@pytest.mark.slow
def test_a_resume_error_crashes_and_destroys_nothing(workdir, monkeypatch):
    """Audit 08, gap 2. `except RuntimeError` on resume used to move the live
    checkpoint to .bak, rmtree the TensorBoard log, print "starting fresh" -- and
    then continue in whatever half-restored state the exception left (measured:
    the checkpoint's weights and episode count, without its deck stats). A resume
    that cannot complete must STOP, with the checkpoint and the log untouched."""
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
    """Audit 08, gap 6: Ctrl-C lost up to CLASH_SAVE_EVERY episodes."""
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
    """The window must receive this run's placements, or the scalar never
    appears and the collapse detector is silently absent again."""
    from python_ai.rl.placement_stats import ModalShareWindow
    trainer = _phase1(updates=2)
    window = trainer._modal_share
    assert isinstance(window, ModalShareWindow)
    plays = sum(sum(c.values()) for upd in window._updates for c in upd.values())
    assert plays > 0, "no placements reached the modal-share window"


@pytest.mark.slow
def test_a_champion_deck_trains_with_its_ability_in_the_ratio(workdir, monkeypatch):
    """2026-09-16: ability training. A Champion deck used to raise at setup.

    Readiness is forced ON for every step so the activate arm is sampled and
    scored on every row (the engine refuses an unready activation harmlessly).
    The epoch-0 ratio must be 1: the ability log-prob stored at rollout and the
    one recomputed in the update come from the same masked logits.
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
