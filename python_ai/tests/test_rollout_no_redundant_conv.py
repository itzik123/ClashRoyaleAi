"""The rollout computes the trunk's first conv once per step.

`extract_features` computes the hires map and discards it, and
`placement_given_card(hires_map=None)` then rebuilds it: the most expensive
layer run twice on identical input. Threading `extract_features_hires`' map
through is bit-identical by construction (same module, same input, no RNG
between).
"""
import pytest
import torch

from python_ai.models.net import MicroRoyaleNet


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = MicroRoyaleNet()
    n.eval()
    return n


@pytest.fixture(scope="module")
def obs(net):
    torch.manual_seed(1)
    return torch.randn(4, net.spatial_size + net.scalar_size)


def _count_conv1(net, fn):
    """Invocations of the trunk's first conv while `fn` runs."""
    calls = []
    conv1 = net.cnn_trunk[0]
    original = conv1.forward

    def counting(x):
        calls.append(x.shape)
        return original(x)

    conv1.forward = counting
    try:
        result = fn()
    finally:
        conv1.forward = original
    return len(calls), result


def _rollout_step(net, obs, hx, cx, use_hires):
    """One step shaped like BaseTrainer.collect_rollout's inner body."""
    with torch.no_grad():
        if use_hires:
            feats, embeds, spatial, hires = net.extract_features_hires(obs)
        else:
            feats, embeds, spatial = net.extract_features(obs)
            hires = None
        mask = net.affordability_mask(obs)
        card_logits, _, _, value, (h, c) = net.step_lstm_and_card(
            feats, (hx, cx), mask)
        idx = card_logits.argmax(-1)
        place = net.placement_given_card(
            h, embeds, idx, obs, spatial, hires_map=hires)
    return place, value, h, c


def test_the_wrapper_really_does_discard_a_computed_map(net, obs):
    """Pins the premise: if the wrapper stops computing the map internally, this
    optimisation is moot and the test should say so.
    """
    n_wrapper, _ = _count_conv1(net, lambda: net.extract_features(obs))
    n_hires, _ = _count_conv1(net, lambda: net.extract_features_hires(obs))
    assert n_wrapper == 1 and n_hires == 1, (
        "both entry points should run conv1 exactly once; the wrapper's cost "
        "is that it then DROPS the result")


def test_a_rollout_step_runs_conv1_exactly_once(net, obs):
    """The regression test."""
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    n, _ = _count_conv1(net, lambda: _rollout_step(net, obs, hx, cx, True))
    assert n == 1, (
        f"the rollout ran the trunk's first conv {n} times in one step; "
        "pass extract_features_hires' map into placement_given_card")


def test_the_unthreaded_shape_is_what_costs_two(net, obs):
    """Negative control: the count is sensitive to the thing fixed."""
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    n, _ = _count_conv1(net, lambda: _rollout_step(net, obs, hx, cx, False))
    assert n == 2


def test_threading_the_map_is_BIT_IDENTICAL(net, obs):
    """Exact, not close. -inf is mapped to a finite sentinel for the comparison,
    since masked cells are -inf.
    """
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    a, va, ha, ca = _rollout_step(net, obs, hx, cx, False)
    b, vb, hb, cb = _rollout_step(net, obs, hx, cx, True)

    fix = lambda t: torch.nan_to_num(t, neginf=-1e30, posinf=1e30)
    assert torch.equal(fix(a), fix(b)), "placement logits diverged"
    assert torch.equal(va, vb), "state value diverged"
    assert torch.equal(ha, hb) and torch.equal(ca, cb), "LSTM state diverged"


def test_the_masked_cells_stay_masked(net, obs):
    """The bit-identity check above must exercise the mask."""
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    place, _, _, _ = _rollout_step(net, obs, hx, cx, True)
    assert torch.isneginf(place).any(), "no cell was masked; mask never ran"
    assert torch.isfinite(place).any(), "every cell masked; distribution empty"


@pytest.mark.slow
def test_the_live_trainer_rollout_runs_conv1_once_per_step(tmp_path,
                                                           monkeypatch):
    """End to end against the real collect_rollout, which the hand-written
    `_rollout_step` could drift from.
    """
    import gymnasium as gym

    from python_ai.envs import gym_wrapper
    from python_ai.rl import base_trainer
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(base_trainer, "HISTORICAL_CHECKPOINT_DIR",
                        str(tmp_path / "hc"))

    steps = 5
    cfg = PPOConfig(num_envs=2, update_timestep=steps, bptt_chunk=steps,
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
        n, _ = _count_conv1(t.net, t.collect_rollout)
    finally:
        t.envs.close()

    assert n == steps, (
        f"collect_rollout ran conv1 {n} times over {steps} steps "
        f"(expected exactly {steps}, i.e. once per step)")


# --- pipeline 2's opponent path ---
# `selfplay_env._opponent_action` runs once per env per step for all of
# pipeline 2, so it is threaded too; cold eval paths may rebuild the map.

@pytest.mark.slow
def test_the_selfplay_opponent_runs_conv1_once_per_decision(monkeypatch):
    import numpy as np

    from python_ai.envs import selfplay_env
    from python_ai.models.net import MicroRoyaleNet

    env = selfplay_env.MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
    env.reset()

    opp = MicroRoyaleNet(num_ability_slots=0)
    opp.eval()
    env.opponent_net = opp
    env.opponent_kind = "neural"
    env.opponent_hx = torch.zeros(1, opp.LSTM_HIDDEN)
    env.opponent_cx = torch.zeros(1, opp.LSTM_HIDDEN)

    n, _ = _count_conv1(opp, env._opponent_action)
    assert n == 1, (
        f"the frozen opponent ran conv1 {n} times for ONE decision; thread "
        "extract_features_hires' map into placement_given_card")


@pytest.mark.slow
def test_the_opponent_action_is_still_well_formed():
    """Threading does not change what the opponent returns."""
    from python_ai.envs import selfplay_env
    from python_ai.models.net import MicroRoyaleNet

    env = selfplay_env.MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})
    env.reset()
    opp = MicroRoyaleNet(num_ability_slots=0)
    opp.eval()
    env.opponent_net = opp
    env.opponent_kind = "neural"
    env.opponent_hx = torch.zeros(1, opp.LSTM_HIDDEN)
    env.opponent_cx = torch.zeros(1, opp.LSTM_HIDDEN)

    card, x, y, a1, a2 = env._opponent_action()
    assert 0 <= card <= opp.hand_size
    assert 0.0 <= x < opp.board_width
    assert 0.0 <= y < opp.placement_rows
    assert a1 in (0, 1, False, True) and a2 in (0, 1, False, True)
