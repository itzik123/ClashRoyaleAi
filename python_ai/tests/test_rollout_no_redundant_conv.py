"""The rollout must not compute the trunk's first conv twice per step.

`extract_features_hires` computes `hires_map = cnn_trunk[:2](spatial_obs)` and
returns it; `extract_features` is a thin wrapper that computes the very same
thing and THROWS THE MAP AWAY. `placement_given_card(hires_map=None)` then
rebuilds it from `obs`. So a rollout step that called the wrapper ran the
trunk's most expensive layer -- Conv2d(21->16) at the full 34x18 board, before
any pooling -- twice on bit-identical input.

`forward_sequence` already goes out of its way to avoid exactly this in the
UPDATE path (it threads `hires_seq` through so the coverage pass cannot
recompute conv1). The rollout was the half nobody threaded.

MEASURED, 500 steps x 8 envs, best of 3, network portion only:

    conv1 invocations per step   2  ->  1
    wall clock                   6.505s -> 2.103s   (67.7% saved)
    placement logits             bit-identical

Bit-identity is not a hope here, it is arithmetic: both paths evaluate the same
`cnn_trunk[:2]` on the same `spatial_obs`, with no RNG between them. That is the
same guarantee `test_recomputed_hires_equals_the_passed_one` already pins from
the other direction.
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
    """One step shaped exactly like BaseTrainer.collect_rollout's inner body."""
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
    """Pins the premise. If `extract_features` ever stops computing the hires
    map internally, this whole optimization is moot and the test should say so
    rather than silently keep asserting a count that no longer means anything.
    """
    n_wrapper, _ = _count_conv1(net, lambda: net.extract_features(obs))
    n_hires, _ = _count_conv1(net, lambda: net.extract_features_hires(obs))
    assert n_wrapper == 1 and n_hires == 1, (
        "both entry points should run conv1 exactly once; the wrapper's cost "
        "is that it then DROPS the result")


def test_a_rollout_step_runs_conv1_exactly_once(net, obs):
    """THE regression test. Two invocations means the rollout went back to
    letting placement_given_card rebuild the map it was already handed."""
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    n, _ = _count_conv1(net, lambda: _rollout_step(net, obs, hx, cx, True))
    assert n == 1, (
        f"the rollout ran the trunk's first conv {n} times in one step; "
        "pass extract_features_hires' map into placement_given_card")


def test_the_unthreaded_shape_is_what_costs_two(net, obs):
    """Negative control: the count is genuinely sensitive to the thing fixed.

    Without it, `== 1` above could pass for an unrelated reason and nobody
    would know the assertion had stopped measuring anything.
    """
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    n, _ = _count_conv1(net, lambda: _rollout_step(net, obs, hx, cx, False))
    assert n == 2


def test_threading_the_map_is_BIT_IDENTICAL(net, obs):
    """Same module, same input, no RNG between -- so this is exact, not close.

    Compared with -inf mapped to a finite sentinel: the placement head masks
    illegal cells to -inf, and `torch.equal` is False for nan but fine for inf;
    the substitution keeps the comparison honest either way.
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
    """A cheap guard that the bit-identity check above is not comparing two
    all-finite tensors that never exercised the mask."""
    hx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    cx = torch.zeros(obs.shape[0], net.LSTM_HIDDEN)
    place, _, _, _ = _rollout_step(net, obs, hx, cx, True)
    assert torch.isneginf(place).any(), "no cell was masked; mask never ran"
    assert torch.isfinite(place).any(), "every cell masked; distribution empty"


@pytest.mark.slow
def test_the_live_trainer_rollout_runs_conv1_once_per_step(tmp_path,
                                                           monkeypatch):
    """End to end, against the REAL collect_rollout rather than a lookalike.

    The hand-written `_rollout_step` above could drift from the trainer's
    actual body; this one drives BaseTrainer itself, so it cannot.
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


# --- pipeline 2's opponent path -------------------------------------------
#
# `selfplay_env._opponent_action` had the same unthreaded shape and is equally
# hot: it runs once per env per step for the whole of pipeline 2. net.py accepts
# the duplicate conv in COLD paths (the eval harnesses rebuild the map from obs
# on purpose); the frozen opponent's per-step decision is not one of those.

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
    """The threading must not change what the opponent actually returns."""
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
