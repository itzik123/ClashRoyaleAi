"""The deck floor fires only where the card is worth playing.

A Cannon is worth a lot when something is attacking and roughly nothing on a
quiet board; an unconditional floor forced wasted plays on ordinary boards and
cost win rate. Gating on `tactics.threat_level` keeps the push only where the
value is.
"""
import numpy as np
import torch

from python_ai.advisors import tactics
from python_ai.rl.deck_coverage import deck_coverage_penalty


def _logits_with(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float32))


# --- the batched threat signal must agree with the scalar one ---
def test_batched_threat_agrees_with_the_scalar_definition(fresh_obs):
    """One definition of threat: the batched form, needed per row by the PPO
    update, must reproduce `threat_level` exactly.
    """
    _, obs = fresh_obs
    obs = np.asarray(obs, dtype=np.float32)
    batch = torch.tensor(np.stack([obs, obs, obs]))

    got = tactics.threat_level_batch(batch)
    want = tactics.threat_level(obs)

    assert got.shape == (3,)
    for v in got.tolist():
        assert abs(v - want) < 1e-3


def test_an_empty_board_reads_as_no_threat(fresh_obs):
    _, obs = fresh_obs
    batch = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :])
    assert float(tactics.threat_level_batch(batch)[0]) == 0.0


# --- the gate ---
def test_the_floor_is_silent_on_an_unthreatened_row():
    """No push where the card is not worth playing."""
    logits = _logits_with([[0.001, 0.30, 0.30, 0.30, 0.099]])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, _, n = deck_coverage_penalty(logits, hand, decision, floor=0.02,
                                      threat=torch.zeros(1))
    assert pen.item() == 0.0
    assert n == 0


def test_the_floor_still_fires_on_a_threatened_row():
    logits = _logits_with([[0.001, 0.30, 0.30, 0.30, 0.099]])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    pen, _, n = deck_coverage_penalty(logits, hand, decision, floor=0.02,
                                      threat=torch.ones(1))
    assert pen.item() > 0.0
    assert n == 4


def test_only_the_threatened_rows_contribute():
    """A mixed batch behaves as if the quiet rows were not there."""
    rows = [[0.001, 0.30, 0.30, 0.30, 0.099],
            [0.001, 0.30, 0.30, 0.30, 0.099]]
    logits = _logits_with(rows)
    hand = torch.tensor([[10, 20, 30, 40], [10, 20, 30, 40]])
    decision = torch.ones(2)

    mixed = deck_coverage_penalty(logits, hand, decision, floor=0.02,
                                  threat=torch.tensor([1.0, 0.0]))[0]
    only_first = deck_coverage_penalty(logits[:1], hand[:1], torch.ones(1),
                                       floor=0.02, threat=torch.ones(1))[0]
    torch.testing.assert_close(mixed, only_first)


def test_omitting_the_gate_keeps_the_old_behaviour():
    """threat=None means every row, so existing callers are unchanged."""
    logits = _logits_with([[0.001, 0.30, 0.30, 0.30, 0.099]])
    hand = torch.tensor([[10, 20, 30, 40]])
    decision = torch.ones(1)

    ungated = deck_coverage_penalty(logits, hand, decision, floor=0.02)[0]
    all_threat = deck_coverage_penalty(logits, hand, decision, floor=0.02,
                                       threat=torch.ones(1))[0]
    torch.testing.assert_close(ungated, all_threat)


def test_a_fully_quiet_batch_is_a_finite_zero():
    """Same degenerate-batch guard as the decision mask: never NaN."""
    logits = _logits_with([[0.001, 0.30, 0.30, 0.30, 0.099]])
    hand = torch.tensor([[10, 20, 30, 40]])

    pen, min_p, n = deck_coverage_penalty(logits, hand, torch.ones(1),
                                          floor=0.02, threat=torch.zeros(1))
    assert n == 0
    assert pen.item() == 0.0
    assert torch.isfinite(pen).all()
    assert min_p == 0.0


def test_the_gate_is_actually_WIRED_into_the_ppo_update():
    """End to end: a quiet rollout leaves the term inactive, proving `rl/ppo.py`
    actually computes and passes the mask; otherwise the term would run ungated
    while every unit test passed.
    """
    import clash_royale_env
    import torch.optim as optim
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.models.policy_io import LSTM_HIDDEN
    from python_ai.rl.buffer import CORE_FIELDS, RolloutBuffer
    from python_ai.rl.config import PPOConfig
    from python_ai.rl.ppo import PPOUpdater

    cfg = PPOConfig(num_envs=2, update_timestep=4, bptt_chunk=2,
                    num_minibatches=1, ppo_epochs=1)
    torch.manual_seed(3)
    net = MicroRoyaleNet(num_ability_slots=0)
    deck = list(DEFAULT_DECK)
    envs = [clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
            for _ in range(cfg.num_envs)]
    for e in envs:
        e.reset()                       # quiet: nothing injected

    buf = RolloutBuffer(CORE_FIELDS)
    hx = torch.zeros(cfg.num_envs, LSTM_HIDDEN)
    cx = torch.zeros(cfg.num_envs, LSTM_HIDDEN)
    for _ in range(cfg.update_timestep):
        obs = torch.tensor(np.asarray(
            [e.get_observation_for_team(0) for e in envs], dtype=np.float32))
        mask = net.affordability_mask(obs)
        with torch.no_grad():
            feats, embeds, spatial = net.extract_features(obs)
            logits, _, _, value, (hx2, cx2) = net.step_lstm_and_card(
                feats, (hx, cx), mask)
            card = torch.distributions.Categorical(logits=logits).sample()
            pl = net.placement_given_card(hx2, embeds, card, obs, spatial)
            cell = torch.distributions.Categorical(logits=pl).sample()
            lp = (torch.distributions.Categorical(logits=logits).log_prob(card)
                  + torch.distributions.Categorical(logits=pl).log_prob(cell))
        buf.add(obs=obs, card_actions=card, placement_actions=cell,
                decision=(mask.sum(dim=1) > 1).float(),
                hx_in=hx, cx_in=cx, logprobs=lp, values=value.squeeze(-1),
                rewards=torch.zeros(cfg.num_envs),
                masks=torch.ones(cfg.num_envs), valid=torch.ones(cfg.num_envs),
                aux_opp_played=torch.full((cfg.num_envs,), -1, dtype=torch.long),
                coverage_slot=torch.zeros(cfg.num_envs, dtype=torch.long))
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(int(card[i]), 0.0, 0.0, 10)

    batch = buf.stack()
    T, N = batch["rewards"].shape
    stats = PPOUpdater(net, optim.Adam(net.parameters(), lr=1e-4), cfg).update(
        batch, torch.randn(T, N), torch.zeros(T, N), vf_clip_range=0.2,
        ent_coef_card=0.05, ent_coef_placement=0.06, coverage_coef=0.02,
        deck_coverage_coef=50.0)

    # No threatened rows, nothing measured: NaN is `_mean([])`'s report for "no
    # samples".
    assert np.isnan(stats.deck_coverage), (
        "the term ran on a quiet board -- the gate is not wired into ppo.py")
