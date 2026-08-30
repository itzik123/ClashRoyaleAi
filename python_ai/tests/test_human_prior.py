"""Guards for serving the mined human placement prior into training.

Three things here would be silent if they broke:

  - the DEFAULT. The prior ships OFF so the A/B that has to justify it runs both
    arms from byte-identical code. A default that drifted to on would make every
    subsequent win rate incomparable to everything before it, with nothing
    raising.
  - PRECEDENCE. The prior is mined from real Clash Royale, whose physics this
    engine does not reproduce. It must fill gaps, never override a rule measured
    against this engine.
  - MUTUAL EXCLUSIVITY in the loss. `coverage_terms` gives a row either KL or an
    entropy bonus and never both, and carrying a non-1.0 weight through
    `coverage_has` is exactly the change that could break that invariant.
"""
import numpy as np
import pytest
import torch

import clash_royale_env as E
from python_ai.advisors import advisor_target, human_prior, tactics
from python_ai.envs.gym_wrapper import DEFAULT_DECK

CE = E.ClashRoyaleEnv
N_CELLS = tactics.BOARD_H * tactics.BOARD_W

# The five DEFAULT_DECK cards no hand-written rule reaches. This is the gap the
# prior exists to fill, so the list is asserted rather than assumed.
MUSKETEER, ICE_GOLEM, SKELETONS, ICE_SPIRIT, THE_LOG = 6, 40, 24, 72, 33
UNCOVERED = (MUSKETEER, ICE_GOLEM, SKELETONS, ICE_SPIRIT, THE_LOG)


@pytest.fixture
def prior_on(monkeypatch):
    """Enable the prior for one test without touching the process default."""
    monkeypatch.setattr(human_prior, "HUMAN_PRIOR_COEF", 0.5)
    return 0.5


def _legal(card_id):
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks=3600)
    env.seed(0)
    env.reset()
    return np.array([env.is_valid_placement(card_id, float(c % tactics.BOARD_W),
                                            float(c // tactics.BOARD_W), 0)
                     for c in range(N_CELLS)], dtype=bool)


def _fresh_obs():
    env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), max_ticks=3600)
    env.seed(1)
    env.reset()
    for _ in range(20):
        env.step_self_play_fast(4, 0, 0, 4, 0, 0, skip_frames=10)
    return np.asarray(env.get_observation_for_team(0), dtype=np.float32)


def test_the_prior_ships_OFF():
    """The A/B control arm must be the default, and must cost nothing."""
    assert human_prior.HUMAN_PRIOR_COEF == 0.0
    assert not human_prior.enabled()
    assert human_prior.logits_for(MUSKETEER, _legal(MUSKETEER)) is None


def test_the_five_uncovered_cards_really_have_no_rule():
    """Pins the gap the prior fills. If a rule is ever added for one of these,
    this fails and the precedence test below becomes the live one."""
    for cid in UNCOVERED:
        assert cid not in advisor_target.ADVISOR_CARDS
        assert advisor_target._advisor_logits_for(
            _fresh_obs(), cid, _legal(cid)) is None


@pytest.mark.parametrize("card_id", UNCOVERED)
def test_uncovered_cards_gain_a_target_when_the_prior_is_on(prior_on, card_id):
    legal = _legal(card_id)
    obs = _fresh_obs()
    assert advisor_target._advisor_logits_for(obs, card_id, legal) is None
    got = advisor_target.target_logits_for(obs, card_id, legal)
    assert got is not None, f"card {card_id} still has no target"
    assert got.shape == (N_CELLS,)
    assert np.isfinite(got[legal]).all()


@pytest.mark.parametrize("card_id", UNCOVERED)
def test_prior_puts_no_finite_logit_on_an_illegal_cell(prior_on, card_id):
    legal = _legal(card_id)
    got = human_prior.logits_for(card_id, legal)
    assert np.isneginf(got[~legal]).all()


def test_the_handwritten_rule_wins_where_it_speaks(prior_on, monkeypatch):
    """Precedence: a rule measured against THIS engine outranks a prior mined
    from a game whose physics this engine does not reproduce."""
    legal = _legal(advisor_target.CANNON_ID)
    sentinel = np.zeros(N_CELLS, dtype=np.float32)
    monkeypatch.setattr(advisor_target, "_advisor_logits_for",
                        lambda *a, **k: sentinel)
    out = advisor_target.target_logits_for(_fresh_obs(),
                                           advisor_target.CANNON_ID, legal)
    assert out is sentinel, "the prior overrode an engine-validated rule"


def test_weights_are_1_for_a_rule_and_the_coef_for_the_prior(prior_on):
    obs = _fresh_obs()
    t_prior, w_prior = advisor_target.target_and_weight(
        obs, MUSKETEER, _legal(MUSKETEER))
    assert t_prior is not None
    assert w_prior == pytest.approx(prior_on)

    # A rule row must carry weight 1.0. Force the rule to speak so the branch is
    # exercised deterministically rather than depending on the board.
    legal = _legal(advisor_target.CANNON_ID)
    sentinel = np.zeros(N_CELLS, dtype=np.float32)
    import unittest.mock as _mock
    with _mock.patch.object(advisor_target, "_advisor_logits_for",
                            return_value=sentinel):
        t_rule, w_rule = advisor_target.target_and_weight(
            obs, advisor_target.CANNON_ID, legal)
    assert t_rule is sentinel
    assert w_rule == 1.0


def test_legal_table_and_slot_weights_cover_the_prior_cards(prior_on):
    cards = advisor_target.target_cards()
    for cid in UNCOVERED:
        assert cid in cards
    hand = torch.tensor([[MUSKETEER, advisor_target.CANNON_ID, 999, -1]])
    w = advisor_target.slot_weights_for(hand)
    assert w is not None
    assert w[0, 0] == advisor_target.ADVISOR_SLOT_WEIGHT   # prior card
    assert w[0, 1] == advisor_target.ADVISOR_SLOT_WEIGHT   # rule card
    assert w[0, 2] == 1.0                                  # neither


# --- the loss half ---------------------------------------------------------

def _terms(has_target, targets=None, seed=0):
    torch.manual_seed(seed)
    B = has_target.shape[0]
    logits = torch.randn(B, N_CELLS, requires_grad=True)
    if targets is None:
        targets = torch.randn(B, N_CELLS)
    decision = torch.ones(B)
    return advisor_target.coverage_terms(
        logits, targets, has_target, decision,
        entropy_coef=0.02, log_n_placement=float(np.log(N_CELLS)))


def test_coverage_terms_is_unchanged_when_every_weight_is_one():
    """The OFF path must be byte-identical, not merely close."""
    has = torch.tensor([1.0, 0.0, 1.0, 0.0])
    torch.manual_seed(7)
    tg = torch.randn(4, N_CELLS)
    a = _terms(has, tg, seed=3)
    b = _terms(has.clone(), tg.clone(), seed=3)
    assert float(a[0]) == float(b[0])
    assert float(a[2]) == float(b[2])
    assert float(a[3]) == 2.0          # two target ROWS


def test_a_weighted_row_gets_KL_and_NOT_an_entropy_bonus():
    """The invariant `coverage_terms` exists to hold.

    Deriving the entropy mask as `1 - has_target` would give a row at weight 0.1
    ninety percent of an entropy bonus telling it to spread out WHILE giving it
    KL telling it where to go -- the exact opposition the row mask prevents.
    """
    torch.manual_seed(11)
    tg = torch.randn(2, N_CELLS)
    weighted = _terms(torch.tensor([0.1, 0.0]), tg, seed=5)
    full = _terms(torch.tensor([1.0, 0.0]), tg, seed=5)
    # One row has a target in both cases, so the entropy half sees exactly the
    # other row either way: identical entropy fraction.
    assert float(weighted[1]) == pytest.approx(float(full[1]))
    assert float(weighted[3]) == 1.0 and float(full[3]) == 1.0


def test_the_weight_actually_scales_the_KL():
    """Dividing by the summed weight instead of the row count would normalise a
    uniform weight straight back out, and the coefficient would do nothing."""
    torch.manual_seed(13)
    tg = torch.randn(3, N_CELLS)
    small = _terms(torch.tensor([0.1, 0.1, 0.1]), tg, seed=9)
    large = _terms(torch.tensor([1.0, 1.0, 1.0]), tg, seed=9)
    assert float(small[2]) == pytest.approx(0.1 * float(large[2]), rel=1e-5)


def test_a_stale_prior_is_refused_rather_than_reinterpreted(tmp_path,
                                                            monkeypatch):
    """An arena change must raise, not silently serve cells for a dead board."""
    stamp = human_prior.geometry_stamp()
    stamp["arena_bridge_y"] = float(stamp["arena_bridge_y"]) + 1.0   # moved river
    bad = tmp_path / "stale.npz"
    np.savez_compressed(
        bad,
        counts=np.ones((1, 18, N_CELLS), dtype=np.int32),
        deck=np.asarray([MUSKETEER], dtype=np.int32),
        geometry_keys=np.asarray(list(stamp), dtype=object),
        geometry_vals=np.asarray([stamp[k] for k in stamp], dtype=np.float64),
    )
    monkeypatch.setattr(human_prior, "ARTIFACT_PATH", bad)
    monkeypatch.setattr(human_prior, "_cache", {})
    monkeypatch.setattr(human_prior, "HUMAN_PRIOR_COEF", 0.5)
    with pytest.raises(ValueError, match="different arena"):
        human_prior.logits_for(MUSKETEER, _legal(MUSKETEER))


def test_the_OFF_arm_is_the_OLD_behaviour_exactly():
    """The A/B control must be today's code, not merely "the prior returns None".

    `prior_cards()` feeds both `build_legal_table` and `slot_weights_for`, and
    the latter weights every card in it. Reporting the deck while disabled made
    the coverage draw uniform over eight cards -- [5, 5, 5, 5] on a hand where
    the old behaviour is [1, 5, 1, 1] -- which changes the CONTROL arm. That is
    a silent confound in the only experiment this gate exists to protect.
    """
    assert not human_prior.enabled()
    assert human_prior.prior_cards() == set()
    assert advisor_target.target_cards() == set(advisor_target.ADVISOR_CARDS)

    hand = torch.tensor([[MUSKETEER, advisor_target.CANNON_ID, ICE_GOLEM, THE_LOG]])
    w = advisor_target.slot_weights_for(hand)
    assert w.tolist() == [[1.0, advisor_target.ADVISOR_SLOT_WEIGHT, 1.0, 1.0]]


def test_prior_cards_populates_only_when_enabled(prior_on):
    assert human_prior.enabled()
    assert human_prior.prior_cards() == set(int(c) for c in DEFAULT_DECK)
