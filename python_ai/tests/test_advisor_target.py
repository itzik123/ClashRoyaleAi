"""The spell-value anneal and the advisor-targeted coverage term.

Split out of the old single `test_python_ai.py` on 2026-08-20. The bodies are
unchanged -- only the shared header moved into `tests/conftest.py`, so the set of
test node ids is the same modulo the file name.

    python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
"""
import os
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
import clash_royale_env as E  # noqa: E402
from python_ai import engine_constants as EC  # noqa: E402
from python_ai.advisors import advisor_target as AT  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.envs import scenario_offense  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.rewards import shaping as T  # noqa: E402
from python_ai.rewards import shaping as train_shaping  # noqa: E402
from python_ai.rewards import weights as TW  # noqa: E402
from python_ai.rewards import weights as train_weights  # noqa: E402
from python_ai.rewards.elixir_shaping import (  # noqa: E402
    SOLVENCY_RESERVE, W_SOLVENCY, bankruptcy_rate, solvency_potential,
    solvency_shaping,
)
from python_ai.rl.coverage import (  # noqa: E402
    PLACEMENT_COVERAGE_COEF, placement_coverage_slots,
)
from python_ai.trainers.distill_tactics import masked_kl  # noqa: E402

CE = clash_royale_env.ClashRoyaleEnv

from python_ai.tests.helpers import (  # noqa: E402
    B, L, chunk_fixture, shaping_stats,
)


# ==========================================================================
# The spell-value anneal, and the advisor-targeted coverage term
# ==========================================================================
# Two changes made 2026-08-14, both GAMEPLAY-AFFECTING.
#
# 1. `spell_value_weight` was dead code: both trainers called
#    `compute_shaping(...)` without `w_spell`, so the Fireball-value weight sat
#    at W_SPELL_VALUE_START for the whole of training and the anneal its own
#    comment block describes never ran. It is wired in now.
#
# 2. The placement COVERAGE term was pure entropy -- "be spread out" -- and a
#    distilled placement map is exactly what that flattens. Coverage now
#    carries the advisor's own score map as a TARGET wherever the advisor has
#    something to say, and falls back to the entropy bonus where it does not.

from python_ai.advisors import advisor_target as AT  # noqa: E402
from python_ai.rewards import shaping as train_shaping  # noqa: E402
from python_ai.rewards import weights as train_weights  # noqa: E402
from python_ai.trainers.distill_tactics import masked_kl  # noqa: E402


def test_spell_value_weight_anneals_from_start_to_final():
    """The schedule the docstring always claimed, now actually reachable."""
    assert train_shaping.spell_value_weight(0) == pytest.approx(train_weights.W_SPELL_VALUE_START)
    end = train_weights.SPELL_VALUE_ANNEAL_EPISODES
    assert train_shaping.spell_value_weight(end) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    assert train_shaping.spell_value_weight(end * 10) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    # Monotone in between, and strictly decreasing end to end.
    xs = [train_shaping.spell_value_weight(int(end * f)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a >= b for a, b in zip(xs, xs[1:])), xs
    assert xs[0] > xs[-1]


def test_spell_value_weight_respects_the_start_offset():
    """Why the offset exists: a warm-start resumes PAST the anneal horizon.

    `model_weights_selfplay.pth` is at episode 64,309 against a 40,000-episode
    horizon, so a faithful wiring pins the weight at FINAL for the whole of any
    resumed run -- correct by the schedule, but it means the anneal can never be
    observed, and Phase 4 has to prove it happens. The offset slides the
    schedule onto the run that is actually being performed. With the default
    offset of 0 the behaviour is exactly the original intent.
    """
    end = train_weights.SPELL_VALUE_ANNEAL_EPISODES
    w = train_shaping.spell_value_weight
    assert w(1000, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_START)
    assert w(1000 + end, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_FINAL)
    # Before the offset the term is still at full strength, never extrapolated
    # past START.
    assert w(0, start=1000) == pytest.approx(train_weights.W_SPELL_VALUE_START)




def test_compute_shaping_actually_responds_to_w_spell():
    """The regression that let the dead code hide.

    Nothing detected `w_spell` being unreachable because no test ever varied
    it. This one does: a two-for-one Fireball must be worth strictly more at
    weight START than at weight 0.
    """
    cur, prev = shaping_stats(fireball_killed=8.0)
    hot = train_shaping.compute_shaping(cur, prev, w_spell=train_weights.W_SPELL_VALUE_START)
    off = train_shaping.compute_shaping(cur, prev, w_spell=0.0)
    assert float(hot[0]) > float(off[0]), (
        "w_spell is not reaching spell_value_shaping -- the dead-code bug is back")

    none_cast, prev2 = shaping_stats(fireball_killed=0.0)
    none_cast["fireball_elixir_spent"] = np.zeros(1, dtype=np.float32)
    assert float(off[0]) == pytest.approx(
        float(train_shaping.compute_shaping(none_cast, prev2, w_spell=0.0)[0]))


# --- the advisor target ----------------------------------------------------

def _clump_obs(card=41, x=6.0, y=12.0):
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    env.inject_enemy(card, x, y)
    env.step(4, 0.0, 0.0, 1)
    return np.asarray(env.get_observation_for_team(0), dtype=np.float32)


def test_advisor_has_nothing_to_say_on_an_empty_board(net, fresh_obs):
    """THE GATE, and the reason this is not just distillation-in-the-loop.

    With no threat the advisor still returns a cell -- the defensive pocket for
    a building, an arbitrary tie-broken lane for the Giant. Training on those
    teaches a CONSTANT, which is the exact failure this workstream exists to
    undo. `target_logits_for` must decline instead.
    """
    _env, obs = fresh_obs
    obs = np.asarray(obs, dtype=np.float32)
    for cid in AT.ADVISOR_CARDS:
        legal = net._placement_legal[cid].numpy().astype(bool)
        assert AT.target_logits_for(obs, cid, legal) is None, cid


def test_advisor_target_never_puts_mass_on_an_illegal_cell(net):
    """-inf on every illegal cell, for every card.

    A finite floor would leave real probability mass on a cell the engine
    refuses. The converse does NOT hold and must not be asserted: a "cell"-kind
    rule (the Giant) is a delta, so it is legitimately -inf on legal cells too.
    Only the map-backed kinds are finite across the legal set.
    """
    obs = _clump_obs()
    spoke = 0
    for cid, kind in AT.ADVISOR_CARDS.items():
        legal = net._placement_legal[cid].numpy().astype(bool)
        t = AT.target_logits_for(obs, cid, legal)
        if t is None:
            # A gated rule declining to speak is the DESIGNED behaviour, not a
            # failure: the win condition refuses to commit into an active push,
            # and _clump_obs() is exactly that board. Asserting `is not None`
            # for every card would forbid the gate this file elsewhere calls
            # load-bearing.
            continue
        spoke += 1
        assert t.shape == (net.placement_cells,)
        assert np.all(np.isneginf(t[~legal])), cid
        assert np.isfinite(t).any(), cid
        if kind in ("building", "spell"):
            assert np.all(np.isfinite(t[legal])), cid
    # ...but "every rule declined" would pass vacuously, so require that the
    # board still produced at least one target.
    assert spoke > 0, "no advisor spoke on a board with a live threat"


def test_advisor_spell_target_peaks_where_the_advisor_aims(net):
    """The distilled surface and the played cell must be the same object."""
    obs = _clump_obs()
    legal = net._placement_legal[tactics.FIREBALL_ID].numpy().astype(bool)
    t = AT.target_logits_for(obs, tactics.FIREBALL_ID, legal)
    peak = int(np.argmax(t))
    ax, ay, val = tactics.best_spell_cell(obs, legal=legal)
    assert val > 0.0
    assert peak == int(ay) * tactics.BOARD_W + int(ax)


def _hog_commit_obs():
    """A quiet board where the win-condition gate OPENS.

    Needs all three conditions at once: our half clear, us solvent past the
    Cannon reserve, and the opponent not banked. Those cannot be arranged by
    STEPPING -- idling to 9 elixir also banks the opponent to 10 and closes the
    third condition -- so the elixir scalar is written directly on a fresh
    board, which keeps time (and therefore the opponent's estimate) at its
    opening value.

    Deliberately not env.set_elixir_for_team: that binding exists only in the
    current build, and python_ai/ holds the stale .pyd whenever the post-build
    copy was blocked by a running trainer.
    """
    env = E.ClashRoyaleEnv(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 3600)
    env.reset()
    # One enemy, deep on THEIR half. Needed because the rule declines when the
    # board is completely empty -- with no enemy anywhere the "weaker lane" is
    # undefined and the tiebreak would emit a fixed cell. Placed at y=25 so it
    # is visible to the lane read without counting as a threat on our half.
    env.inject(6, 13.0, 25.0, 1)
    env.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)   # inject lands on the tick
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32).copy()
    obs[tactics.SPATIAL] = 0.9      # ClashEnv stores elixir / 10
    return obs


def test_advisor_wincon_target_is_a_delta_and_its_kl_is_cross_entropy(net):
    """The Hog rule yields a CELL, not a surface, so its target is a delta.

    Encoding it as a delta keeps one code path for every card: masked_kl
    against a delta is exactly cross-entropy to that cell, so the trainers need
    no separate hard-label branch.
    """
    obs = _hog_commit_obs()
    legal = net._placement_legal[tactics.HOG_ID].numpy().astype(bool)
    t = AT.target_logits_for(obs, tactics.HOG_ID, legal)
    assert t is not None, "gate should be open on a quiet, solvent board"
    gx, gy, _ = tactics.best_hog_cell(obs, legal=legal)
    cell = int(gy) * tactics.BOARD_W + int(gx)
    assert int(np.argmax(t)) == cell
    assert int(np.isneginf(t).sum()) == len(t) - 1

    logits = torch.randn(1, net.placement_cells)
    logits[0, ~torch.tensor(legal)] = float("-inf")
    kl = masked_kl(logits, torch.tensor(t).unsqueeze(0))
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([cell]))
    assert float(kl) == pytest.approx(float(ce), abs=1e-5)


def test_advisor_standardization_matches_the_offline_harness(net):
    """One definition of "score map -> target logits", provable, not asserted.

    `advisor_target._standardize` is a numpy re-implementation of
    `prove_hires.soft_target_logits` rather than an import of it, deliberately:
    importing the offline harness would pull `expert_iteration` into the
    trainers' rollout path. That is exactly the second-copy drift CLAUDE.md
    forbids, so it is pinned here instead of trusted.
    """
    from python_ai.eval.prove_hires import soft_target_logits

    obs = _clump_obs()
    legal = net._placement_legal[tactics.FIREBALL_ID].numpy().astype(bool)
    flat = tactics.spell_catch_map(obs).reshape(-1).astype(np.float32).copy()
    flat[~legal] = float("-inf")

    mine = AT._standardize(flat, legal, 0.25)
    theirs = soft_target_logits(torch.tensor(flat).unsqueeze(0), 0.25)[0].numpy()
    assert np.array_equal(np.isneginf(mine), np.isneginf(theirs))
    np.testing.assert_allclose(mine[legal], theirs[legal], rtol=1e-5, atol=1e-5)


def test_cannon_target_stays_broader_than_the_spell_target(net):
    """The Cannon's argmax is row-major noise off a tie plateau (handoff 2.3).

    So its target must keep the plateau's mass spread rather than committing to
    one arbitrary cell. Measured there: temperature cannot push the Cannon
    target below ~66% of maximum entropy while Fireball reaches ~41%. This pins
    the ORDERING, which is a property of the two surfaces rather than of the
    temperature.
    """
    obs = _clump_obs(card=2, x=9.0, y=13.0)      # a Giant: one big body, real threat
    ents = {}
    for cid in (tactics.CANNON_ID, tactics.FIREBALL_ID):
        legal = net._placement_legal[cid].numpy().astype(bool)
        t = AT.target_logits_for(obs, cid, legal)
        if t is None:
            pytest.skip(f"advisor declined card {cid} on this fixture")
        p = torch.softmax(torch.tensor(t), -1)
        ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        ents[cid] = ent / float(np.log(int(legal.sum())))
    assert ents[tactics.CANNON_ID] > ents[tactics.FIREBALL_ID], ents


# --- the coverage loss itself ----------------------------------------------

def test_elementwise_kl_averages_to_the_scalar_one():
    """masked_kl_elementwise must be masked_kl without the mean, exactly."""
    torch.manual_seed(3)
    new = torch.randn(7, 40)
    tgt = torch.randn(7, 40)
    new[:, 5:9] = float("-inf")
    tgt[:, 5:9] = float("-inf")
    assert float(AT.masked_kl_elementwise(new, tgt).mean()) == pytest.approx(
        float(masked_kl(new, tgt)), abs=1e-6)


def _coverage_fixture(n=6, cells=612):
    torch.manual_seed(4)
    logits = torch.randn(n, cells, requires_grad=True)
    targets = torch.randn(n, cells)
    decision = torch.ones(n)
    return logits, targets, decision


def test_with_no_advisor_target_coverage_is_exactly_the_old_entropy_bonus():
    """The fallback path must be the term it replaces, to the bit.

    Whatever else changes, a state the advisor declines has to behave the way
    v1.2.0 behaved, or the control arm of the A/B is not a control.
    """
    logits, targets, decision = _coverage_fixture()
    has = torch.zeros(len(decision))
    delta, ent_frac, kl, n = AT.coverage_terms(
        logits, targets, has, decision, PLACEMENT_COVERAGE_COEF, np.log(612))
    expected = float(Categorical(logits=logits).entropy().mean().detach()) / np.log(612)
    assert float(ent_frac) == pytest.approx(expected, abs=1e-6)
    assert float(delta) == pytest.approx(
        -PLACEMENT_COVERAGE_COEF * expected, abs=1e-6)
    assert float(kl) == 0.0 and float(n) == 0.0


def test_a_row_gets_the_target_or_the_entropy_bonus_but_never_both():
    """THE RESOLUTION. Entropy flattens, KL concentrates; one row cannot want
    both. Rows split cleanly, and each denominator counts only its own rows."""
    logits, targets, decision = _coverage_fixture(n=4)
    has = torch.tensor([1.0, 1.0, 0.0, 0.0])
    _, ent_frac, kl, n = AT.coverage_terms(
        logits, targets, has, decision, PLACEMENT_COVERAGE_COEF, np.log(612))
    assert float(n) == 2.0

    ent_all = Categorical(logits=logits).entropy()
    assert float(ent_frac) == pytest.approx(
        float(ent_all[2:].mean() / np.log(612)), abs=1e-6), (
        "the entropy bonus leaked onto rows that carry an advisor target")
    kl_all = AT.masked_kl_elementwise(logits, targets)
    assert float(kl) == pytest.approx(float(kl_all[:2].mean()), abs=1e-6)


def test_the_advisor_term_never_reaches_the_chosen_action(fixture_net=None):
    """Same property the entropy coverage term has, and for the same reason.

    The coverage pass scores a card that was AFFORDABLE, not the one that was
    CHOSEN, so nothing it does may reach the chosen action's log-prob or the
    critic. If it did, it would silently corrupt the quantity PPO clips.
    """
    net, obs_seq, feats_seq, embeds_seq, spatial_seq, card_mask, resets, hidden, hand = chunk_fixture()
    chosen = torch.zeros(L, B, dtype=torch.long)
    cover = torch.ones(L, B, dtype=torch.long)

    cl, pl, values, aux, _, extra = net.forward_sequence(
        feats_seq, embeds_seq, spatial_seq, obs_seq, card_mask, chosen, resets,
        hidden, extra_card_idx_seq=cover)

    targets = torch.randn(L, B, net.placement_cells)
    delta, _, _, _ = AT.coverage_terms(
        extra, targets, torch.ones(L, B), torch.ones(L, B),
        PLACEMENT_COVERAGE_COEF, np.log(net.placement_cells))

    for name, tensor in (("card logits", cl), ("chosen placement", pl),
                         ("values", values), ("aux", aux)):
        g = torch.autograd.grad(delta, tensor, retain_graph=True,
                                allow_unused=True)[0]
        assert g is None or float(g.abs().sum()) == 0.0, (
            f"the advisor coverage term leaked gradient into {name}")


def test_one_step_of_the_advisor_term_moves_the_head_toward_the_advisor():
    """The end-to-end claim: this term is a TARGET, not just noise.

    Optimizing the coverage loss alone must reduce the distance between the
    head's argmax cell and the advisor's own cell. Entropy cannot do this by
    construction -- it is a marginal objective with no opinion about which cell
    is right in which state -- so this is the property that separates the fix
    from the measured-insufficient version it replaces.
    """
    from python_ai.models.net import MicroRoyaleNet

    torch.manual_seed(11)
    np.random.seed(11)
    n = MicroRoyaleNet(num_ability_slots=0)
    legal_table = AT.build_legal_table(n)
    cid = tactics.FIREBALL_ID

    # The opening hand is drawn by an unseeded mt19937 the engine will not let
    # us set (CLAUDE.md, "Driving the simulator from outside is awkward"), so
    # Fireball is in only ~half of the 4-slot hands and which states survive is
    # not reproducible. Draw until there are enough rather than over-generating
    # a fixed list and hoping -- the fixed-list version passed alone and failed
    # inside the full suite, because the global RNG position differs.
    kept = []
    for i in range(60):
        o = _clump_obs(x=float(3 + (i % 13)), y=12.0)
        if bool((n.hand_card_ids(torch.tensor(o).unsqueeze(0)) == cid).any()):
            kept.append(o)
        if len(kept) == 8:
            break
    assert len(kept) >= 4, f"only {len(kept)} hands held Fireball in 60 draws"
    obs_np = np.stack(kept)
    rows = obs_np.shape[0]

    tgt_np, has_np = AT.targets_for_batch(obs_np, np.full(rows, cid), legal_table)
    assert has_np.all(), "fixture must give the advisor something to say"

    ob = torch.tensor(obs_np)
    targets = torch.tensor(tgt_np)
    hand_ids = n.hand_card_ids(ob)
    slot = (hand_ids == cid).float().argmax(dim=1)
    assert all(int(hand_ids[i, slot[i]]) == cid for i in range(rows))

    zeros = (torch.zeros(rows, 256), torch.zeros(rows, 256))

    def mean_distance():
        with torch.no_grad():
            feats, embeds, sp = n.extract_features(ob)
            hx, _ = n.lstm(feats, zeros)
            got = n.placement_given_card(hx, embeds, slot, ob, sp).argmax(-1)
            want = targets.argmax(-1)
            return float(torch.maximum((got % 18 - want % 18).abs(),
                                       (got // 18 - want // 18).abs()).float().mean())

    before = mean_distance()
    opt = torch.optim.Adam(n.parameters(), lr=3e-3)
    for _ in range(30):
        feats, embeds, sp = n.extract_features(ob)
        hx, _ = n.lstm(feats, zeros)
        logits = n.placement_given_card(hx, embeds, slot, ob, sp)
        delta, _, _, _ = AT.coverage_terms(
            logits, targets, torch.ones(rows), torch.ones(rows),
            PLACEMENT_COVERAGE_COEF, np.log(n.placement_cells))
        opt.zero_grad(set_to_none=True)
        delta.backward()
        opt.step()
    after = mean_distance()

    print(f"\n  advisor-target pull: mean cell distance {before:.2f} -> {after:.2f} "
          f"({rows} states)")
    # A fresh head's map is near-uniform, so `before` is a long way out; the
    # <= 1.0 branch only exists so a lucky init cannot fail a test about
    # LEARNING. Board diagonal is 34 tiles, so 1.0 is already at the target.
    if before <= 1.0:
        assert after <= before, f"the target pushed the head away ({before} -> {after})"
    else:
        assert after < before, (
            f"the advisor target did not move the head ({before} -> {after})")
