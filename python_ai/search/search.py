"""Decision-time lookahead: roll candidate actions forward, pick by the critic.

MEASURED, and it is the largest single inference-time win this project has:
160 paired trials at 1.5x opponent elixir, greedy 0.625 vs 0.944 with 1-ply
search, delta +0.319 [95% CI +0.237, +0.401], exact McNemar p = 5.6e-12, at 2.2x
wall clock. Search overrode the greedy action on only ~1 decision in 7.

READ THAT DEVIATION RATE FIRST, because it is the interpretation. The scorer is
this same network's OWN critic, so a +32-point swing off 13.8% of decisions says
the VALUE head is substantially better than the ACTION head is at exploiting it
-- which is precisely the condition under which expert iteration pays. Distilling
it back is worth +0.045 (`trainers/expert_iteration.py`), i.e. about a seventh;
search at inference remains ~7x more valuable than distilling it.

WHY SIMULATION IS FREE HERE. A snapshot costs 0.033 ms and a 10-tick step
0.027 ms, against ~50 ms for the single network forward that scores a batch of
candidates. Simulation is not the budget; SCORING is, which is why every
candidate is evaluated in ONE batched forward rather than K sequential ones.

MOVED OUT OF `eval/search_ab_test.py` on 2026-08-20. Six modules imported
`_build_candidates`, `_search_action`, `_policy_head` and `_greedy_from_logits`
from that A/B harness -- underscore-private names, reached across module
boundaries, from a script whose `main()` runs a whole experiment. The functions
are the reusable half and now live here under public names; the harness kept
its argparse.
"""
import os

import numpy as np
import torch

import clash_royale_env
from python_ai.engine_constants import BOARD_H, BOARD_W
from python_ai.models.policy_io import LSTM_HIDDEN
from python_ai.rewards.weights import DRAW_PENALTY

HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
#: The no-op arm of the card head is the column past the hand.
NOOP = HAND_SIZE


@torch.no_grad()
def policy_head(net, obs_t, hidden):
    """One network forward. Returns the pieces both arms need.

    Deliberately still returns FIVE values and calls the plain
    extract_features. Six callers across five files unpack exactly five, and
    models/net.py's own extract_features docstring records the same reasoning for
    keeping that wrapper rather than widening its signature. The hires map is
    recovered locally in build_candidates instead -- see there.
    """
    card_mask = net.affordability_mask(obs_t)
    features, card_embeds, spatial_map = net.extract_features(obs_t)
    card_logits, _, _, value, hidden_next = net.step_lstm_and_card(features, hidden, card_mask)
    return card_logits, card_embeds, spatial_map, value, hidden_next


@torch.no_grad()
def greedy_from_logits(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next):
    """argmax over MASKED card logits then argmax cell -- identical to
    trainers.league.evaluate_against_roster, so the control arm IS this
    project's own definition of 'the policy playing'."""
    card_idx_t = card_logits.argmax(dim=-1)
    place_logits = net.placement_given_card(hidden_next[0], card_embeds, card_idx_t, obs_t, spatial_map)
    cell = place_logits.argmax(dim=-1)
    x_t, y_t = net.cell_to_xy(cell)
    return int(card_idx_t.item()), float(x_t.item()), float(y_t.item()), place_logits


#: Placement top-1 probability below which a card's head is treated as having
#: nothing useful for search to RANK, so the proposals are widened instead.
#:
#: MEASURED SITING, not chosen. Mean top-1 per deck card on the ep-32,484
#: policy: Hog 0.7654, Musketeer 0.3939, Skeletons 0.3527 | Ice Golem 0.1259,
#: Fireball 0.1074, Ice Spirit 0.0987, Cannon 0.0970, The Log 0.0288. The
#: largest gap inside the diffuse region is 0.227, and 0.25 sits in it -- 1.4x
#: below Skeletons and 2.0x above Ice Golem.
#:
#: NOT A DEAD-CARD DETECTOR. Ice Golem and Ice Spirit are two of the most-played
#: cards in the deck and their heads are as flat as the Cannon's, because for a
#: cheap cycle card placement genuinely matters less. This selects "search has
#: nothing useful to rank here", which is the question search needs answered.
#: Widening a card whose placement does not matter costs compute, not accuracy.
WIDE_PROPOSAL_TOP1 = float(os.environ.get("CLASH_WIDE_PROPOSAL_TOP1", 0.25))

#: Board stride for the widened sweep. (2, 2) reproduces the grid the measured
#: arm used -- ~56 cells before the legality filter, against k_cells=2.
WIDE_PROPOSAL_STRIDE = (2, 2)


def propose_cells(place_logits, k_cells,
                  top1=None, stride=WIDE_PROPOSAL_STRIDE):
    """Cell indices for one card: the head's top-k, or a spread if it is flat.

    `place_logits` is one row, already -inf on illegal cells.

    A SHARP head is left exactly as it was -- search ranking its own top-k is
    the right operation and the Hog measures 0.7654 top-1. A DIFFUSE head's
    top-k is near-arbitrary, and search over it gained +58 tower HP against the
    raw argmax while a wide sweep gained +994 (81% of the engine oracle). So the
    widened set is drawn by BOARD POSITION, not by probability: drawing more
    cells from a flat distribution just yields more of the same noise.
    """
    top1 = WIDE_PROPOSAL_TOP1 if top1 is None else top1
    legal = torch.isfinite(place_logits)
    if not bool(legal.any()):
        return []

    probs = torch.softmax(place_logits, dim=-1)
    if float(probs.max()) >= top1:
        return [int(c) for c in torch.topk(place_logits, k_cells).indices
                if torch.isfinite(place_logits[int(c)])]

    sx, sy = stride
    out = []
    for y in range(0, BOARD_H, sy):
        for x in range(0, BOARD_W, sx):
            idx = y * BOARD_W + x
            if idx < place_logits.numel() and bool(legal[idx]):
                out.append(idx)
    # The argmax always survives: search must never be able to do WORSE than
    # the head it is helping, which is the same reason greedy is candidate 0.
    best = int(torch.topk(place_logits, 1).indices[0])
    if best not in out and bool(legal[best]):
        out.append(best)
    return out


@torch.no_grad()
def build_candidates(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                      greedy, k_cards, k_cells):
    """Greedy action first, then alternatives from the top of each head.

    Greedy is candidate 0 deliberately: search then deviates only when the
    critic prefers something else, so it can never be worse than the policy
    BY THE CRITIC'S OWN MEASURE. Any loss it takes is the critic being wrong,
    which is a different and more interesting failure than search being wrong.
    """
    # Canonicalise a no-op greedy to (NOOP, 0, 0) before seeding `seen`.
    # ClashEnv::step ignores placement whenever cardIndex >= HAND_SIZE, so
    # (NOOP, gx, gy) and (NOOP, 0, 0) are THE SAME ACTION -- but as tuples they
    # differ, so without this the no-op is emitted twice and the two copies roll
    # out identically and score identically.
    #
    # That cost nothing for the win-rate A/B (the duplicate always ties, and
    # argmax returns the lower index, which is greedy). It is fatal for
    # distribution distillation: with the policy no-oping ~80-90% of the time,
    # the majority of rows ended up with exactly two candidates that were the
    # same action, giving a target with a measured median value spread of
    # EXACTLY 0.0 -- a uniform distribution over one action, carrying no signal
    # while still contributing gradient.
    if greedy[0] >= HAND_SIZE:
        greedy = (NOOP, 0.0, 0.0)
    cands = [greedy]
    seen = {greedy}

    # Build the pre-pool trunk activation ONCE for this decision. Without it
    # every placement_given_card below rebuilds it from obs internally
    # (models/net.py: `hires_map` defaults to None -> recomputed), so a k_cards=3
    # decision ran cnn_trunk[:2] up to four times on identical input -- on the
    # deadline-bounded path search_action_deadline exists to budget. Passing it
    # is the seam models/net.py documents for hot callers, and the two routes are
    # pinned bit-identical by
    # test_python_ai.test_recomputed_hires_equals_the_passed_one.
    hires_map = net.hires_features(obs_t)

    n_cards = card_logits.shape[-1]
    top_cards = torch.topk(card_logits[0], min(k_cards, n_cards)).indices
    for c in top_cards:
        if not torch.isfinite(card_logits[0, c]):
            continue  # masked: unaffordable, and playCard would drop it silently
        card_idx = int(c.item())
        if card_idx >= HAND_SIZE:
            key = (NOOP, 0.0, 0.0)  # placement is meaningless for the no-op
            if key not in seen:
                seen.add(key)
                cands.append(key)
            continue
        c_t = c.view(1)
        place_logits = net.placement_given_card(hidden_next[0], card_embeds, c_t, obs_t,
                                                spatial_map, hires_map=hires_map)
        # Widened where the head is flat -- see propose_cells. Illegal cells
        # are already filtered there, so the guard below is now belt-and-braces.
        top_cells = propose_cells(place_logits[0], k_cells)
        for cell in top_cells:
            cell = torch.tensor(cell)
            if not torch.isfinite(place_logits[0, cell]):
                continue  # masked: illegal placement for this card class
            x_t, y_t = net.cell_to_xy(cell.view(1))
            key = (card_idx, float(x_t.item()), float(y_t.item()))
            if key not in seen:
                seen.add(key)
                cands.append(key)
    return cands


def terminal_score(reward, weight):
    """What a rollout that ENDED is worth, on the CRITIC's scale.

    A finished game is not a position to be valued -- the critic's estimate of
    one is meaningless -- so the real outcome is used, weighted to dominate any
    bootstrapped value.

    A DRAW IS PRICED LIKE A LOSS, because that is what the policy is trained on.
    This used to be `reward * weight`, and the engine pays ~0 for a draw, so a
    drawn line scored 0.0 against a lost line's -10.0 -- a ten-point advantage
    for running the clock out. The training reward gives both exactly -1.0
    (a loss is raw -1.0; a draw is raw 0.0 minus DRAW_PENALTY 1.0), and
    DRAW_PENALTY exists precisely to stop a timeout being the safe outcome.

    A search that maximises a different objective from the one the policy is
    trained on is not a policy-improvement operator, which is the entire claim
    being made for it. The horizon is 4-12 decision steps against a ~360 s
    match, so this only bites when a rollout can actually reach the clock --
    i.e. in the endgame, which is exactly where stalling is tempting.
    """
    if abs(reward) > 0.5:
        return reward * weight
    return -DRAW_PENALTY * weight


@torch.no_grad()
def search_action(net, env, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                   greedy, cfg, device, return_details=False):
    """Roll every candidate forward on its own snapshot, score, pick the best.

    `return_details` appends the full (candidates, scores) pair as a 4th return
    value. Off by default, so every existing caller keeps the same 3-tuple and
    the arm that measured +0.319 is untouched. It exists for expert-iteration
    distillation: the argmax alone discards how MUCH better the winner was, and
    that margin is the only thing distinguishing "waiting is marginally better"
    from "playing here is a blunder".
    """
    cands = build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if len(cands) == 1:
        # One candidate carries no preference information at all -- there is
        # nothing to rank, so distillation must skip these rows rather than
        # train on a degenerate one-hot.
        if return_details:
            return greedy, False, 1, None
        return greedy, False, 1

    final_obs = []
    terminal = []
    for card_idx, x, y in cands:
        sim = env.snapshot()
        result = sim.step(card_idx, x, y)
        done = result.done
        for _ in range(cfg.horizon - 1):
            if done:
                break
            result = sim.step(NOOP, 0.0, 0.0)
            done = result.done
        final_obs.append(sim.get_observation_for_team(0))
        terminal.append((done, float(result.reward)))

    batch = torch.tensor(np.asarray(final_obs, dtype=np.float32), device=device)
    feats, _, _ = net.extract_features(batch)
    # Candidates are scored from the SAME pre-rollout hidden state. That is an
    # approximation -- the LSTM has not consumed the rollout's intervening
    # observations -- but it is common to every candidate and costs one batched
    # forward instead of horizon x K sequential ones. Advancing the recurrence
    # per candidate would be more faithful and roughly `horizon` times dearer.
    n = len(cands)
    hx = hidden_next[0].expand(n, LSTM_HIDDEN).contiguous()
    cx = hidden_next[1].expand(n, LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()

    # A rollout that ENDED is scored by its real outcome -- see terminal_score,
    # which also prices a draw like a loss, as the training reward does.
    for i, (done, reward) in enumerate(terminal):
        if done:
            scores[i] = terminal_score(reward, cfg.terminal_weight)

    best = int(scores.argmax().item())
    if return_details:
        return cands[best], best != 0, len(cands), (cands, scores.detach().cpu().numpy())
    return cands[best], best != 0, len(cands)


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------

@torch.no_grad()
def play_episode(net, env, device, use_search, cfg):
    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))
    obs = env.get_observation_for_team(0)
    reward, steps, deviations, cand_total = 0.0, 0, 0, 0
    done = False

    while not done and steps < cfg.max_steps:
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        card_logits, card_embeds, spatial_map, _, hidden_next = policy_head(net, obs_t, hidden)
        greedy_card, gx, gy, _ = greedy_from_logits(
            net, obs_t, card_logits, card_embeds, spatial_map, hidden_next)
        greedy = (greedy_card, gx, gy)

        if use_search:
            action, deviated, n_cands = search_action(
                net, env, obs_t, card_logits, card_embeds, spatial_map,
                hidden_next, greedy, cfg, device)
            deviations += int(deviated)
            cand_total += n_cands
        else:
            action = greedy

        hidden = hidden_next
        result = env.step(action[0], action[1], action[2])
        obs, reward, done = result.observation, float(result.reward), result.done
        steps += 1

    return reward, steps, deviations, cand_total



def outcome_score(reward):
    """win 1.0 / draw 0.5 / loss 0.0 -- the same convention as the Elo roster."""
    if reward > 0.5:
        return 1.0
    if reward < -0.5:
        return 0.0
    return 0.5
