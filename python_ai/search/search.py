"""Decision-time lookahead: roll candidate actions forward, pick by the critic.

Against the C++ heuristic at 1.5x elixir, 1-ply search took the greedy policy
from 0.625 to 0.944 (160 paired trials) while overriding only ~1 decision in 7:
the value head can exploit states the action head cannot. Distilling it back
recovers about a seventh of that (`trainers/expert_iteration.py`). Against the
UtilityTeacher it measured negative; see `rollout`.

Simulation is nearly free (snapshot 0.033 ms, 10-tick step 0.027 ms); scoring
is the budget, so all candidates are evaluated in one batched forward.
"""
import os

import numpy as np
import torch

import clash_royale_env
from python_ai.engine_constants import BOARD_H, BOARD_W
from python_ai.models.policy_io import LSTM_HIDDEN
from python_ai.rewards.weights import DRAW_PENALTY
# Defined in config.py, which must stay cheap to import.
from python_ai.search.config import (WIDE_PROPOSAL_MAX_CELLS,
                                     WIDE_PROPOSAL_STRIDE,
                                     WIDE_PROPOSAL_TOP1)

HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
#: The no-op arm of the card head is the column past the hand.
NOOP = HAND_SIZE


@torch.no_grad()
def policy_head(net, obs_t, hidden):
    """One network forward, returning the pieces both arms need.

    Returns five values from the plain extract_features, as its callers expect;
    build_candidates recovers the hi-res map itself.
    """
    card_mask = net.affordability_mask(obs_t)
    features, card_embeds, spatial_map = net.extract_features(obs_t)
    card_logits, _, _, value, hidden_next = net.step_lstm_and_card(features, hidden, card_mask)
    return card_logits, card_embeds, spatial_map, value, hidden_next


@torch.no_grad()
def greedy_from_logits(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next):
    """argmax over masked card logits, then argmax cell: the same definition of
    "the policy playing" as trainers.league.evaluate_against_roster.
    """
    card_idx_t = card_logits.argmax(dim=-1)
    place_logits = net.placement_given_card(hidden_next[0], card_embeds, card_idx_t, obs_t, spatial_map)
    cell = place_logits.argmax(dim=-1)
    x_t, y_t = net.cell_to_xy(cell)
    return int(card_idx_t.item()), float(x_t.item()), float(y_t.item()), place_logits


def propose_cells(place_logits, k_cells,
                  top1=None, stride=WIDE_PROPOSAL_STRIDE):
    """Cell indices for one card: the head's top-k, or a spread if it is flat.

    `place_logits` is one row, already -inf on illegal cells. A sharp head's
    top-k is what search should rank; a diffuse head's top-k is near-arbitrary,
    so the widened set is drawn by board position rather than probability.
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

    # The argmax always survives, so search can never do worse than the head;
    # added before the cap, or the cap could be exceeded by one.
    best = int(torch.topk(place_logits, 1).indices[0])
    if best not in out and bool(legal[best]):
        out.append(best)

    # Cap by even subsampling, never truncation, which would keep only the top
    # of the board.
    if len(out) > WIDE_PROPOSAL_MAX_CELLS:
        step = len(out) / float(WIDE_PROPOSAL_MAX_CELLS)
        kept = [out[int(i * step)] for i in range(WIDE_PROPOSAL_MAX_CELLS)]
        if best not in kept:
            kept[-1] = best      # the cap never evicts the head's own pick
        out = kept
    return out


@torch.no_grad()
def build_candidates(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                      greedy, k_cards, k_cells):
    """Greedy action first, then alternatives from the top of each head.

    With greedy as candidate 0, search deviates only when the critic prefers
    something else, so it cannot be worse than the policy by the critic's own
    measure.
    """
    # Canonicalise a no-op greedy to (NOOP, 0, 0): ClashEnv::step ignores
    # placement for the no-op, so the two are the same action, and emitting
    # both would give distribution distillation targets with zero spread.
    if greedy[0] >= HAND_SIZE:
        greedy = (NOOP, 0.0, 0.0)
    cands = [greedy]
    seen = {greedy}

    # Build the pre-pool trunk activation once per decision instead of once per
    # placement call (test_recomputed_hires_equals_the_passed_one pins them
    # equal).
    hires_map = net.hires_features(obs_t)

    n_cards = card_logits.shape[-1]
    top_cards = torch.topk(card_logits[0], min(k_cards, n_cards)).indices
    for c in top_cards:
        if not torch.isfinite(card_logits[0, c]):
            continue  # masked: unaffordable
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
        # Widened where the head is flat; see propose_cells.
        top_cells = propose_cells(place_logits[0], k_cells)
        for cell in top_cells:
            cell = torch.tensor(cell)
            if not torch.isfinite(place_logits[0, cell]):
                continue  # masked: illegal placement for this card
            x_t, y_t = net.cell_to_xy(cell.view(1))
            key = (card_idx, float(x_t.item()), float(y_t.item()))
            if key not in seen:
                seen.add(key)
                cands.append(key)
    return cands


def terminal_score(reward, weight):
    """What a rollout that ended is worth, on the critic's scale.

    The real outcome, weighted to dominate any bootstrapped value. A draw is
    priced like a loss, as in the training reward (raw 0 minus DRAW_PENALTY),
    or search would prefer running out the clock to a loss.
    """
    if abs(reward) > 0.5:
        return reward * weight
    return -DRAW_PENALTY * weight



class _RolloutResult:
    """`step_self_play` in the shape `step` returns, so one rollout loop serves
    both opponent paths.
    """
    __slots__ = ("observation", "reward", "done")

    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done


def rollout(sim, card, x, y, horizon, opponent=None, skip_frames=10):
    """Play one candidate forward on `sim` and return the final step result.

    The opponent decides what search optimises against. `sim.step(...)` runs
    the C++ HeuristicOpponent, which is why search measured -0.3 to -0.5 win
    rate against the UtilityTeacher (tests/test_search_opponent_model.py).
    `opponent` is any object with `act(env, obs_own) -> (slot, x, y)`, like
    UtilityTeacher; None keeps the heuristic path bit-identical.

    Our side idles after the candidate action.
    """
    if opponent is None:
        result = sim.step(card, x, y, skip_frames)
        done = result.done
        for _ in range(horizon - 1):
            if done:
                break
            result = sim.step(NOOP, 0.0, 0.0, skip_frames)
            done = result.done
        return result

    our_card, our_x, our_y = card, x, y
    result = None
    for _ in range(horizon):
        obs1 = np.asarray(sim.get_observation_for_team(1), dtype=np.float32)
        slot, ox, oy = opponent.act(sim, obs1)
        raw = sim.step_self_play(our_card, our_x, our_y, slot, ox, oy,
                                 skip_frames, False, False, False, False)
        result = _RolloutResult(raw.observation0, float(raw.reward0), raw.done)
        if result.done:
            break
        our_card, our_x, our_y = NOOP, 0.0, 0.0
    return result


@torch.no_grad()
def search_action(net, env, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                   greedy, cfg, device, return_details=False, opponent=None):
    """Roll every candidate forward on its own snapshot, score, pick the best.

    `return_details` appends (candidates, scores) as a 4th value, for
    expert-iteration distillation: the margin, not just the argmax, separates a
    marginal preference from a blunder.
    """
    cands = build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if len(cands) == 1:
        # One candidate carries no preference; distillation skips these rows.
        if return_details:
            return greedy, False, 1, None
        return greedy, False, 1

    final_obs = []
    terminal = []
    for card_idx, x, y in cands:
        sim = env.snapshot()
        # Reset per candidate: the opponent model carries state (cycle, pending
        # combo), and one rollout must not leave it for the next.
        if opponent is not None:
            opponent.reset()
        result = rollout(sim, card_idx, x, y, cfg.horizon, opponent)
        done = result.done
        final_obs.append(sim.get_observation_for_team(0))
        terminal.append((done, float(result.reward)))

    batch = torch.tensor(np.asarray(final_obs, dtype=np.float32), device=device)
    feats, _, _ = net.extract_features(batch)
    # Every candidate is scored from the same pre-rollout hidden state: an
    # approximation common to all of them, for one batched forward instead of
    # horizon x K.
    n = len(cands)
    hx = hidden_next[0].expand(n, LSTM_HIDDEN).contiguous()
    cx = hidden_next[1].expand(n, LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()

    # A finished rollout is scored by its real outcome; see terminal_score.
    for i, (done, reward) in enumerate(terminal):
        if done:
            scores[i] = terminal_score(reward, cfg.terminal_weight)

    best = int(scores.argmax().item())
    if return_details:
        return cands[best], best != 0, len(cands), (cands, scores.detach().cpu().numpy())
    return cands[best], best != 0, len(cands)


# --- episodes ---

@torch.no_grad()
def play_episode(net, env, device, use_search, cfg, opponent=None):
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
                hidden_next, greedy, cfg, device, opponent=opponent)
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
    """Win 1.0 / draw 0.5 / loss 0.0, the Elo roster's convention."""
    if reward > 0.5:
        return 1.0
    if reward < -0.5:
        return 0.0
    return 0.5
