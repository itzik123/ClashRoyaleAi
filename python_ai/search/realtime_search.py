"""Deadline-bounded ("anytime") decision-time search, for real-time play.

Live, what matters is the per-decision tail latency, and on a shared machine
that is set by scheduling, not by (horizon, K). So this takes a wall-clock
deadline and returns the best action it managed to evaluate, degrading to
greedy rather than overrunning.

The engine is nearly free (snapshot and 10-tick step ~0.02 ms at p50); the
network forwards are the whole budget, so a long horizon is affordable live.

Greedy is computed first, so a legal action is always available. The deadline
is checked before each rollout (overrun bounded by one rollout) and before the
scoring forward, which is skipped if `reserve_ms` would not cover it. The
search cannot block.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.search.search import build_candidates  # noqa: E402

NOOP_PLACEHOLDER = None


class SearchStats:
    """What the last call managed to do, for live telemetry."""

    __slots__ = ("elapsed_ms", "candidates", "evaluated", "aborted",
                 "scored", "deviated")

    def __init__(self):
        self.elapsed_ms = 0.0
        self.candidates = 0
        self.evaluated = 0
        self.aborted = False
        self.scored = False
        self.deviated = False

    def __repr__(self):
        return (f"<search {self.elapsed_ms:.1f}ms cands={self.candidates} "
                f"eval={self.evaluated} scored={self.scored} "
                f"aborted={self.aborted} deviated={self.deviated}>")


@torch.no_grad()
def search_action_deadline(net, env, obs_t, card_logits, card_embeds, spatial_map,
                           hidden_next, greedy, cfg, device, *,
                           deadline_ms=120.0, reserve_ms=35.0, noop_slot=None,
                           legal_filter=None, stats=None):
    """Best action within `deadline_ms`, falling back to `greedy`.

    env          must expose snapshot()/step()/get_observation_for_team(0) and
    hold the true state; a reconstructed board makes every score fictional.
    greedy       (card_idx, x, y), already computed; always the fallback.
    legal_filter (card_idx, x, y) -> bool, optional. Live only: the actuator
    cannot tap every engine-legal cell, so an unreachable candidate is dropped
    before it is rolled out.
    reserve_ms   time held back for the scoring forward, which cannot be
    interrupted.
    """
    st = stats if stats is not None else SearchStats()
    t0 = time.perf_counter()

    def elapsed():
        return (time.perf_counter() - t0) * 1000.0

    if noop_slot is None:
        noop_slot = net.hand_size

    cands = build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if legal_filter is not None:
        # Greedy stays at index 0 unconditionally: it is the fallback and
        # already validated.
        cands = [cands[0]] + [c for c in cands[1:] if legal_filter(*c)]
    st.candidates = len(cands)

    if len(cands) < 2:
        st.elapsed_ms = elapsed()
        return greedy, st

    final_obs, terminal, kept = [], [], []
    for cand in cands:
        if elapsed() + reserve_ms >= deadline_ms:
            st.aborted = True
            break
        card_idx, x, y = cand
        sim = env.snapshot()
        result = sim.step(card_idx, x, y)
        done = result.done
        for _ in range(cfg.horizon - 1):
            if done:
                break
            result = sim.step(noop_slot, 0.0, 0.0)
            done = result.done
        final_obs.append(sim.get_observation_for_team(0))
        terminal.append((done, float(result.reward)))
        kept.append(cand)

    st.evaluated = len(kept)
    # Fewer than two evaluated candidates is no comparison.
    if len(kept) < 2 or elapsed() + reserve_ms >= deadline_ms:
        st.elapsed_ms = elapsed()
        return greedy, st

    batch = torch.as_tensor(np.asarray(final_obs, dtype=np.float32), device=device)
    feats, _, _ = net.extract_features(batch)
    n = len(kept)
    hx = hidden_next[0].expand(n, LSTM_HIDDEN).contiguous()
    cx = hidden_next[1].expand(n, LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()
    for i, (done, reward) in enumerate(terminal):
        if done:
            # A finished rollout is scored by its outcome, weighted above any
            # critic bootstrap.
            scores[i] = reward * cfg.terminal_weight

    best = int(scores.argmax().item())
    st.scored = True
    st.deviated = best != 0
    st.elapsed_ms = elapsed()
    return kept[best], st
