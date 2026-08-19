"""Deadline-bounded ("anytime") decision-time search, for real-time play.

WHY A DEADLINE AND NOT A TUNED (h, K)
-------------------------------------
The offline search picks horizon/K once and pays whatever they cost, because
nothing is waiting. Live, the emulator does not pause: what matters is the
per-decision TAIL, and the tail on a shared machine is set by SCHEDULING, not
by the configuration. Measured on this box while pipeline-2 training ran, the
end-to-end latency curve was non-monotonic -- h=12 with K=4 measured a FASTER
p99 (67.7 ms) than h=2 with K=4 (393.9 ms), and greedy alone spiked to 260 ms.
A latency curve cannot fall with depth; that is contention, not cost.

So a precomputed budget is the wrong instrument. This module takes a wall-clock
DEADLINE instead and returns the best action it managed to evaluate, degrading
to greedy rather than overrunning. That is robust to contention, to a different
deployment machine, and to future changes in net size.

WHY THE COMPONENT COSTS MAKE THIS CHEAP
---------------------------------------
Measured, 3000 reps each, same contended machine:

    snapshot()                0.021 ms p50   0.078 ms p99
    step(10 ticks)            0.020 ms p50   0.145 ms p99   <- one horizon unit
    extract_features batch=1  4.33  ms p50   27.5   ms p99
    extract_features batch=7  7.04  ms p50   18.5   ms p99

SIMULATION IS FREE; THE NETWORK IS THE ENTIRE BUDGET. A horizon-12 rollout of
7 candidates costs 7 x 12 x 0.020 = 1.7 ms of engine time. That is why DEPTH is
nearly free and WIDTH is not -- width adds rows to the scoring forward, depth
adds only engine ticks. shipping.py reached the same conclusion from wall-clock
per episode; these numbers say it in absolute terms.

The practical consequence: h=12 is NOT too slow for live. The budget is spent
on two-to-three network passes regardless of horizon.

GUARANTEE
---------
Greedy is computed FIRST and is always available, so there is always a legal
action to return. The deadline is checked
  * before each candidate rollout   -- overrun bounded by one rollout
    (horizon x 0.02 ms, i.e. ~0.25 ms at h=12), and
  * before the scoring forward       -- which is skipped entirely if the
    remaining time cannot cover it, since a forward cannot be interrupted.
`reserve_ms` is that reservation. The search therefore cannot block: worst case
it does slightly more engine stepping than it needed and returns greedy.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from python_ai.eval.search_ab_test import _build_candidates, LSTM_HIDDEN

NOOP_PLACEHOLDER = None


class SearchStats:
    """What the last call actually managed to do -- for live telemetry."""

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

    env         must expose snapshot()/step()/get_observation_for_team(0) --
                the real ClashRoyaleEnv, holding the TRUE state. Passing a
                reconstructed board makes every score a score of a fabricated
                position; see this module's companion note in the report.
    greedy      (card_idx, x, y) -- already computed, always the fallback.
    legal_filter(card_idx, x, y) -> bool, optional. Live only: the actuator
                cannot tap every engine-legal cell (mvp_loop's _tappable_cells),
                so a candidate the bot physically cannot reach must be dropped
                BEFORE it is rolled forward, not after it is chosen.
    reserve_ms  time held back for the scoring forward, which cannot be
                interrupted once started.
    """
    st = stats if stats is not None else SearchStats()
    t0 = time.perf_counter()

    def elapsed():
        return (time.perf_counter() - t0) * 1000.0

    if noop_slot is None:
        noop_slot = net.hand_size

    cands = _build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if legal_filter is not None:
        # Greedy stays at index 0 unconditionally -- it is the fallback and the
        # caller has already validated it. Filtering it out here would make
        # "search deviated" meaningless.
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
    # Fewer than two evaluated candidates carries no preference information --
    # ranking one candidate against nothing is not a comparison.
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
            # A finished rollout is not a position to value -- use the outcome,
            # weighted to dominate any bootstrapped critic estimate.
            scores[i] = reward * cfg.terminal_weight

    best = int(scores.argmax().item())
    st.scored = True
    st.deviated = best != 0
    st.elapsed_ms = elapsed()
    return kept[best], st
