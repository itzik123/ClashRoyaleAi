"""Head-to-head self-play between two nets, EITHER SIDE optionally using
decision-time search, with the sides swapped.

    python_ai/venv/Scripts/python.exe python_ai/net_h2h_search.py \
        --a model_weights_selfplay.pth --b model_weights_cured.pth \
        --b-search --n 150

WHY THIS EXISTS. `net_h2h.py` duels two nets greedy-vs-greedy and measured the
cured net at 0.6125 against v1.2.0. But the shipping configuration is not the
greedy policy -- decision-time search is worth +0.183 against the C++ heuristic
(measured 2026-08-15, n=60 paired, CI [+0.032, +0.334]), and the question that
decides whether we ship is whether the DEPLOYABLE agent beats the DEPLOYED
baseline. That is cured+search vs v1.2.0-greedy, which nothing measured before.

Run it both ways: `--b-search` alone gives the ship-vs-ship comparison, and
omitting it reproduces net_h2h.py's greedy-vs-greedy number as a control, so
the search contribution is visible rather than confounded with the weights.

TWO THINGS THE ROLLOUT DOES DELIBERATELY:

  * The OPPONENT NO-OPS inside a candidate rollout. `step_self_play` needs an
    action for both sides, and the honest options are "no-op" or "feed it the
    opponent's real move". The second is an information leak -- both sides
    genuinely decide from the same board simultaneously, so knowing their
    current move is knowledge the engine does not grant. No-op is an
    approximation, but it is the SAME approximation for every candidate, so the
    ranking it produces is fair.

  * TERMINAL POSITIONS ARE SCORED FROM SURVIVING TOWERS, not from
    `result.reward`. A finished game has no meaningful critic estimate, and
    this file does not depend on `step_self_play`'s reward convention matching
    `step`'s.

    Both the leaf value and the final duel verdict now go through
    `match_outcome`, which applies TimeoutRules' FULL rule -- tower count,
    then weakest surviving tower, then draw. They previously used tower count
    alone and called every equal-count finish a draw, which is not what the
    engine decides and which biased this file's own ship/no-ship number.
    Because the leaf value feeds candidate ranking, search behaviour changes
    across that fix: A/B figures measured before it are not comparable.

SIDES ARE SWAPPED and each pairing is played twice, for the reason net_h2h.py
records: a policy beating a bit-exact copy of itself measured 0.598 once purely
by side assignment, and reads 0.530 today.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
from policy_io import load_net  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from match_outcome import score_from_towers, terminal_value  # noqa: E402
from search_ab_test import (  # noqa: E402
    HAND_SIZE, LSTM_HIDDEN, NOOP, _build_candidates, _greedy_from_logits,
    _policy_head,
)

CE = E.ClashRoyaleEnv


class Cfg:
    def __init__(self, horizon=4, k_cards=3, k_cells=2, terminal_weight=10.0):
        self.horizon = horizon
        self.k_cards = k_cards
        self.k_cells = k_cells
        self.terminal_weight = terminal_weight


def _terminal_value(sim, team):
    """+1 / 0 / -1 from surviving towers, from `team`'s point of view.

    Delegates to match_outcome so the leaf value and the duel verdict can never
    disagree about what a win is -- see this module's docstring for why the old
    tower-count-only version was wrong and what it invalidates.
    """
    return terminal_value(sim, team)


@torch.no_grad()
def _search_action_selfplay(net, env, team, obs_t, card_logits, card_embeds,
                            spatial_map, hidden_next, greedy, cfg):
    """Roll every candidate forward on its own snapshot; pick the best by value.

    Greedy is candidate 0 (see _build_candidates), so search can only deviate
    when the critic prefers something else.
    """
    cands = _build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if len(cands) == 1:
        return greedy, False

    final_obs, terminal = [], []
    for card_idx, x, y in cands:
        sim = env.snapshot()
        if team == 0:
            r = sim.step_self_play(card_idx, x, y, NOOP, 0.0, 0.0, 10)
        else:
            r = sim.step_self_play(NOOP, 0.0, 0.0, card_idx, x, y, 10)
        done = r.done
        for _ in range(cfg.horizon - 1):
            if done:
                break
            r = sim.step_self_play(NOOP, 0.0, 0.0, NOOP, 0.0, 0.0, 10)
            done = r.done
        final_obs.append(sim.get_observation_for_team(team))
        terminal.append((done, _terminal_value(sim, team) if done else 0.0))

    batch = torch.tensor(np.asarray(final_obs, dtype=np.float32))
    feats, _, _ = net.extract_features(batch)
    n = len(cands)
    hx = hidden_next[0].expand(n, LSTM_HIDDEN).contiguous()
    cx = hidden_next[1].expand(n, LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()
    for i, (done, val) in enumerate(terminal):
        if done:
            scores[i] = val * cfg.terminal_weight

    best = int(scores.argmax().item())
    return cands[best], best != 0


@torch.no_grad()
def act(net, env, team, hid, use_search, cfg):
    obs_t = torch.tensor(np.asarray(env.get_observation_for_team(team),
                                    dtype=np.float32)).unsqueeze(0)
    card_logits, card_embeds, spatial_map, _, hid_next = _policy_head(net, obs_t, hid)
    gi, gx, gy, _ = _greedy_from_logits(net, obs_t, card_logits, card_embeds,
                                        spatial_map, hid_next)
    action, deviated = (gi, gx, gy), False
    if use_search:
        action, deviated = _search_action_selfplay(
            net, env, team, obs_t, card_logits, card_embeds, spatial_map,
            hid_next, (gi, gx, gy), cfg)
    return action, hid_next, deviated


def duel(net0, net1, env, s0, s1, cfg, max_steps=400):
    """net0 as team 0, net1 as team 1. Returns net0's score and deviation counts."""
    h0 = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    h1 = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    dev = [0, 0]
    steps = 0
    for _ in range(max_steps):
        (g0, x0, y0), h0, d0 = act(net0, env, 0, h0, s0, cfg)
        (g1, x1, y1), h1, d1 = act(net1, env, 1, h1, s1, cfg)
        dev[0] += int(d0)
        dev[1] += int(d1)
        steps += 1
        if env.step_self_play(g0, x0, y0, g1, x1, y1, 10).done:
            break
    return score_from_towers(env, 0), dev, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline net")
    ap.add_argument("--b", required=True, help="candidate net")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--a-search", action="store_true")
    ap.add_argument("--b-search", action="store_true")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--max-ticks", type=int, default=3600)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dev_t = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    cfg = Cfg(args.horizon, args.k_cards, args.k_cells)

    def _load(p):
        return load_net(p if os.path.isabs(p) else os.path.join(here, p), dev_t)

    A, B = _load(args.a), _load(args.b)
    for n in (A, B):
        for p in n.parameters():
            p.requires_grad_(False)

    print(f"  A = {args.a}  search={args.a_search}")
    print(f"  B = {args.b}  search={args.b_search}")
    print(f"  search cfg: horizon={cfg.horizon} k_cards={cfg.k_cards} "
          f"k_cells={cfg.k_cells}\n", flush=True)

    scores = []
    dev_b, steps_b = 0, 0
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), args.max_ticks)
        root.reset()
        base = root.snapshot()
        # B on team 0, then B on team 1, from the SAME opening.
        b0, d1, s1_ = duel(B, A, base.snapshot(), args.b_search, args.a_search, cfg)
        a0, d2, s2_ = duel(A, B, base.snapshot(), args.a_search, args.b_search, cfg)
        dev_b += d1[0] + d2[1]
        steps_b += s1_ + s2_
        scores.append(0.5 * (b0 + (1.0 - a0)))
        if (i + 1) % 25 == 0:
            print(f"  pair {i+1:>4}: B score {np.mean(scores):.3f}"
                  f" | B deviation {dev_b / max(1, steps_b):.1%}", flush=True)

    s = np.array(scores)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(s, len(s), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n{args.n} side-swapped pairings, no C++ heuristic (pure net vs net)")
    print(f"  B ({args.b}, search={args.b_search}) scores {s.mean():.4f}")
    print(f"     against A ({args.a}, search={args.a_search})")
    print(f"  95% CI [{lo:.4f}, {hi:.4f}]   (0.500 = evenly matched)")
    print(f"  B deviation rate: {dev_b / max(1, steps_b):.2%}")
    verdict = ("B is STRONGER" if lo > 0.5 else
               "B is WEAKER" if hi < 0.5 else "no difference resolved")
    print(f"  verdict: {verdict}")
    print(f"  clears the >0.60 bar? "
          f"{'YES' if lo > 0.60 else 'not resolved at this n' if hi > 0.60 else 'NO'}")


if __name__ == "__main__":
    main()
