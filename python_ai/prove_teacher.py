"""Is the utility teacher a better Phase-1 opponent than the C++ heuristic?

    python_ai/venv/Scripts/python.exe python_ai/prove_teacher.py --n 60
    python_ai/venv/Scripts/python.exe python_ai/prove_teacher.py --sweep --n 40
    python_ai/venv/Scripts/python.exe python_ai/prove_teacher.py --vs shipping --n 60

TWO BARS, AND BOTH MUST BE CLEARED
----------------------------------
1. STRONG ENOUGH -- the teacher beats the C++ `HeuristicOpponent` decisively at
   1.0x. If it does not, it is not an upgrade on what Phase 1 already had, and
   the whole pivot is a lateral move.

2. NOT A WALL -- a trained policy can still take games off it. An opponent
   nothing can beat produces a FLAT reward signal, which is the zero-gradient
   failure this pivot is supposed to fix, wearing a different costume. At 1.0x
   the C++ heuristic fails bar 1 (the ep-64k and ep-25202 nets both beat it
   ~100%); a wall would fail bar 2. The teacher has to sit between.

WHY BAR 2 IS PROBED WITH THE GREEDY NET BY DEFAULT
---------------------------------------------------
`search_ab_test._search_action` rolls candidates forward with `sim.step()`,
which runs the C++ heuristic INSIDE the rollout. Against a teacher opponent
that is the wrong opponent model, so reusing it here would measure a search
handicapped by a mismatch rather than the teacher's beatability.

`--search` therefore uses a local rollout built on `step_self_play` with team 1
no-oping -- the same both-sides-no-op assumption `teacher.rollout_stats` makes,
so the two sides of the comparison at least share it. The DEFAULT is greedy,
which is the stricter bar: if a greedy policy can already take games, the
teacher is certainly not a wall.

PAIRING
-------
Every comparison is paired through `env.snapshot()`: one `reset()`, then a
bit-exact opening handed to both arms, so the shuffled hand and the heuristic's
lane draw are identical. `UPSTREAM_REQUESTS.md` item 7 asks for `ClashEnv::seed()`
for exactly this and snapshot already supplies it -- unpaired, resolving 5
win-rate points needs ~1,568 episodes per arm.

Outcome convention matches `net_h2h.py`: surviving tower count, win/draw/loss
1.0/0.5/0.0. That ignores TimeoutRules' HP tiebreak, deliberately, so every
number in this file is comparable to every other harness in the project.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env as E  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from teacher import PROFILES, TEACHER_STAGES, UtilityTeacher  # noqa: E402

CE = E.ClashRoyaleEnv
HAND_SIZE = CE.HAND_SIZE
MAX_STEPS = 400


def _score(env):
    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def make_teacher(team, stage, profile=None, seed=None):
    t = UtilityTeacher(DEFAULT_DECK, team=team, profile=profile, seed=seed)
    t.set_stage(stage)
    return t


# --------------------------------------------------------------------------
# bar 1: teacher vs the C++ HeuristicOpponent
# --------------------------------------------------------------------------
def teacher_vs_heuristic(env, teacher):
    """Teacher on team 0 through `env.step()`, so ClashEnv::opponentTurn runs
    the C++ heuristic for team 1. Returns the teacher's score."""
    teacher.reset()
    for _ in range(MAX_STEPS):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = teacher.act(env, obs)
        if env.step(slot, x, y, 10).done:
            break
    return _score(env)


# --------------------------------------------------------------------------
# bar 2: a net against the teacher
# --------------------------------------------------------------------------
def _net_greedy(net, env, team, hid):
    import torch
    obs = torch.tensor(np.asarray(env.get_observation_for_team(team),
                                  np.float32)).unsqueeze(0)
    feats, emb, sp = net.extract_features(obs)
    lg, _, _, _, hid = net.step_lstm_and_card(feats, hid,
                                              net.affordability_mask(obs))
    gi = int(lg.argmax(-1).item())
    cell = int(net.placement_given_card(hid[0], emb, torch.tensor([gi]), obs,
                                        sp).argmax(-1).item())
    return gi, float(cell % CE.BOARD_WIDTH), float(cell // CE.BOARD_WIDTH), hid


def _net_search(net, env, team, hid, horizon, k_cards, k_cells,
                terminal_weight):
    """Greedy plus top-k alternatives, rolled forward with `step_self_play` and
    team 1 no-oping, scored by the net's own critic.

    Local rather than reused from `search_ab_test` because that module's rollout
    calls `sim.step()`, which runs the C++ heuristic for team 1 -- the wrong
    opponent when the real opponent is the teacher.
    """
    import torch
    obs = torch.tensor(np.asarray(env.get_observation_for_team(team),
                                  np.float32)).unsqueeze(0)
    feats, emb, sp = net.extract_features(obs)
    lg, _, _, _, hid = net.step_lstm_and_card(feats, hid,
                                              net.affordability_mask(obs))

    def cell_xy(cell):
        return float(cell % CE.BOARD_WIDTH), float(cell // CE.BOARD_WIDTH)

    gi = int(lg.argmax(-1).item())
    gcell = int(net.placement_given_card(hid[0], emb, torch.tensor([gi]), obs,
                                         sp).argmax(-1).item())
    greedy = (gi, 0.0, 0.0) if gi >= HAND_SIZE else (gi,) + cell_xy(gcell)
    cands, seen = [greedy], {greedy}

    for c in torch.topk(lg[0], min(k_cards, lg.shape[-1])).indices:
        if not torch.isfinite(lg[0, c]):
            continue
        ci = int(c.item())
        if ci >= HAND_SIZE:
            if (HAND_SIZE, 0.0, 0.0) not in seen:
                seen.add((HAND_SIZE, 0.0, 0.0))
                cands.append((HAND_SIZE, 0.0, 0.0))
            continue
        pl = net.placement_given_card(hid[0], emb, c.view(1), obs, sp)
        for cell in torch.topk(pl[0], k_cells).indices:
            if not torch.isfinite(pl[0, cell]):
                continue
            a = (ci,) + cell_xy(int(cell.item()))
            if a not in seen:
                seen.add(a)
                cands.append(a)

    if len(cands) == 1:
        return greedy[0], greedy[1], greedy[2], hid

    finals, terminal = [], []
    for ci, x, y in cands:
        sim = env.snapshot()
        a = (ci, x, y) if team == 0 else (-1, 0.0, 0.0)
        b = (-1, 0.0, 0.0) if team == 0 else (ci, x, y)
        r = sim.step_self_play(a[0], a[1], a[2], b[0], b[1], b[2], 10)
        for _ in range(horizon - 1):
            if r.done:
                break
            r = sim.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 10)
        finals.append(sim.get_observation_for_team(team))
        # reward0 is team 0's; team 1's outcome is its negation.
        terminal.append((r.done, float(r.reward0) * (1.0 if team == 0 else -1.0)))

    batch = torch.tensor(np.asarray(finals, np.float32))
    f2, _, _ = net.extract_features(batch)
    n = len(cands)
    hx = hid[0].expand(n, hid[0].shape[-1]).contiguous()
    cx = hid[1].expand(n, hid[1].shape[-1]).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(f2, (hx, cx))
    scores = values.squeeze(-1).clone()
    for i, (done, rew) in enumerate(terminal):
        if done:
            scores[i] = rew * terminal_weight
    best = cands[int(scores.argmax().item())]
    return best[0], best[1], best[2], hid


def net_vs_teacher(net, env, net_team, teacher, use_search, cfg):
    """Returns the NET's score. `net_team` is 0 or 1; the teacher takes the
    other side. Sides are swapped by the caller because a policy once beat a
    bit-exact copy of itself 0.598 purely by side assignment."""
    import torch
    teacher.reset()
    hid = (torch.zeros(1, 256), torch.zeros(1, 256))
    t_team = 1 - net_team
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            if use_search:
                ni, nx, ny, hid = _net_search(
                    net, env, net_team, hid, cfg["horizon"], cfg["k_cards"],
                    cfg["k_cells"], cfg["terminal_weight"])
            else:
                ni, nx, ny, hid = _net_greedy(net, env, net_team, hid)
        tobs = np.asarray(env.get_observation_for_team(t_team), np.float32)
        ti, tx, ty = teacher.act(env, tobs)
        if net_team == 0:
            r = env.step_self_play(ni, nx, ny, ti, tx, ty, 10)
        else:
            r = env.step_self_play(ti, tx, ty, ni, nx, ny, 10)
        if r.done:
            break
    s = _score(env)
    return s if net_team == 0 else 1.0 - s


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(label, scores, bar=None):
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(s, len(s), replace=True).mean()
                     for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    line = f"  {label:<42} {s.mean():.4f}  CI [{lo:.4f}, {hi:.4f}]  n={len(s)}"
    if bar is not None:
        line += "   PASS" if s.mean() >= bar else "   FAIL"
    print(line, flush=True)
    return s.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--vs", choices=["heuristic", "shipping", "ladder"],
                    default="heuristic")
    ap.add_argument("--sweep", action="store_true",
                    help="score every weight profile against the heuristic")
    ap.add_argument("--opp-elixir", type=float, default=1.0)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="net for --vs shipping (default: shipping.py's)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\ndeck {DEFAULT_DECK}   stage {args.stage} "
          f"{TEACHER_STAGES[args.stage]}   opp_elixir {args.opp_elixir}\n")

    if args.sweep:
        print("BAR 1 SWEEP -- teacher vs C++ HeuristicOpponent, per profile")
        print("(selection only; the winner must be CONFIRMED on a fresh run)")
        for name in PROFILES:
            scores = []
            for i in range(args.n):
                env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
                env.set_opponent_elixir_multiplier(args.opp_elixir)
                env.reset()
                scores.append(teacher_vs_heuristic(
                    env, make_teacher(0, args.stage, name, args.seed + i)))
            report(f"profile={name}", scores)
        return

    if args.vs == "heuristic":
        print("BAR 1 -- teacher vs the C++ HeuristicOpponent")
        for stage in range(len(TEACHER_STAGES)):
            scores = []
            for i in range(args.n):
                env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
                env.set_opponent_elixir_multiplier(args.opp_elixir)
                env.reset()
                scores.append(teacher_vs_heuristic(
                    env, make_teacher(0, stage, None, args.seed + i)))
            report(f"stage {stage} "
                   f"(h={TEACHER_STAGES[stage]['horizon_ticks']}, "
                   f"eps={TEACHER_STAGES[stage]['epsilon']})", scores)
        return

    if args.vs == "ladder":
        print("MONOTONICITY -- stage 5 teacher vs every lower stage, paired")
        for lo_stage in range(len(TEACHER_STAGES) - 1):
            scores = []
            for i in range(args.n):
                root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
                root.reset()
                env = root.snapshot()
                hi_t = make_teacher(0, 5, None, args.seed + i)
                lo_t = make_teacher(1, lo_stage, None, args.seed + 9000 + i)
                hi_t.reset()
                lo_t.reset()
                for _ in range(MAX_STEPS):
                    o0 = np.asarray(env.get_observation_for_team(0), np.float32)
                    o1 = np.asarray(env.get_observation_for_team(1), np.float32)
                    a0 = hi_t.act(env, o0)
                    a1 = lo_t.act(env, o1)
                    if env.step_self_play(a0[0], a0[1], a0[2],
                                          a1[0], a1[1], a1[2], 10).done:
                        break
                scores.append(_score(env))
            report(f"stage 5 vs stage {lo_stage}", scores)
        return

    # --vs shipping
    import torch
    from expert_iteration import load_net
    import shipping
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.weights or shipping.SHIPPING_WEIGHTS
    net = load_net(path if os.path.isabs(path) else os.path.join(here, path),
                   torch.device("cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    cfg = {"horizon": shipping.SEARCH_HORIZON,
           "k_cards": shipping.SEARCH_K_CARDS,
           "k_cells": shipping.SEARCH_K_CELLS,
           "terminal_weight": shipping.SEARCH_TERMINAL_WEIGHT}
    mode = f"search h={cfg['horizon']}" if args.search else "greedy"
    print(f"BAR 2 -- {os.path.basename(path)} ({mode}) vs the teacher, "
          f"sides swapped")
    scores = []
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.reset()
        base = root.snapshot()
        a = net_vs_teacher(net, base.snapshot(), 0,
                           make_teacher(1, args.stage, None, args.seed + i),
                           args.search, cfg)
        b = net_vs_teacher(net, base.snapshot(), 1,
                           make_teacher(0, args.stage, None, args.seed + i),
                           args.search, cfg)
        scores.append(0.5 * (a + b))
        if (i + 1) % 10 == 0:
            print(f"    pair {i+1:>4}: net {np.mean(scores):.3f}", flush=True)
    mean, lo, hi = report(f"net score vs teacher stage {args.stage}", scores)
    print()
    if hi < 0.05:
        print("  VERDICT: the teacher is a WALL -- a trained policy cannot "
              "take games off it, so the reward signal would be flat.")
    elif lo > 0.95:
        print("  VERDICT: the teacher is too WEAK -- same ceiling problem the "
              "C++ heuristic has at 1.0x.")
    else:
        print("  VERDICT: bar 2 PASSES -- beatable, but not free.")


if __name__ == "__main__":
    main()
