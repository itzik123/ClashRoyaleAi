"""Does decision-time search still beat the greedy policy on the current
distribution?

A gate on expert iteration: distilling an expert that is no longer better than
the student degrades it.

Two arms, same seed and pool deck per trial, differing only in whether
`search_action` runs:

    A  greedy policy                 (use_search=False)
    B  identical net + 1-ply search  (use_search=True)

Paired by seed, not snapshot (see force_card_ab.py). Decks are cycled rather
than PFSP-sampled, so every deck gets the same number of trials and the
per-deck breakdown is readable.

Search's rollouts are stepped by the C++ heuristic while the real episode is
played by the teacher, unless `--opponent-model` is set. That is a property of
the scorer, not a confound between the arms.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

from python_ai.eval.stats import paired  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.opponents.deck_pool import load_pool  # noqa: E402
from python_ai.opponents.teacher import UtilityTeacher  # noqa: E402
from python_ai.rl.checkpointing import weights_path  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.search.search import outcome_score, play_episode  # noqa: E402
from python_ai.trainers.expert_collect import PoolTeacherEnv  # noqa: E402


def _seeded(env, seed):
    """Give the env's UtilityTeacher a deterministic RNG.

    Reaches through the gym wrapper: the teacher's seed is not part of any
    public constructor, and a harness that cannot pin its opponent is not
    measuring what it claims.
    """
    teacher = getattr(env._gym, "teacher", None)
    if teacher is None:
        raise SystemExit("env has no teacher -- opponent is not the UtilityTeacher")
    teacher.rng = np.random.default_rng(seed)
    teacher.reset(np.random.default_rng(seed))
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=64, help="paired trials")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12,
                    help="the measured optimum, so a negative result is not a "
                         "tuning artefact")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--terminal-weight", type=float, default=10.0,
                    help="weight of a finished rollout's real outcome, used "
                         "instead of the critic")
    ap.add_argument("--seed", type=int, default=90601)
    ap.add_argument("--decks", default="", help="comma-separated subset")
    ap.add_argument("--opponent-model", action="store_true",
                    help="drive search's rollouts with a rules-only "
                         "UtilityTeacher on the real opponent's deck instead "
                         "of the C++ HeuristicOpponent")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    device = torch.device("cpu")
    net = load_net(weights_path(args.weights), device)
    net.eval()

    decks = load_pool()
    if args.decks:
        want = {s.strip() for s in args.decks.split(",") if s.strip()}
        decks = [d for d in decks if d.name in want]
    cfg = SearchCfg(horizon=args.horizon, k_cards=args.k_cards,
                    k_cells=args.k_cells,
                    terminal_weight=args.terminal_weight)

    print(f"weights   : {args.weights}")
    print(f"opponent  : UtilityTeacher rung {args.stage}, {len(decks)}-deck pool")
    print(f"search    : horizon {cfg.horizon}, k_cards {cfg.k_cards}, "
          f"k_cells {cfg.k_cells}, max {cfg.max_candidates} candidates")
    print(f"trials    : {args.n} paired ({2 * args.n} episodes)\n")
    print(f"rollout   : "
          f"{'UtilityTeacher rules-only' if args.opponent_model else 'C++ HeuristicOpponent'}")

    a_scores, b_scores = [], []
    per_deck = defaultdict(lambda: [0.0, 0.0, 0])
    dev, dev_steps, cands = 0, 0, 0
    t0 = time.perf_counter()

    for i in range(args.n):
        deck = decks[i % len(decks)]
        seed = args.seed + i

        # Seed the teacher too: PoolTeacherEnv seeds the engine, but
        # UtilityTeacher holds its own RNG, and below rung 10 its epsilon is
        # non-zero. The greedy arm is the control that must not move between
        # runs on identical seeds.
        envA = _seeded(PoolTeacherEnv(args.stage, args.max_ticks, seed, deck), seed)
        rA, _sA, _d, _c = play_episode(net, envA, device, False, cfg)

        envB = _seeded(PoolTeacherEnv(args.stage, args.max_ticks, seed, deck), seed)
        opp = None
        if args.opponent_model:
            # The same deck the real opponent holds, or the model is a
            # different opponent.
            opp = UtilityTeacher(list(deck.card_ids), team=1, epsilon=0.0,
                                 horizon_ticks=0, k_cells=1, max_combos=0,
                                 seed=seed)
            opp.reset()
        rB, sB, dB, cB = play_episode(net, envB, device, True, cfg, opponent=opp)

        sa, sb = outcome_score(rA), outcome_score(rB)
        a_scores.append(sa)
        b_scores.append(sb)
        dev += dB
        dev_steps += sB
        cands += cB
        acc = per_deck[deck.name]
        acc[0] += sa
        acc[1] += sb
        acc[2] += 1

        if (i + 1) % 8 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:3d}/{args.n}  greedy {np.mean(a_scores):.3f}  "
                  f"search {np.mean(b_scores):.3f}  ({el/60:.1f} min)")
            sys.stdout.flush()

    res = paired(a_scores, b_scores)
    print(f"\n{'':22s}{'greedy':>9s}{'search':>9s}")
    print(f"{'win rate':22s}{res.mean_a:9.3f}{res.mean_b:9.3f}")
    print(f"\npaired delta   {res.delta:+.4f}   95% CI "
          f"[{res.lo:+.4f}, {res.hi:+.4f}]")
    print(f"discordant     {res.better} search-better / {res.worse} "
          f"search-worse / {res.tied} tied")
    print(f"sign test      p = {res.p:.4g}")
    print(f"deviation rate {dev/max(1, dev_steps):.3%} "
          f"({dev} of {dev_steps} decisions)")
    # Actual candidates scored, not the `max_candidates` bound.
    print(f"candidates/dec {cands/max(1, dev_steps):.1f} "
          f"(bound {cfg.max_candidates})")
    print(f"wall clock     {(time.perf_counter()-t0)/60:.1f} min")

    print(f"\n{'deck':26s}{'n':>4s}{'greedy':>9s}{'search':>9s}{'delta':>9s}")
    for name in sorted(per_deck, key=lambda k: (per_deck[k][1] - per_deck[k][0]) / per_deck[k][2]):
        a, b, n = per_deck[name]
        print(f"{name:26s}{n:4d}{a/n:9.3f}{b/n:9.3f}{(b-a)/n:+9.3f}")

    # Search can only deviate where the critic prefers another action, so a
    # rate near zero makes a null uninformative rather than negative.
    if dev == 0:
        print("\nNOTE: search never deviated from greedy -- the arms are the "
              "same policy and this comparison is vacuous, not negative.")


if __name__ == "__main__":
    main()
