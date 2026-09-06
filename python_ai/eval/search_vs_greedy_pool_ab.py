"""Does decision-time search still beat the greedy policy, on the CURRENT
distribution?

WHY THIS EXISTS. The +0.319 that justifies expert iteration was measured on
2026-08-11 against the C++ HeuristicOpponent at 1.5x elixir, with the ep-64k
GIANT-deck checkpoint, at horizon 4. Every one of those has since changed: the
deck is 2.6, the opponent is the UtilityTeacher on a 16-deck pool, the policy is
~110k episodes further on, and the engine has had the movement, sight, rolling
spell, elixir-phase and match-end changes. CLAUDE.md's own caveat on that result
says it "says nothing about neural opponents" and names the regime as a limit.

There is also direct evidence the edge has been shrinking: the 2026-09-05 gate
re-run found search's per-card aiming advantage had fallen to non-significance
(The Log +46 HP p=0.146, Cannon +262 p=0.092, Fireball +200 p=0.227), and
concluded the placement gap had closed and the constraint had moved to
selection.

So this is a GATE on the whole expert-iteration plan, not a curiosity. Distilling
an expert that is no longer better than the student is not neutral: CLAUDE.md
records that more data from a weak expert actively DEGRADES selectivity (the
ratio fell 3.07 -> 2.31 -> 2.17 as data grew).

DESIGN
------
Two arms, same seed and same pool deck per trial, differing ONLY in whether
`search_action` runs at each decision:

    A  greedy policy                 (use_search=False)
    B  identical net + 1-ply search  (use_search=True)

PAIRED BY SEED, NOT BY SNAPSHOT, for the reason `force_card_ab.py` documents: a
raw `env.snapshot()` copies the engine but NOT the Python-side teacher, so a
snapshot-paired arm would quietly face the C++ HeuristicOpponent instead of the
opponent the run actually trains against.

DECKS ARE CYCLED, NOT SAMPLED. PFSP weighting is what the training loop wants;
here it would silently concentrate the estimate on two matchups and make the
result a statement about those. Round-robin gives every deck the same number of
paired trials and makes the per-deck breakdown readable.

THE SNAPSHOT ASYMMETRY IS INHERENT AND IS NOT A CONFOUND: search's candidate
rollouts are stepped by the C++ heuristic while the real episode is played by
the teacher. That is how `search_action` has always worked and how it would work
in deployment -- it is a property of the SCORER, identical in both arms' notion
of what search is, and `PoolTeacherEnv` documents it.
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
from python_ai.rl.checkpointing import weights_path  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.search.search import outcome_score, play_episode  # noqa: E402
from python_ai.trainers.expert_collect import PoolTeacherEnv  # noqa: E402


def _seeded(env, seed):
    """Give the env's UtilityTeacher a deterministic RNG.

    Reaches through the gym wrapper deliberately: the teacher's seed is not part
    of any public constructor, and a measurement harness that cannot pin its
    opponent is not measuring what it claims to.
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
                    help="12 is the repo's own measured optimum (0.963 on the "
                         "2026-08 sweep); search is given its best shot so a "
                         "negative result is not a tuning artefact")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--terminal-weight", type=float, default=10.0,
                    help="a rollout that ENDED is scored by its real outcome at "
                         "this weight instead of by the critic. Exposed because "
                         "the 2026-09-06 match rules end a match at 3:00 on a "
                         "crown lead, so a 12-step rollout now reaches a "
                         "terminal far more often than when 10.0 was chosen -- "
                         "which makes this the most load-bearing constant in "
                         "the config and the one most likely to be stale.")
    ap.add_argument("--seed", type=int, default=90601)
    ap.add_argument("--decks", default="", help="comma-separated subset")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    device = torch.device("cpu")
    # weights_path, not the bare name: `python_ai/` is where the
    # checkpoints live and cwd must not be able to redirect it.
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

    a_scores, b_scores = [], []
    per_deck = defaultdict(lambda: [0.0, 0.0, 0])
    dev, dev_steps, cands = 0, 0, 0
    t0 = time.perf_counter()

    for i in range(args.n):
        deck = decks[i % len(decks)]
        seed = args.seed + i

        # SEED THE TEACHER, NOT JUST THE ENGINE. `PoolTeacherEnv` seeds the
        # engine (opening hands, lane bias), but `UtilityTeacher` holds its OWN
        # numpy RNG and gym_wrapper constructs it unseeded. Below rung 10 its
        # epsilon is non-zero -- 0.12 at rung 3 -- so each arm faced an opponent
        # making DIFFERENT random choices and the pairing was only partial.
        #
        # Caught because the greedy arm, which cannot be affected by search at
        # all, scored 0.750 in one run and 0.875 in another on identical seeds.
        # A control that must be constant and is not is the cheapest possible
        # detector for a broken pairing, and it is the reason to always have one.
        envA = _seeded(PoolTeacherEnv(args.stage, args.max_ticks, seed, deck), seed)
        rA, _sA, _d, _c = play_episode(net, envA, device, False, cfg)

        envB = _seeded(PoolTeacherEnv(args.stage, args.max_ticks, seed, deck), seed)
        rB, sB, dB, cB = play_episode(net, envB, device, True, cfg)

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
    # ACTUAL candidates scored per decision, not `cfg.max_candidates`, which is
    # only the proof-bound. If widening is firing on heads that are no longer
    # flat, this is where it shows: the critic ranking 40 cells it has never had
    # to separate is a different task from ranking 3.
    print(f"candidates/dec {cands/max(1, dev_steps):.1f} "
          f"(bound {cfg.max_candidates})")
    print(f"wall clock     {(time.perf_counter()-t0)/60:.1f} min")

    print(f"\n{'deck':26s}{'n':>4s}{'greedy':>9s}{'search':>9s}{'delta':>9s}")
    for name in sorted(per_deck, key=lambda k: (per_deck[k][1] - per_deck[k][0]) / per_deck[k][2]):
        a, b, n = per_deck[name]
        print(f"{name:26s}{n:4d}{a/n:9.3f}{b/n:9.3f}{(b-a)/n:+9.3f}")

    # THE DEVIATION RATE IS THE FIRST THING TO READ, not the delta. Search can
    # only deviate where the critic prefers something to the greedy action, so a
    # rate near zero means the two arms played nearly the same games and a null
    # is uninformative rather than evidence search does not help.
    if dev == 0:
        print("\nNOTE: search never deviated from greedy -- the arms are the "
              "same policy and this comparison is vacuous, not negative.")


if __name__ == "__main__":
    main()
