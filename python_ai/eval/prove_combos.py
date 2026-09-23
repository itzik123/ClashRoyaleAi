"""Does multi-card combo planning reach the board, and does it pay?

    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --usage
    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --reserve-ab --n 40
    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --vs-net --n 30

  --usage          how often a combo is proposed, chosen and completed, and its latency cost
  --combo-ab       the safety arm: combos on vs off, same teacher otherwise
  --reserve-ab     sweep of the plan reserve against 0.0
  --profile-sweep  one weight swept against both strength and usage
  --vs-net         the strength benchmark, sweeping lookahead

Usage is reported apart from win rate because a win-rate arm cannot tell "the
combo did not help" from "the combo never happened".

Every opening is `env.seed(args.seed + ENGINE_SEED_OFFSET + i)` and both arms
share its snapshot, so comparisons are paired. The same `--seed` should draw
the same match population across invocations; until that is confirmed by a run
(UPSTREAM_REQUESTS.md item 23), compare deltas across runs, never arm levels.
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.eval import stats  # noqa: E402
from python_ai.eval.match_outcome import score_from_towers  # noqa: E402
from python_ai.opponents.teacher import (  # noqa: E402
    PROFILES, TEACHER_STAGES, UtilityTeacher,
)

CE = E.ClashRoyaleEnv
HAND_SIZE = CE.HAND_SIZE
MAX_STEPS = 400

#: Separates the engine's seed stream from the teachers' (`args.seed + i`), so
#: a teacher's lane bias and its dealt hand do not move together. Any stable
#: value works.
ENGINE_SEED_OFFSET = 104729

#: The teacher's shipped play_margin, read rather than restated, so a sweep
#: pairs against current behaviour.
DEFAULT_PLAY_MARGIN = UtilityTeacher(DEFAULT_DECK, team=0).play_margin

COMBO_KINDS = ("supported_push", "counter_push", "defensive_stack",
               "cheap_defence", "spell_then_push", "push_then_spell")


def _score(env):
    return score_from_towers(env, 0)


def make_teacher(team, stage, seed, reserve=None, horizon=None, combos=None,
                 drop_family=None):
    t = UtilityTeacher(DEFAULT_DECK, team=team, profile="balanced", seed=seed,
                       combo_reserve=reserve)
    t.set_stage(stage)
    if horizon is not None:
        t.horizon_ticks = int(horizon)
    if combos is not None:
        t.max_combos = int(combos)
    if drop_family:
        t.combo_families = tuple(k for k in t.combo_families
                                 if k != drop_family)
    t.reset()
    return t


class Telemetry:
    """Everything a combo run reports, accumulated across matches.

    `completed` counts a pair only when the second card is the action actually
    taken on the following decision, i.e. the plan survived re-scoring.
    """

    def __init__(self):
        self.proposed = Counter()
        self.chosen = Counter()
        self.completed = Counter()
        self.decisions = 0
        self.plays = 0
        self.elixir = []
        self.seconds = 0.0

    def line(self, label):
        e = np.asarray(self.elixir or [0.0])
        n = max(1, self.decisions)
        combos = sum(self.chosen[k] for k in COMBO_KINDS)
        done = sum(self.completed.values())
        # Percent of plays, not decisions: a bot that holds more has fewer
        # active decisions, so a per-decision rate rises for free when spending
        # falls.
        plays = max(1, self.plays)
        return (f"  {label:<26} dec {self.decisions:>5}  "
                f"elixir {e.mean():.2f}/p90 {np.percentile(e, 90):.2f}  "
                f"plays {self.plays:>4}  "
                f"proposed {sum(self.proposed[k] for k in COMBO_KINDS):>4}  "
                f"chosen {combos:>3} ({100.0 * combos / plays:.1f}% of plays)  "
                f"completed {done:>3}  "
                f"{1000.0 * self.seconds / n:.2f} ms/dec")


def play_match(env, t0, t1, tele=None, net=None, net_team=None):
    """One match. Team 0 is `t0` unless `net_team == 0`, in which case the net
    plays that side. Returns team 0's score.

    Telemetry reads the teacher's own `last_kind` label: reduced to
    coordinates, a combo's first step looks like a lone play, and a pending
    plan spans several decisions, so counting "a plan exists" per decision
    would inflate `chosen`.
    """
    import torch
    hid = None
    if net is not None:
        hid = (torch.zeros(1, 256), torch.zeros(1, 256))
    plan_kind = {0: None, 1: None}
    for _ in range(MAX_STEPS):
        acts = {}
        for team, t in ((0, t0), (1, t1)):
            if net is not None and team == net_team:
                from python_ai.eval.prove_teacher import _net_greedy
                with torch.no_grad():
                    gi, gx, gy, hid = _net_greedy(net, env, team, hid)
                acts[team] = (gi, gx, gy)
                continue
            obs = np.asarray(env.get_observation_for_team(team), np.float32)
            clock = time.perf_counter()
            if tele is not None and team == 0:
                tele.decisions += 1
                tele.elixir.append(float(env.get_elixir_for_team(team)))
                for c in t.candidates(env, obs):
                    tele.proposed[c.kind] += 1
            slot, x, y = t.act(env, obs)
            if tele is not None and team == 0:
                tele.seconds += time.perf_counter() - clock
                if slot < HAND_SIZE:
                    tele.plays += 1
                if t.last_kind in COMBO_KINDS:
                    tele.chosen[t.last_kind] += 1
                elif t.last_kind == "followup" and plan_kind[team]:
                    tele.completed[plan_kind[team]] += 1
            if t.last_kind in COMBO_KINDS:
                plan_kind[team] = t.last_kind
            elif t.last_kind == "followup":
                plan_kind[team] = None
            acts[team] = (slot, x, y)
        a, b = acts[0], acts[1]
        if env.step_self_play(a[0], a[1], a[2], b[0], b[1], b[2], 10).done:
            break
    return _score(env)


# --- 1. usage ---
def run_usage(args):
    print("\nCOMBO USAGE -- teacher vs teacher, both sides identical\n")
    stages = ([args.stage] if args.stage is not None
              else range(len(TEACHER_STAGES)))
    for stage in stages:
        tele = Telemetry()
        for i in range(args.n):
            env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
            # seed() is a seeded reset(): it seeds both engine generators and
            # re-deals.
            env.seed(args.seed + ENGINE_SEED_OFFSET + i)
            play_match(env,
                       make_teacher(0, stage, args.seed + i, args.reserve),
                       make_teacher(1, stage, args.seed + 7777 + i, args.reserve),
                       tele)
        cfg = TEACHER_STAGES[stage]
        print(tele.line(f"stage {stage} h={cfg['horizon_ticks']} "
                        f"mc={cfg['max_combos']}"))
        by_kind = {k: (tele.proposed[k], tele.chosen[k], tele.completed[k])
                   for k in COMBO_KINDS if tele.proposed[k]}
        print(f"       proposed/chosen/completed by kind: {by_kind}")


# --- 2. the reserve ---
def run_reserve_ab(args):
    """Sweep `teacher.COMBO_RESERVE`, the charge protecting a committed plan.

    Only team 0's reserve varies (team 1 stays at 0.0), and sides are swapped.
    """
    values = [float(v) for v in args.reserve_grid.split(",")]
    print(f"\nCOMBO RESERVE A/B -- stage {args.stage or 5}, "
          f"{args.n} paired openings, sides swapped\n")
    stage = args.stage if args.stage is not None else 5
    arms = {v: [] for v in values}
    teles = {v: Telemetry() for v in values}
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        # seed() is a seeded reset().
        root.seed(args.seed + ENGINE_SEED_OFFSET + i)
        base = root.snapshot()
        for v in values:
            side_scores = []
            for me in (0, 1):
                env = base.snapshot()
                mine = make_teacher(me, stage, args.seed + i, v)
                theirs = make_teacher(1 - me, stage, args.seed + 7777 + i, 0.0)
                t0, t1 = (mine, theirs) if me == 0 else (theirs, mine)
                s = play_match(env, t0, t1,
                               teles[v] if me == 0 else None)
                side_scores.append(s if me == 0 else 1.0 - s)
            arms[v].append(float(np.mean(side_scores)))
    for v in values:
        print(teles[v].line(f"reserve={v}"))
    print()
    for v in values[1:]:
        r = stats.paired(arms[values[0]], arms[v])
        print(r.format(f"reserve {values[0]} -> {v}",
                       f"r={values[0]}", f"r={v}"))


def run_combo_ab(args):
    """The safety measurement: combos on vs off, same teacher otherwise.

    Combo candidates spend rollouts that would have gone to single cards, and a
    plan holds a slot and elixir for several decisions; those costs are paid
    every decision. If the pair is a net loss, report it.

    Only team 0's combo width varies; sides are swapped.
    """
    stage = args.stage if args.stage is not None else 5
    off, on = [], []
    tele_off, tele_on = Telemetry(), Telemetry()
    drop = (f" (ON arm without {args.drop_family})"
            if args.drop_family else "")
    print(f"\nCOMBOS ON vs OFF{drop} -- stage {stage}, {args.n} paired "
          f"openings, sides swapped\n")
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        # seed() is a seeded reset().
        root.seed(args.seed + ENGINE_SEED_OFFSET + i)
        base = root.snapshot()
        for combos, bucket, tele in ((0, off, tele_off), (None, on, tele_on)):
            side = []
            for me in (0, 1):
                env = base.snapshot()
                mine = make_teacher(me, stage, args.seed + i, args.reserve,
                                    combos=combos,
                                    drop_family=args.drop_family)
                theirs = make_teacher(1 - me, stage, args.seed + 7777 + i,
                                      args.reserve)
                t0, t1 = (mine, theirs) if me == 0 else (theirs, mine)
                s = play_match(env, t0, t1, tele if me == 0 else None)
                side.append(s if me == 0 else 1.0 - s)
            bucket.append(float(np.mean(side)))
    print(tele_off.line("combos OFF"))
    print(tele_on.line("combos ON"))
    print()
    print(stats.paired(off, on).format("combos off -> on", "off", "on"))


# --- 2b. the profile sweep ---
def teacher_vs_heuristic(env, teacher, tele=None):
    """Teacher on team 0 through `env.step()`, so the C++ HeuristicOpponent plays
    team 1. Returns the teacher's score.

    Same routing as `prove_teacher.teacher_vs_heuristic`, so the numbers are
    comparable.
    """
    for _ in range(MAX_STEPS):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        clock = time.perf_counter()
        if tele is not None:
            tele.decisions += 1
            tele.elixir.append(float(env.get_elixir_for_team(0)))
            for c in teacher.candidates(env, obs):
                tele.proposed[c.kind] += 1
        slot, x, y = teacher.act(env, obs)
        if tele is not None:
            tele.seconds += time.perf_counter() - clock
            if slot < HAND_SIZE:
                tele.plays += 1
            if teacher.last_kind in COMBO_KINDS:
                tele.chosen[teacher.last_kind] += 1
            elif teacher.last_kind == "followup":
                tele.completed["followup"] += 1
        if env.step(slot, x, y, 10).done:
            break
    return _score(env)


def run_profile_sweep(args):
    """Sweep one weight against both bars a profile must clear at once.

    Every arm plays snapshots of the same root openings, so the comparison is
    paired within this process.

      STRENGTH   against the C++ heuristic, which saturates (it can only say "not broken"), and head to head against the shipped profile with sides swapped, where 0.500 means no strength was traded away
      USAGE      combos as a share of plays, not decisions
    """
    weight = args.sweep_weight
    values = [float(v) for v in args.sweep_grid.split(",")]
    base = dict(PROFILES[args.base_profile])
    stage = args.stage if args.stage is not None else 5
    print()
    print(f"PROFILE SWEEP -- {weight} over {values}")
    print(f"  base {args.base_profile} {base}")
    print(f"  stage {stage}, {args.n} shared openings, vs the C++ "
          f"HeuristicOpponent @1.0x")
    print()

    roots = []
    for i in range(args.n):
        env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        # seed() is a seeded reset().
        env.seed(args.seed + ENGINE_SEED_OFFSET + i)
        roots.append(env.snapshot())

    results = {}
    for v in values:
        # play_margin is a teacher attribute, not a profile weight, so it is
        # handled separately rather than added to PROFILES.
        weights = dict(base)
        if weight != "play_margin":
            weights[weight] = v
        def _make(team, w, margin):
            t = UtilityTeacher(DEFAULT_DECK, team=team, profile=w,
                               seed=args.seed, combo_reserve=args.reserve)
            t.set_stage(stage)
            t.reset()
            if margin is not None:
                t.play_margin = margin
            return t

        margin = v if weight == "play_margin" else None
        tele = Telemetry()
        scores, h2h = [], []
        for root in roots:
            env = root.snapshot()
            scores.append(teacher_vs_heuristic(
                env, _make(0, weights, margin), tele))
            # Head to head against the shipped profile, sides swapped: the arm
            # that can rank two profiles that both beat the heuristic.
            side = []
            for me in (0, 1):
                duel = root.snapshot()
                mine = _make(me, weights, margin)
                theirs = _make(1 - me, base, None)
                t0, t1 = (mine, theirs) if me == 0 else (theirs, mine)
                sc = play_match(duel, t0, t1)
                side.append(sc if me == 0 else 1.0 - sc)
            h2h.append(float(np.mean(side)))
        m, lo, hi = stats.bootstrap_ci(scores)
        plays = max(1, tele.plays)
        combos = sum(tele.chosen[k] for k in COMBO_KINDS)
        e = np.asarray(tele.elixir or [0.0])
        hm, hlo, hhi = stats.bootstrap_ci(h2h)
        results[v] = dict(score=m, lo=lo, hi=hi, combo_share=combos / plays,
                          combos=combos, plays=plays, scores=scores,
                          h2h=h2h, h2h_mean=hm, h2h_lo=hlo, h2h_hi=hhi,
                          elixir=float(e.mean()), p90=float(np.percentile(e, 90)))
        # Share of decisions at or above the overflow line: past some point,
        # holding elixir wastes income. 9.0 matches `score`'s overflow relief
        # and W_ELIXIR_OVERFLOW.
        waste = float((e >= 9.0).mean())
        results[v]["overflow"] = waste
        print(f"  {weight}={v:<6g} vsHeur {m:.3f}  vsBase {hm:.3f} "
              f"[{hlo:.3f}, {hhi:.3f}]   "
              f"elixir {e.mean():.2f}/p90 {np.percentile(e, 90):.2f} "
              f"/overflow {100.0 * waste:4.1f}%   "
              f"plays {tele.plays:>4}  combos {combos:>3} "
              f"= {100.0 * combos / plays:5.1f}% of plays")

    print()
    print("  paired against the base value, same openings:")
    base_v = float(base[weight]) if weight in base else DEFAULT_PLAY_MARGIN
    if base_v in results:
        for v in values:
            if v == base_v:
                continue
            r = stats.paired(results[base_v]["scores"], results[v]["scores"])
            print(r.format(f"{weight} {base_v} -> {v}", f"{base_v}", f"{v}"))
    return results


# --- 3. the strength benchmark ---
def run_vs_net(args):
    import torch
    from python_ai.models.policy_io import load_net
    path = args.weights or "model_weights_selfplay.pth"
    full = path if os.path.isabs(path) else os.path.join(
        python_ai.PACKAGE_DIR, path)
    net = load_net(full, torch.device("cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"\nTEACHER vs {os.path.basename(full)} (greedy) -- "
          f"{args.n} paired openings, sides swapped, lookahead sweep\n")
    print("  reported score is the TEACHER's\n")
    results = {}
    for h in horizons:
        tele = Telemetry()
        scores = []
        for i in range(args.n):
            root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
            # seed() is a seeded reset().
            root.seed(args.seed + ENGINE_SEED_OFFSET + i)
            base = root.snapshot()
            side = []
            for tteam in (0, 1):
                env = base.snapshot()
                t = make_teacher(tteam, args.stage or 5, args.seed + i,
                                 args.reserve, horizon=h)
                other = make_teacher(1 - tteam, 0, args.seed + 1, 0.0)
                t0, t1 = (t, other) if tteam == 0 else (other, t)
                s = play_match(env, t0, t1, tele if tteam == 0 else None,
                               net=net, net_team=1 - tteam)
                side.append(s if tteam == 0 else 1.0 - s)
            scores.append(float(np.mean(side)))
        results[h] = scores
        m, lo, hi = stats.bootstrap_ci(scores)
        print(f"  horizon {h:>3} ({h / 10.0:>4.1f} s)  teacher {m:.4f}  "
              f"CI [{lo:.4f}, {hi:.4f}]")
        print(tele.line(f"    h={h}"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", action="store_true")
    ap.add_argument("--reserve-ab", action="store_true")
    ap.add_argument("--combo-ab", action="store_true")
    ap.add_argument("--profile-sweep", action="store_true")
    ap.add_argument("--sweep-weight", default="w_pos")
    ap.add_argument("--sweep-grid", default="4,6,8,10,14,20")
    ap.add_argument("--base-profile", default="balanced")
    ap.add_argument("--drop-family", default=None,
                    help="ablate one combo family from the ON arm, e.g. "
                         "cheap_defence")
    ap.add_argument("--vs-net", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--reserve", type=float, default=None,
                    help="teacher.COMBO_RESERVE override (None = the default)")
    ap.add_argument("--reserve-grid", default="0.0,1.5,3.0,5.0")
    ap.add_argument("--horizons", default="30,50,70,100,120")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\ndeck {list(DEFAULT_DECK)}")
    if args.usage:
        run_usage(args)
    if args.reserve_ab:
        run_reserve_ab(args)
    if args.combo_ab:
        run_combo_ab(args)
    if args.profile_sweep:
        run_profile_sweep(args)
    if args.vs_net:
        run_vs_net(args)
    if not (args.usage or args.reserve_ab or args.combo_ab or args.vs_net
            or args.profile_sweep):
        ap.error("pick one of --usage / --combo-ab / --reserve-ab / "
                 "--profile-sweep / --vs-net")


if __name__ == "__main__":
    main()
