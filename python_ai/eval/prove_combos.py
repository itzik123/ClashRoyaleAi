"""Does multi-card COMBO planning actually reach the board, and does it pay?

    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --usage
    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --reserve-ab --n 40
    python_ai/venv/Scripts/python.exe python_ai/eval/prove_combos.py --vs-net --n 30

THREE QUESTIONS, DELIBERATELY SEPARATE
--------------------------------------
1. `--usage`    can the teacher EXPRESS a combo, how often does one reach the
                board, and what does the extra width cost in latency.
2. `--combo-ab` the SAFETY arm: combos on vs off, same teacher otherwise.
3. `--reserve-ab` the plan reserve is a behaviour change with no prior
                measurement, so it is swept against 0.0 before being believed.
4. `--vs-net`   the strength benchmark, sweeping lookahead.

WHY USAGE IS REPORTED SEPARATELY FROM WIN RATE, AND WHY BOTH ARE NEEDED. A
generator can propose a play the bot can never afford -- which is exactly what
the first version of this change did, measured at P(bar >= 6) = 0.3% -- and a
win-rate arm alone cannot tell "the combo did not help" apart from "the combo
never happened". CLAUDE.md records the same confusion three times under a
different name: an aggregate that cannot see the conditional.

PAIRING, AND THE ONE THING IT DOES NOT BUY
------------------------------------------
Every comparison shares one `env.snapshot()` opening, so within a run both arms
get a bit-exact hand and lane draw. Unpaired, resolving 5 win-rate points needs
~1,568 episodes per arm; this project has the snapshot, so it pairs.

**But `--seed` does NOT make a run reproducible, and expecting it to cost this
harness a control.** `ClashEnv::reset()`'s opening-hand shuffle is unseeded
(`UPSTREAM_REQUESTS.md` item 7, still open); `--seed` reaches only the teachers'
own RNG, which at stage 5 is just the lane bias. So two invocations draw
entirely different match populations whatever seed is passed.

Measured: an ablation run designed to share seed 300 with an earlier run, and to
be validated by its OFF arm reproducing that run's 0.537, instead reported
0.475. Nothing was wrong with either run -- the expectation was wrong.

**Across runs, compare DELTAS only, never arm levels.** Two runs happening to
report the same OFF arm is coincidence, not reproducibility.

THE CHECKPOINT. The session brief asks for "ep 25202". That checkpoint no
longer exists -- the 2026-08-19 cleanup kept only `model_weights.pth` (ep 7,063,
phase 1) and `model_weights_selfplay.pth` (ep 31,312), and CLAUDE.md says so
plainly. `model_weights_selfplay.pth` is the surviving descendant of that same
2.6 Hog Cycle run, ~6k episodes further on, and is the default here.
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.eval import stats  # noqa: E402
from python_ai.opponents.teacher import (  # noqa: E402
    PROFILES, TEACHER_STAGES, UtilityTeacher,
)

CE = E.ClashRoyaleEnv
HAND_SIZE = CE.HAND_SIZE
MAX_STEPS = 400

#: The families `teacher._legal_combos` can emit. Named here so a run that
#: emits a kind nobody expected shows up as a new column rather than silently
#: joining "other".
#: `UtilityTeacher.play_margin`'s shipped value, so a sweep over it can pair
#: against the current behaviour the same way a weight sweep pairs against
#: PROFILES. READ FROM THE TEACHER, never restated -- a second copy went stale
#: the moment the default moved 0.05 -> 3.0, and the sweep then labelled its
#: rows against a baseline that was no longer the baseline.
DEFAULT_PLAY_MARGIN = UtilityTeacher(DEFAULT_DECK, team=0).play_margin

COMBO_KINDS = ("supported_push", "counter_push", "defensive_stack",
               "cheap_defence", "spell_then_push", "push_then_spell")


def _score(env):
    a, b = env.get_towers_alive(0), env.get_towers_alive(1)
    return 1.0 if a > b else (0.5 if a == b else 0.0)


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
    """Everything a combo run has to report, accumulated across matches.

    `completed` is the number that matters and the only one that cannot be
    faked: a pair is counted only when the SECOND card is the action actually
    taken on the following decision, i.e. the plan survived re-scoring. A combo
    that is proposed, chosen, and then abandoned is a combo that did nothing.
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
        # % of PLAYS, not of decisions. A bot that holds more has fewer
        # decisions that do anything, so a per-decision rate rises for free
        # when spending falls -- which would grade an economy change on the
        # very thing it changes.
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
    plays that side and `t0` is ignored. Returns team 0's score.

    TELEMETRY READS THE TEACHER'S OWN LABEL, `last_kind`, rather than
    re-deriving anything from the returned `(slot, x, y)`. Two reasons, both
    learned by getting it wrong first: a combo's first step is indistinguishable
    from the same card played alone once it is reduced to coordinates, and a
    plan with a five-second gap stays pending for five decisions, so counting
    "a plan exists" per decision inflates `chosen` fivefold.
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


# --------------------------------------------------------------------------
# 1. usage
# --------------------------------------------------------------------------
def run_usage(args):
    print("\nCOMBO USAGE -- teacher vs teacher, both sides identical\n")
    stages = ([args.stage] if args.stage is not None
              else range(len(TEACHER_STAGES)))
    for stage in stages:
        tele = Telemetry()
        for i in range(args.n):
            env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
            env.reset()
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


# --------------------------------------------------------------------------
# 2. the reserve
# --------------------------------------------------------------------------
def run_reserve_ab(args):
    """Sweep `teacher.COMBO_RESERVE`, the charge protecting a committed plan.

    Vary ONLY team 0's reserve; team 1 is held at 0.0 in both arms, and the
    sides are swapped. A policy once beat a bit-exact copy of itself 0.598
    purely by side assignment, so a one-sided duel would fold that straight
    into the result.
    """
    values = [float(v) for v in args.reserve_grid.split(",")]
    print(f"\nCOMBO RESERVE A/B -- stage {args.stage or 5}, "
          f"{args.n} paired openings, sides swapped\n")
    stage = args.stage if args.stage is not None else 5
    arms = {v: [] for v in values}
    teles = {v: Telemetry() for v in values}
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.reset()
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
    """THE SAFETY MEASUREMENT: combos ON vs OFF, same teacher otherwise.

    This is the one that could come back negative and still matter. Combo
    candidates spend rollouts that would otherwise have gone to single cards,
    and a plan holds a hand slot and some elixir for up to five decisions --
    both are real costs, paid on every decision, against a benefit that lands
    on a fraction of a percent of them. If the pair is a net loss the honest
    move is to say so, not to keep the feature because it was the assignment.

    Only team 0's combo width varies; team 1 is held at the stage default in
    both arms, and the sides are swapped, because a policy once beat a
    bit-exact copy of itself 0.598 purely by side assignment.
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
        root.reset()
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


# --------------------------------------------------------------------------
# 2b. the profile sweep -- what unlocks combos
# --------------------------------------------------------------------------
def teacher_vs_heuristic(env, teacher, tele=None):
    """Teacher on team 0 through `env.step()`, so ClashEnv::opponentTurn runs
    the C++ HeuristicOpponent for team 1. Returns the teacher's score.

    Deliberately the same routing `prove_teacher.teacher_vs_heuristic` uses, so
    a number here is comparable with the strength bar recorded there.
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
    """Sweep one weight against BOTH bars the profile has to clear at once.

    WHY THIS IS VALID WITHOUT A SEEDABLE ENGINE. Every arm plays the SAME
    openings, because the root envs are built once and each arm gets a
    `snapshot()` of them -- so the comparison is paired within this one process,
    which is exactly the guarantee `env.snapshot()` was added for. What is NOT
    available is comparing these numbers against a different invocation's; see
    this module's docstring and UPSTREAM_REQUESTS item 7.

    TWO BARS, because moving a weight to unlock combos is trivial if the bot is
    allowed to get worse:

      STRENGTH   two of them, because the heuristic bar SATURATES. Stage 5
                 scores 1.000 against the C++ HeuristicOpponent, so that arm can
                 only ever say "still not broken" -- it cannot rank two profiles
                 that both clear it. The discriminating arm is the swept profile
                 played HEAD TO HEAD against the shipped one, sides swapped, on
                 the same openings: 0.500 means no strength was traded away.
      USAGE      combos as a share of PLAYS -- not of decisions. A bot that
                 holds more has fewer decisions that do anything, so per-decision
                 usage rises for free when spending falls, which would make this
                 sweep grade itself on the very thing it is changing.
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
        env.reset()
        roots.append(env.snapshot())

    results = {}
    for v in values:
        # `play_margin` is a teacher attribute rather than a profile weight,
        # and it is the better-targeted lever of the two -- see the sweep's
        # own write-up. Handled here rather than by pretending it is a weight,
        # because putting it in PROFILES would make it look like one.
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
            # Head to head against the SHIPPED profile, sides swapped -- the
            # arm that can actually rank two profiles that both beat the
            # heuristic. A policy once beat a bit-exact copy of itself 0.598
            # purely by side assignment, so a one-sided duel would fold that
            # straight into the result.
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
        # Share of decisions spent at or above the overflow line. Raising
        # `play_margin` buys elixir by not spending it, and past some point
        # that stops being thrift and starts being wasted income -- the bar
        # caps at 10.0, so regeneration above ~9 is thrown away. A win-rate arm
        # at this n would not necessarily show that cost, so it is reported
        # directly. 9.0 is the same threshold `score`'s overflow relief and
        # W_ELIXIR_OVERFLOW both use.
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


# --------------------------------------------------------------------------
# 3. the strength benchmark
# --------------------------------------------------------------------------
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
            root.reset()
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
