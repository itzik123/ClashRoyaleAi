"""Does the environment price the win condition positively? No network involved.

    python_ai/venv/Scripts/python.exe python_ai/eval/prove_environment.py --n 120

Tests whether an opponent-elixir multiplier suppresses punish cards, without
confounding "the environment prices this badly" with "this net cannot execute
it": both sides are the deterministic UtilityTeacher, and the arms differ only
in what team 0 does with its win condition.

    attack   the win condition goes to a bridge
    cycle    it is still drawn, played, paid for and rotated, but placed in our own back half, where it deals almost no damage

Identical spend, cycle and hand occupancy; only where the card lands differs.
The obvious control, never playing it, is confounded: an unplayed card clogs a
hand slot. `--include-ban` runs it anyway, labelled.

Reading the result:
    attack beats cycle at 1.0x and the gap shrinks as m rises -> hypothesis supported
    no gap at any multiplier -> the win condition is unviable in this engine regardless of economy
    attack beats cycle equally at every m -> the multiplier is not what suppresses it

`--opponent teacher` (default) is the symmetric test at a fair 1.0x; above 1.0x
a mirror-strength opponent wins everything and both arms floor. `--opponent
heuristic` is the dose-response test against the C++ HeuristicOpponent at
multiplier m.

A row with the attack arm >= 0.95, or both arms <= 0.05, is printed as VOID
rather than dropped. Every arm starts from the same snapshot with seeded
teachers, so the comparison is paired.
"""
import argparse
import os
import sys

import numpy as np

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.eval import stats  # noqa: E402
from python_ai.eval.match_outcome import score_from_towers  # noqa: E402
from python_ai.opponents.teacher import TEACHER_STAGES, UtilityTeacher  # noqa: E402

CE = E.ClashRoyaleEnv
MAX_STEPS = 400


def duel(env, t0, t1):
    """Teacher t0 (team 0) against teacher t1 (team 1). Returns t0's score."""
    t0.reset()
    t1.reset()
    for _ in range(MAX_STEPS):
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        if env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10).done:
            break
    return _outcome(env)


def _outcome(env):
    """(score, enemy tower damage we dealt, tower damage we took).

    Tower damage is reported beside the outcome because mirror matches between
    defensive bots mostly reach the tick limit, making win rate a coarse
    readout of a continuous difference.
    """
    score = score_from_towers(env, 0)
    return (score, float(env.get_tower_damage_dealt(0)),
            float(env.get_tower_damage_dealt(1)))


def duel_heuristic(env, t0):
    """Teacher t0 (team 0) against the C++ HeuristicOpponent, which runs inside
    `env.step()`. Returns t0's score.
    """
    t0.reset()
    for _ in range(MAX_STEPS):
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t0.act(env, o0)
        if env.step(slot, x, y, 10).done:
            break
    return _outcome(env)


def arm(mode, stage, seed, base, multiplier, opponent):
    """One episode of `mode` for team 0, from a bit-identical opening."""
    env = base.snapshot()
    env.set_opponent_elixir_multiplier(multiplier)
    t0 = UtilityTeacher(DEFAULT_DECK, team=0, seed=seed, wincon_mode=mode)
    t0.set_stage(stage)
    if opponent == "heuristic":
        return duel_heuristic(env, t0)
    t1 = UtilityTeacher(DEFAULT_DECK, team=1, seed=seed + 7777)
    t1.set_stage(stage)
    return duel(env, t0, t1)


def paired(diffs):
    """(mean, lo, hi, better, worse, p); see eval/stats.py."""
    r = stats.paired_from_diffs(diffs)
    return r.delta, r.lo, r.hi, r.better, r.worse, r.p

def _tower_diff(env, team=0):
    """(enemy tower damage we dealt) - (tower damage we took)."""
    return (float(env.get_tower_damage_dealt(team))
            - float(env.get_tower_damage_dealt(1 - team)))


def _play_out(env, t0, t1, steps):
    """Both teachers keep playing normally for `steps` decisions.

    The opponent must keep playing: the cost of committing a win condition is
    the counter-push that arrives while our half is empty, which a no-op
    rollout never produces.
    """
    for _ in range(steps):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        if env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10).done:
            break


def _escort_slot(env, hand, teacher):
    """(slot, x, y) for the cheapest affordable tank to send ahead of the win
    condition, or None.

    A tank is any unit whose role is not spell, building or win condition
    (roles come from the engine via teacher.roles). Cheapest, since the escort
    should not cost more than the push it protects.
    """
    import clash_royale_env as _E
    elixir = env.get_elixir_for_team(0)
    best = None
    for slot, cid in enumerate(hand):
        role = teacher.roles.get(cid)
        if role in (None, "spell", "building", "wincon"):
            continue
        cost = float(_E.get_card_info(cid)["cost"])
        if cost > elixir + 1e-6:
            continue
        if best is None or cost < best[0]:
            best = (cost, slot, cid)
    if best is None:
        return None
    _, slot, _cid = best
    return slot, None, None   # lane filled in by the caller


def marginal_value(args):
    """At states where the advisor's gate says commit, is committing worth it?

    The strategy-level A/B is confounded (a back-placed win condition is an
    accidental defensive body, and an over-eager rule makes attacking look
    bad); this measures one decision. Paired on identical snapshots at
    gate-selected states:
        arm PLAY   commit the win condition at the advisor's bridge cell
        arm HOLD   no-op this step
    then both teachers play on for `--horizon` decisions, scored by tower-HP
    differential. The difference is the marginal value of that commitment,
    counter-push included.
    """
    from python_ai.advisors import tactics
    # The gate threshold is the variable under test: if the punish window is
    # what pays, a tighter gate should move the verdict.
    if args.max_opp_elixir is not None:
        tactics.HOG_MAX_OPP_ELIXIR = float(args.max_opp_elixir)
    print(f"  gate: commit while estimated opponent elixir <= "
          f"{tactics.HOG_MAX_OPP_ELIXIR}")
    diffs, played, gate_hits, seen = [], 0, 0, 0
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
        root.reset()
        env = root.snapshot()
        t0 = UtilityTeacher(DEFAULT_DECK, team=0, seed=args.seed + i)
        t1 = UtilityTeacher(DEFAULT_DECK, team=1, seed=args.seed + 7777 + i)
        t0.set_stage(args.stage)
        t1.set_stage(args.stage)
        t0.reset()
        t1.reset()

        for _ in range(MAX_STEPS):
            if env.is_game_over():
                break
            seen += 1
            o0 = np.asarray(env.get_observation_for_team(0), np.float32)
            hand = list(env.get_hand_for_team(0))
            wincon = t0.wincon_id
            # Only states where committing is possible and the advisor's timing
            # gate approves.
            if (wincon in hand
                    and env.get_elixir_for_team(0) >= E.get_card_info(wincon)["cost"]
                    and tactics.hog_should_commit(o0)):
                gate_hits += 1
                x, y, _ = tactics.best_hog_cell(o0)
                slot = hand.index(wincon)

                a = env.snapshot()
                b = env.snapshot()
                ta0 = UtilityTeacher(DEFAULT_DECK, team=0, seed=args.seed + i)
                ta1 = UtilityTeacher(DEFAULT_DECK, team=1, seed=args.seed + 7777 + i)
                tb0 = UtilityTeacher(DEFAULT_DECK, team=0, seed=args.seed + i)
                tb1 = UtilityTeacher(DEFAULT_DECK, team=1, seed=args.seed + 7777 + i)
                for t in (ta0, ta1, tb0, tb1):
                    t.set_stage(args.stage)
                    t.reset()

                o1 = np.asarray(env.get_observation_for_team(1), np.float32)
                opp = t1.act(env, o1)

                if args.supported:
                    # A real 2.6 push: the tank goes first to absorb the
                    # building's targeting and the tower shots while the Hog
                    # connects. With deploy time a lone Hog stands inert under
                    # fire for a second.
                    tank = _escort_slot(env, hand, t0)
                    if tank is None:
                        # No affordable escort: not a supported-push state, so
                        # skip it rather than score a lone Hog.
                        o1b = np.asarray(env.get_observation_for_team(1), np.float32)
                        a0 = t0.act(env, o0)
                        a1 = t1.act(env, o1b)
                        if env.step_self_play(a0[0], a0[1], a0[2],
                                              a1[0], a1[1], a1[2], 10).done:
                            break
                        continue
                    tslot, _, _ = tank
                    # Same lane and row, so the tank is in front of the win
                    # condition.
                    tx, ty = x, y
                    a.step_self_play(tslot, tx, ty, opp[0], opp[1], opp[2], 10)
                    b.step_self_play(-1, 0.0, 0.0, opp[0], opp[1], opp[2], 10)
                    # The win condition follows one decision later.
                    hand_a = list(a.get_hand_for_team(0))
                    if wincon in hand_a:
                        a.step_self_play(hand_a.index(wincon), x, y, -1, 0.0, 0.0, 10)
                    b.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 10)
                else:
                    a.step_self_play(slot, x, y, opp[0], opp[1], opp[2], 10)
                    b.step_self_play(-1, 0.0, 0.0, opp[0], opp[1], opp[2], 10)

                _play_out(a, ta0, ta1, args.horizon)
                _play_out(b, tb0, tb1, args.horizon)
                diffs.append(_tower_diff(a) - _tower_diff(b))
                played += 1

            o1 = np.asarray(env.get_observation_for_team(1), np.float32)
            a0 = t0.act(env, o0)
            a1 = t1.act(env, o1)
            if env.step_self_play(a0[0], a0[1], a0[2],
                                  a1[0], a1[1], a1[2], 10).done:
                break
        if played >= args.max_states:
            break

    label = ("SUPPORTED PUSH (tank + win condition)" if args.supported
             else "LONE WIN-CONDITION COMMITMENT")
    print(f"\nMARGINAL VALUE OF ONE {label}")
    print(f"  states scored      {played}  (gate fired on {gate_hits} of "
          f"{seen} decisions)")
    if not diffs:
        print("  the gate never fired -- nothing measured. Not a null result.")
        return
    mean, lo, hi, better, worse, pv = paired(diffs)
    print(f"  tower-HP delta     {mean:>+9.1f}   95% CI "
          f"[{lo:>+8.1f}, {hi:>+8.1f}]")
    print(f"  better/worse       {better}/{worse}   sign test p = {pv:.3}")
    print()
    if lo > 0:
        print("  Committing the win condition is NET POSITIVE in this "
              "environment.")
    elif hi < 0:
        print("  Committing the win condition is NET NEGATIVE even at a "
              "symmetric economy\n  and with perfect timing. The card is not "
              "rehabilitated by the curriculum\n  change, and no amount of "
              "policy pressure will make it pay.")
    else:
        print("  No difference resolved. Either the commitment is roughly "
              "neutral, or n is\n  too small -- check better/worse against "
              "the CI width before reading it as\n  a null.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5])
    ap.add_argument("--opponent", choices=["teacher", "heuristic"],
                    default="teacher")
    ap.add_argument("--include-ban", action="store_true",
                    help="also run the CONFOUNDED never-play arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["winrate", "marginal"], default="winrate")
    ap.add_argument("--horizon", type=int, default=40,
                    help="decisions to play on after the commitment (mode=marginal)")
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--supported", action="store_true",
                    help="mode=marginal: escort the win condition with a tank, "
                         "i.e. the push a real 2.6 deck actually sends")
    ap.add_argument("--max-opp-elixir", type=float, default=None,
                    help="override tactics.HOG_MAX_OPP_ELIXIR for mode=marginal")
    args = ap.parse_args()

    if args.mode == "marginal":
        print(f"\ndeck {DEFAULT_DECK}")
        print(f"teacher stage {args.stage} {TEACHER_STAGES[args.stage]}")
        marginal_value(args)
        return

    modes = ["attack", "cycle"] + (["ban"] if args.include_ban else [])

    print(f"\ndeck {DEFAULT_DECK}")
    print(f"teacher stage {args.stage} {TEACHER_STAGES[args.stage]}")
    print(f"{args.n} paired episodes per multiplier, "
          f"teacher vs {args.opponent}\n")
    print(f"{'opp elixir':>10} {'attack':>8} {'cycle':>8} {'delta':>9} "
          f"{'95% CI':>20} {'better/worse':>13} {'p':>10}")
    print("-" * 84)

    rows = []
    for m in args.multipliers:
        scores = {k: [] for k in modes}
        dealt = {k: [] for k in modes}
        taken = {k: [] for k in modes}
        for i in range(args.n):
            root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), 3600)
            root.reset()
            base = root.snapshot()
            for mode in modes:
                sc, dl, tk = arm(mode, args.stage, args.seed + i,
                                 base, m, args.opponent)
                scores[mode].append(sc)
                dealt[mode].append(dl)
                taken[mode].append(tk)
        a = np.asarray(scores["attack"])
        c = np.asarray(scores["cycle"])
        mean, lo, hi, better, worse, p = paired(a - c)
        ceiling = a.mean() >= 0.95
        floor = a.mean() <= 0.05 and c.mean() <= 0.05
        void = ceiling or floor
        flag = ("  VOID (attack arm at ceiling)" if ceiling else
                "  VOID (both arms at floor)" if floor else "")
        print(f"{m:>10.2f} {a.mean():>8.3f} {c.mean():>8.3f} {mean:>+9.4f} "
              f"  [{lo:>+7.4f}, {hi:>+7.4f}] {better:>6}/{worse:<6} {p:>10.2}"
              f"{flag}")
        rows.append((m, a.mean(), c.mean(), mean, lo, hi, p, void))
        # The continuous readout, which win rate quantises away.
        da, dc = np.asarray(dealt["attack"]), np.asarray(dealt["cycle"])
        ta, tc = np.asarray(taken["attack"]), np.asarray(taken["cycle"])
        dmean, dlo, dhi, dbet, dwor, dp = paired(da - dc)
        print(f"{'':>10} enemy tower dmg  attack {da.mean():>7.1f}  "
              f"cycle {dc.mean():>7.1f}  delta {dmean:>+8.1f} "
              f"[{dlo:>+8.1f}, {dhi:>+8.1f}] {dbet:>4}/{dwor:<4} p={dp:.2}")
        print(f"{'':>10} tower dmg TAKEN  attack {ta.mean():>7.1f}  "
              f"cycle {tc.mean():>7.1f}")
        if "ban" in scores:
            b = np.asarray(scores["ban"])
            bm, blo, bhi, _, _, bp = paired(a - b)
            print(f"{'':>10} {'':>8} ban={b.mean():>6.3f} {bm:>+9.4f} "
                  f"  [{blo:>+7.4f}, {bhi:>+7.4f}] {'':>13} {bp:>10.2}"
                  f"   (CONFOUNDED: a banned card also clogs a hand slot)")

    print("\nREADING IT")
    live = [r for r in rows if not r[7]]
    if not live:
        print("  Every arm is at ceiling. Nothing is measurable here -- rerun at")
        print("  a stage where the attack arm sits nearer 0.7-0.8.")
        return
    lowest, highest = live[0], live[-1]
    print(f"  lowest multiplier tested with a live baseline: {lowest[0]:.2f}x, "
          f"delta {lowest[3]:+.4f}")
    print(f"  highest:                                       {highest[0]:.2f}x, "
          f"delta {highest[3]:+.4f}")
    if lowest[3] > 0 and lowest[4] > 0:
        print("  The win condition PAYS at the lowest live multiplier.")
        if len(live) > 1 and highest[3] < lowest[3]:
            print("  ...and the payoff SHRINKS as the opponent's economy grows, "
                  "which is\n  the hypothesis's central prediction.")
        elif len(live) > 1:
            print("  ...but it does NOT shrink as the economy grows, so the "
                  "multiplier is\n  not what was suppressing it. Do not rebuild "
                  "the curriculum on this.")
    else:
        print("  The win condition does NOT pay even at the lowest live "
              "multiplier.\n  Either the teacher is too weak for its own economy "
              "to matter, or the\n  card is unviable in this engine regardless "
              "of economy.")
    print("\n  HEADROOM NOTE: a baseline near 1.0 has less room to lose, so part "
          "of any\n  shrinkage is compression. Normalise by headroom before "
          "claiming an effect:")
    for m, am, cm, d, lo, hi, p, void in rows:
        if not void and am > 0:
            print(f"    {m:.2f}x: delta {d:+.4f} = {abs(d) / am * 100:5.1f}% "
                  f"of the attack arm's own win rate")


if __name__ == "__main__":
    main()
