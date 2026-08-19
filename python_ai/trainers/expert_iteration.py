"""Expert iteration: distil 1-ply decision-time search back into the policy.

THE MEASUREMENT THIS FOLLOWS FROM
---------------------------------
`search_ab_test.py` measured, over 160 paired trials at 1.5x opponent elixir:

    greedy policy 0.625  ->  policy + 1-ply search 0.944
    paired delta +0.319, 95% CI [+0.237, +0.401], exact McNemar p = 5.6e-12
    deviation rate 13.8%, cost 2.2x wall clock

Search is scored by *this same network's critic*, and the greedy action is
always candidate 0, so search only deviates when the critic disagrees with the
action head. Beating the network's own action selection by 32 points therefore
says the VALUE head is much better than the POLICY head is at exploiting it --
the underfit is in the policy head. That is exactly the gap expert iteration
closes: search is a policy-improvement operator, and its output is a free
supply of expert labels for the head that is behind.

WHAT SUCCESS LOOKS LIKE, AND WHAT IT DOES NOT
---------------------------------------------
The point of distillation is to stop paying the 2.2x search cost, so the
metric is DISTILLED GREEDY vs ORIGINAL GREEDY, paired, both without search.
"Distilled + search beats original" would be a different and much weaker claim.

The band to read the result against is fixed by the numbers above: the original
greedy policy sits at 0.625 and the expert it is imitating sits at 0.944, so a
perfect distillation lands at 0.944 and no distillation lands at 0.625.

THE TRAP THIS FILE IS BUILT AROUND
----------------------------------
86.2% of expert labels are IDENTICAL to what the policy already does (that is
what a 13.8% deviation rate means). So a network that learned nothing at all
already scores ~0.862 card-match on this dataset. Reporting "card match 0.88"
would look like success and mean nothing.

    THE NULL FOR action_match_rate IS THE ORIGINAL NET'S OWN SCORE ON THE
    SAME DATA, NOT ZERO.

`--train` therefore always evaluates BOTH nets on the same rows and prints them
side by side. This is the same correction CLAUDE.md already records for
placement (always-guess-the-modal-cell scores 0.273, so cell-match must be read
against 0.273 and not against 1/612).

WHY THE TRUNK IS FROZEN BY DEFAULT
----------------------------------
The measured finding is specifically that the POLICY HEAD is underfit relative
to the critic. Freezing the trunk/LSTM/value/aux and training only the action
heads tests that hypothesis directly, and preserves bit-exactly the critic that
makes the expert work in the first place -- a full fine-tune moves the shared
features the critic reads, so it can silently degrade the very thing that
generated the labels. `--full-finetune` runs the other arm; both report critic
drift, which must be exactly 0.0 under the default.

Run:
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --collect 120
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --train
    python_ai/venv/Scripts/python.exe python_ai/expert_iteration.py --eval --trials 120
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.trainers import bc_pretrain  # noqa: E402
from python_ai.trainers.bc_pretrain import action_match_rate, train_bc  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN, load_net  # noqa: E402
from python_ai.eval.stats import report_paired_winrate  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.search.search import (  # noqa: E402
    greedy_from_logits, outcome_score, play_episode, policy_head,
    search_action,
)


from python_ai.trainers.expert_collect import (  # noqa: E402
    K_MAX, collect_expert_labels, make_env, merge_datasets,
    verify_cell_roundtrip,
)
from python_ai.trainers.expert_distill import (  # noqa: E402
    POLICY_MODULES, TRUNK_MODULES, TRUNK_PARAMS, candidate_target,
    freeze_trunk, train_distribution,
)
from python_ai.trainers.expert_metrics import (  # noqa: E402
    conditional_lift, conditional_match_rate, critic_drift,
    deviation_breakdown, disagreement_weights, modal_cell_baseline, noop_rate,
)

CE = clash_royale_env.ClashRoyaleEnv

# --------------------------------------------------------------------------
# stage 3: paired greedy-vs-greedy evaluation
# --------------------------------------------------------------------------

def paired_greedy_ab(net_a, net_b, trials, cfg, device, opp_elixir, max_ticks,
                     label_a="original", label_b="distilled", time_budget=0.0):
    """Both nets play GREEDILY from a bit-identical opening. No search anywhere.

    This is the measurement that decides whether expert iteration worked: if
    distillation only helps when search is also running, it has not bought the
    thing it exists to buy.
    """
    diffs, a_scores, b_scores = [], [], []
    started = time.perf_counter()

    for trial in range(trials):
        if time_budget and (time.perf_counter() - started) > time_budget:
            print(f"\n  [time budget reached after {trial} trials]")
            break
        root = make_env(opp_elixir, max_ticks)
        base = root.snapshot()

        r_a, _, _, _ = play_episode(net_a, base.snapshot(), device, False, cfg)
        r_b, _, _, _ = play_episode(net_b, base.snapshot(), device, False, cfg)

        sa, sb = outcome_score(r_a), outcome_score(r_b)
        a_scores.append(sa)
        b_scores.append(sb)
        diffs.append(sb - sa)

        if (trial + 1) % 10 == 0:
            n = len(diffs)
            print(f"  trial {trial + 1:4d} | {label_a} {np.mean(a_scores):.3f} "
                  f"{label_b} {np.mean(b_scores):.3f} | delta {np.mean(diffs):+.3f} "
                  f"| {(time.perf_counter() - started) / n:.1f}s/trial")

    return np.asarray(diffs), np.asarray(a_scores), np.asarray(b_scores)

# --------------------------------------------------------------------------

def run_ablation(args, cfg, device, resolve):
    """Which lever, if any, forces the policy to learn the CONDITIONAL rule.

    Four configurations over one shared dataset and one shared train/held-out
    split, so nothing between them is confounded:

        A  frozen trunk,   w=1     the configuration that already measured
                                   +0.016 [-0.030, +0.061] on win rate
        B  frozen trunk,   w=F     lever 1 -- upweight disagreement rows
        C  full finetune,  w=1     lever 2 -- capacity
        D  full finetune,  w=F     both

    Deliberately NO win-rate evaluation here. The 800-trial run established that
    this comparison needs ~6,700 paired trials to resolve effects of the size on
    offer, so a per-config win rate at any affordable n would be noise wearing a
    number. These are learning-dynamics metrics only, used to decide where the
    big compute goes.

    The held-out split is what makes the numbers mean anything: unfreezing the
    trunk adds ~118x the trainable parameters, so it will fit the training rows
    better whether or not it generalises. Train and held-out are reported side
    by side and the gap IS the result for lever 2.
    """
    data = bc_pretrain.load_dataset(resolve(args.data))
    extras = np.load(resolve(args.data))
    greedy_card = extras["greedy_card"]

    ep_ids = np.unique(data["episode"])
    n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
    held = set(int(e) for e in ep_ids[-n_hold:])
    train_eps = [int(e) for e in ep_ids if int(e) not in held]
    print(f"dataset : {len(data['card'])} rows, {len(ep_ids)} episodes")
    print(f"split   : {len(train_eps)} train / {len(held)} held-out episodes")
    dis_all = int((data["card"] != greedy_card).sum())
    print(f"disagreement rows: {dis_all} ({dis_all / len(data['card']):.2%})")
    print(f"balancing weight : {(len(data['card']) - dis_all) / max(1, dis_all):.1f} "
          f"(using --upweight {args.upweight})\n")

    original = load_net(resolve(args.weights), device, verbose=False)
    base_tr = conditional_match_rate(original, data, greedy_card, device, train_eps)
    base_ho = conditional_match_rate(original, data, greedy_card, device, held)
    print(f"ORIGINAL net (the null):")
    print(f"  train    disagree {base_tr['disagreement_match']:.4f} "
          f"agree {base_tr['agreement_match']:.4f} noop {base_tr['pred_noop_rate']:.3f}")
    print(f"  held-out disagree {base_ho['disagreement_match']:.4f} "
          f"agree {base_ho['agreement_match']:.4f} noop {base_ho['pred_noop_rate']:.3f}")
    print(f"  expert no-op rate {noop_rate(data['card']):.3f}, "
          f"greedy {noop_rate(greedy_card):.3f}\n")

    configs = [
        ("A frozen  w=1", False, None),
        ("B frozen  w=%g" % args.upweight, False, args.upweight),
        ("C full    w=1", True, None),
        ("D full    w=%g" % args.upweight, True, args.upweight),
    ]
    rows = []
    for name, full_ft, w in configs:
        print("=" * 70)
        print(f"config {name}")
        student = load_net(resolve(args.weights), device, verbose=False)
        if full_ft:
            n_train = sum(p.numel() for p in student.parameters())
            n_froz = 0
        else:
            n_train, n_froz = freeze_trunk(student)
        weights = disagreement_weights(data, greedy_card, w)
        print(f"  {n_train:,} trainable / {n_froz:,} frozen; "
              f"weighting {'off' if weights is None else f'{w}x on disagreements'}")
        t0 = time.perf_counter()
        student.train()
        student, _ = train_bc(data, net=student, epochs=args.epochs, lr=args.lr,
                              batch_episodes=args.batch_episodes, device=device,
                              placement_weight=args.placement_weight, verbose=True,
                              sample_weights=weights, episode_filter=train_eps)
        student.eval()
        tr = conditional_match_rate(student, data, greedy_card, device, train_eps)
        ho = conditional_match_rate(student, data, greedy_card, device, held)
        vd, ad = critic_drift(original, student, data["obs"], device)
        took = time.perf_counter() - t0
        rows.append((name, tr, ho, vd, ad, took))
        print(f"  train    disagree {tr['disagreement_match']:.4f} "
              f"agree {tr['agreement_match']:.4f} noop {tr['pred_noop_rate']:.3f}")
        print(f"  held-out disagree {ho['disagreement_match']:.4f} "
              f"agree {ho['agreement_match']:.4f} noop {ho['pred_noop_rate']:.3f}")
        print(f"  critic drift |dV| {vd:.6f}  aux {ad:.6f}   ({took / 60:.1f} min)")
        torch.save({"model": student.state_dict(), "config": name},
                   resolve(f"exit_ablate_{name.split()[0]}.pth"))

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY  (held-out is the one that matters)")
    print("=" * 78)
    print(f"{'config':16s} {'HO disagree':>12s} {'HO agree':>10s} {'HO noop':>9s} "
          f"{'TR disagree':>12s} {'critic dV':>10s}")
    print(f"{'-- null --':16s} {base_ho['disagreement_match']:12.4f} "
          f"{base_ho['agreement_match']:10.4f} {base_ho['pred_noop_rate']:9.3f} "
          f"{base_tr['disagreement_match']:12.4f} {0.0:10.6f}")
    for name, tr, ho, vd, _ad, _t in rows:
        print(f"{name:16s} {ho['disagreement_match']:12.4f} "
              f"{ho['agreement_match']:10.4f} {ho['pred_noop_rate']:9.3f} "
              f"{tr['disagreement_match']:12.4f} {vd:10.6f}")
    print(f"\nexpert no-op rate {noop_rate(data['card']):.3f} -- a config whose "
          f"HO noop climbs toward it")
    print("without HO disagree moving learned the MARGINAL, not the conditional.")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect", type=int, metavar="N_EPISODES",
                    help="generate expert labels by playing with search on")
    ap.add_argument("--train", action="store_true", help="distil --data into --weights")
    ap.add_argument("--eval", action="store_true", help="paired greedy A/B, original vs distilled")
    ap.add_argument("--data", default="exit_labels.npz")
    ap.add_argument("--weights", default="model_weights_selfplay.pth")
    ap.add_argument("--out", default="model_weights_exit.pth")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--placement-weight", type=float, default=1.0)
    ap.add_argument("--full-finetune", action="store_true",
                    help="train the trunk too (default freezes it; see module docstring)")
    ap.add_argument("--train-dist", action="store_true",
                    help="AlphaZero-style distillation of the candidate value distribution")
    ap.add_argument("--target-entropy", action="store_true",
                    help="report target entropy vs temperature; pick T before measuring outcomes")
    # 0.05, not the 0.25 first guessed. Chosen from --target-entropy on the
    # collected labels BEFORE any outcome was measured: candidate value spread
    # is mean 0.221 / median 0.187, at which 0.25 puts the target at 94% of
    # maximum entropy (near-uniform, no signal) while 0.05 puts it at ~50% --
    # the midpoint between a hard label and no information.
    ap.add_argument("--temperature", type=float, default=0.05,
                    help="softmax temperature on candidate values (see --target-entropy)")
    ap.add_argument("--lift", action="store_true",
                    help="conditional-lift analysis over the saved ablation checkpoints")
    ap.add_argument("--ablate", action="store_true",
                    help="grid over {frozen,full} x {w=1,w=upweight}; learning dynamics only")
    ap.add_argument("--upweight", type=float, default=8.0,
                    help="loss weight on rows where the expert overrode greedy "
                         "(~8.6 balances the two populations)")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of EPISODES held out of training for the ablation")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--terminal-weight", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--time-budget", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    here = python_ai.PACKAGE_DIR

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    cfg = SearchCfg(args.horizon, args.k_cards, args.k_cells,
                    args.terminal_weight, args.max_steps)

    if args.collect:
        print("=== stage 1: collecting expert labels ===")
        net = load_net(resolve(args.weights), device)
        print(f"  search      : K<={1 + args.k_cards * args.k_cells} candidates, "
              f"horizon={args.horizon} steps")
        print(f"  opponent    : HeuristicOpponent at {args.opp_elixir}x elixir")
        data, meta = collect_expert_labels(
            net, args.collect, cfg, device, args.opp_elixir, args.max_ticks,
            args.time_budget)
        bc_pretrain.save_dataset(data, resolve(args.data))
        mb = data["obs"].nbytes / 2 ** 20
        print(f"\n  wrote {len(data['card'])} rows from {meta['episodes']} episodes "
              f"to {args.data} ({mb:.0f} MB uncompressed)")
        print(f"  expert win rate  : {meta['expert_win_rate']:.4f}")
        print(f"  deviation rate   : {meta['deviation_rate']:.2%}")
        print(f"  expert no-op rate: {noop_rate(data['card']):.2%} "
              f"(greedy {noop_rate(data['greedy_card']):.2%})")

    if args.train:
        print("\n=== stage 2: distilling search into the policy ===")
        data = bc_pretrain.load_dataset(resolve(args.data))
        extras = np.load(resolve(args.data))
        greedy_card = extras["greedy_card"] if "greedy_card" in extras else None

        original = load_net(resolve(args.weights), device)
        student = load_net(resolve(args.weights), device, verbose=False)

        if args.full_finetune:
            print("  mode: FULL FINE-TUNE -- the trunk moves, so the critic that")
            print("        generated these labels can drift underneath them.")
            trainable = sum(p.numel() for p in student.parameters())
            frozen = 0
        else:
            trainable, frozen = freeze_trunk(student)
            print(f"  mode: FROZEN TRUNK -- action heads only")
        print(f"  parameters: {trainable:,} trainable / {frozen:,} frozen")

        # The null, computed BEFORE training and printed next to the result.
        base_match = action_match_rate(original, data, device, limit_episodes=10)
        modal, n_played = modal_cell_baseline(data)
        agree = float((data["card"] == greedy_card).mean()) if greedy_card is not None else float("nan")
        print(f"\n  dataset: {len(data['card'])} rows, {len(np.unique(data['episode']))} episodes")
        print(f"    expert agrees with greedy on : {agree:.2%} of rows")
        print(f"    modal-cell baseline          : {modal:.3f} ({n_played} played rows)")
        bd = deviation_breakdown(extras)
        if bd:
            print(f"    what search CHANGED:")
            print(f"      different card   : {bd['card_diff']:6d} ({bd['card_diff'] / bd['n']:.2%})")
            print(f"        expert waits where greedy plays: {bd['expert_waits_greedy_plays']}")
            print(f"        expert plays where greedy waits: {bd['expert_plays_greedy_waits']}")
            print(f"      same card, cell  : {bd['cell_diff']:6d} ({bd['cell_diff'] / bd['n']:.2%})")
            print(f"      placement head trains on {bd['expert_plays']} rows "
                  f"({bd['expert_plays'] / bd['n']:.1%}); card head on all {bd['n']}")
        print(f"  NULL (original net, untrained on these labels):")
        print(f"    card_match_decisions {base_match['card_match_decisions']:.4f}  "
              f"cell_match {base_match['cell_match']:.4f}")

        student.train()
        student, hist = train_bc(data, net=student, epochs=args.epochs, lr=args.lr,
                                 batch_episodes=args.batch_episodes, device=device,
                                 placement_weight=args.placement_weight)
        student.eval()

        new_match = action_match_rate(student, data, device, limit_episodes=10)
        v_drift, aux_drift = critic_drift(original, student, data["obs"], device)
        print(f"\n  AFTER distillation:")
        print(f"    card_match_decisions {new_match['card_match_decisions']:.4f}  "
              f"(null {base_match['card_match_decisions']:.4f}, "
              f"delta {new_match['card_match_decisions'] - base_match['card_match_decisions']:+.4f})")
        print(f"    cell_match           {new_match['cell_match']:.4f}  "
              f"(null {base_match['cell_match']:.4f}, modal {modal:.3f}, "
              f"delta {new_match['cell_match'] - base_match['cell_match']:+.4f})")
        print(f"    critic drift |dV|    {v_drift:.6f}   aux |d| {aux_drift:.6f}")
        if not args.full_finetune and (v_drift != 0.0 or aux_drift != 0.0):
            print("    !! NONZERO under --freeze-trunk. The freeze did not take; the critic")
            print("       that generated these labels is moving underneath the experiment.")
        torch.save({"model": student.state_dict(),
                    "distilled_from": os.path.basename(args.weights),
                    "labels": os.path.basename(args.data),
                    "frozen_trunk": not args.full_finetune}, resolve(args.out))
        print(f"  wrote {args.out}")

    if args.eval:
        print("\n=== stage 3: paired greedy A/B, original vs distilled ===")
        print("  NO SEARCH in either arm -- the point of distillation is to stop")
        print("  paying for it, so search-on would measure the wrong thing.")
        net_a = load_net(resolve(args.weights), device)
        net_b = load_net(resolve(args.out), device)
        print(f"  opponent : HeuristicOpponent at {args.opp_elixir}x elixir")
        print(f"  reference: original greedy 0.625 -> search expert 0.944 "
              f"(search_ab_test, n=160)\n")
        diffs, a, b = paired_greedy_ab(net_a, net_b, args.trials, cfg, device,
                                       args.opp_elixir, args.max_ticks,
                                       time_budget=args.time_budget)
        report_paired_winrate(diffs, a, b, "original greedy", "distilled greedy")

    if args.ablate:
        run_ablation(args, cfg, device, resolve)

    if args.target_entropy:
        print("=== target-distribution entropy vs temperature ===")
        print("Chosen BEFORE any outcome is measured. A target at ~0 nats is the")
        print("argmax label that already failed; at ~log(K) it carries no signal.\n")
        z = np.load(resolve(args.data))
        cv, cn = z["cand_value"], z["cand_n"]
        live = np.where(cn >= 2)[0]
        print(f"rows with >=2 candidates: {len(live)} of {len(cn)} "
              f"({len(live) / max(1, len(cn)):.1%})")
        spread = np.array([cv[r][:cn[r]].max() - cv[r][:cn[r]].min() for r in live])
        print(f"candidate value spread  : mean {spread.mean():.4f} "
              f"median {np.median(spread):.4f}\n")
        print(f"{'T':>8s} {'mean entropy':>14s} {'max possible':>14s} {'frac of max':>12s}")
        for T in (0.05, 0.1, 0.25, 0.5, 1.0):
            ents, maxes = [], []
            for r in live[:4000]:
                n = int(cn[r])
                q = candidate_target(cv[r], n, T)
                ents.append(float(-(q * np.log(q + 1e-12)).sum()))
                maxes.append(math.log(n))
            print(f"{T:8.2f} {np.mean(ents):14.4f} {np.mean(maxes):14.4f} "
                  f"{np.mean(ents) / max(1e-9, np.mean(maxes)):12.3f}")

    if args.train_dist:
        print("\n=== distribution distillation (AlphaZero-style) ===")
        paths = [resolve(p.strip()) for p in args.data.split(",") if p.strip()]
        print("  datasets:")
        data = merge_datasets(paths)
        # Same guard load_dataset applies: a dataset recorded against a different
        # observation layout must not be silently reinterpreted.
        expected = make_env(1.0, 100).observation_size()
        if data["obs"].shape[1] != expected:
            raise SystemExit(f"observation size {data['obs'].shape[1]} != engine's {expected}")
        greedy_card = data["greedy_card"]

        ep_ids = np.unique(data["episode"])
        n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
        held = [int(e) for e in ep_ids[-n_hold:]]
        train_eps = [int(e) for e in ep_ids if int(e) not in held]

        original = load_net(resolve(args.weights), device, verbose=False)
        student = load_net(resolve(args.weights), device, verbose=False)
        if args.full_finetune:
            n_tr, n_fz = sum(p.numel() for p in student.parameters()), 0
        else:
            n_tr, n_fz = freeze_trunk(student)
        print(f"  {n_tr:,} trainable / {n_fz:,} frozen, T={args.temperature}, "
              f"{len(train_eps)} train / {len(held)} held-out episodes")

        base = conditional_lift(original, data, greedy_card, device, held)
        print(f"  NULL lift {base['lift']:+.4f} +/- {1.96 * base['se']:.4f}\n")

        student, _ = train_distribution(
            data, student, device, epochs=args.epochs, lr=args.lr,
            batch_episodes=args.batch_episodes, temperature=args.temperature,
            episode_filter=train_eps)
        student.eval()

        lift = conditional_lift(student, data, greedy_card, device, held)
        cond = conditional_match_rate(student, data, greedy_card, device, held)
        vd, ad = critic_drift(original, student, data["obs"], device)
        ci = 1.96 * lift["se"]
        print(f"\n  CONDITIONAL LIFT {lift['lift']:+.4f} +/- {ci:.4f}  "
              f"({'CONDITIONAL' if lift['lift'] - ci > 0 else 'not significant'})")
        print(f"    p1 (expert waited) {lift['p1_expert_waited']:.4f}  "
              f"p0 (expert played) {lift['p0_expert_played']:.4f}")
        print(f"  held-out disagree {cond['disagreement_match']:.4f}  "
              f"agree {cond['agreement_match']:.4f}  noop {cond['pred_noop_rate']:.3f} "
              f"(expert {noop_rate(data['card']):.3f})")
        print(f"  critic drift |dV| {vd:.6f}  aux {ad:.6f}")
        torch.save({"model": student.state_dict(), "temperature": args.temperature},
                   resolve(args.out))
        print(f"  wrote {args.out}")

    if args.lift:
        print("=== conditional lift: aimed restraint vs indiscriminate no-op drift ===")
        data = bc_pretrain.load_dataset(resolve(args.data))
        greedy_card = np.load(resolve(args.data))["greedy_card"]
        ep_ids = np.unique(data["episode"])
        n_hold = max(1, int(round(len(ep_ids) * args.holdout_frac)))
        held = [int(e) for e in ep_ids[-n_hold:]]

        nets = [("-- null --", resolve(args.weights))]
        for tag in ("A", "B", "C", "D"):
            p = resolve(f"exit_ablate_{tag}.pth")
            if os.path.exists(p):
                nets.append((f"{tag}", p))
        print(f"held-out episodes: {len(held)}   "
              f"expert no-op rate {noop_rate(data['card']):.3f}\n")
        print(f"{'config':10s} {'p1 (exp waited)':>16s} {'p0 (exp played)':>16s} "
              f"{'lift':>9s} {'+/-1.96se':>10s} {'verdict':>22s}")
        for tag, path in nets:
            net = load_net(path, device, verbose=False)
            r = conditional_lift(net, data, greedy_card, device, held)
            ci = 1.96 * r["se"]
            verdict = "CONDITIONAL" if r["lift"] - ci > 0 else "indiscriminate/none"
            print(f"{tag:10s} {r['p1_expert_waited']:16.4f} {r['p0_expert_played']:16.4f} "
                  f"{r['lift']:+9.4f} {ci:10.4f} {verdict:>22s}")
        print(f"\nn1={r['n1']} rows where greedy played and the expert waited; "
              f"n0={r['n0']} where both played.")
        print("lift ~ 0 means the extra waiting is untargeted -- the policy learned")
        print("the expert's MARGINAL restraint, not the state-dependent rule.")

    if not (args.collect or args.train or args.eval or args.ablate or args.lift
            or args.train_dist or args.target_entropy):
        ap.print_help()


if __name__ == "__main__":
    main()
