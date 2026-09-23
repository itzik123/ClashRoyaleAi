"""Is playing the win condition worth it? Forced usage, paired, engine-scored.

A near-zero play rate fits two explanations with opposite fixes:

  PATHOLOGY   a zero-gradient trap: the win-condition reward pays only for damage the card deals, so an unplayed card is never rewarded. Fix: force exploration.
  VALUATION   the card is a bad buy against this opponent and the policy knows it. Fix: nothing; forcing it makes the agent worse.

Paired on `env.snapshot()` against the C++ heuristic at 1.5x elixir (at 1.0x
both arms saturate). The forced arm changes only which card is played, never
where: it goes to the net's own placement cell, keeping the question about the
card rather than the placement (prove_hog.py measures that). `--force-prob`
below 1.0 makes it an epsilon-exploration probe, the shape a real fix would
take.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

HERE = python_ai.PACKAGE_DIR

import clash_royale_env as CE  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK, WIN_CONDITION_ID  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402

# BOARD_WIDTH lives on the class; the module has no such attribute.
E = CE.ClashRoyaleEnv

NOOP = 4


def load(path):
    net = MicroRoyaleNet(num_ability_slots=0)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"] if "model" in blob else blob, strict=False)
    net.eval()
    return net


@torch.no_grad()
def play(net, env, force_prob, rng, stats, advisor_cell=False,
         hog_legal=None, smart_force=False, gate_mult=1.0):
    """One episode. Returns 1.0 win / 0.0 loss / 0.5 draw."""
    obs_l = env.get_observation_for_team(0)
    hx = torch.zeros(1, LSTM_HIDDEN)
    cx = torch.zeros(1, LSTM_HIDDEN)
    for _ in range(400):
        obs = torch.from_numpy(np.asarray(obs_l, np.float32)).view(1, -1)
        feats, embeds, spatial, hires = net.extract_features_hires(obs)
        mask = net.affordability_mask(obs)
        cl, _, _, _, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)

        card = int(cl.argmax(1))
        ids = net.hand_card_ids(obs)[0].tolist()
        forced_now = False
        smart_cell = None

        if smart_force and WIN_CONDITION_ID in ids:
            # Smart force: triggered by the advisor's timing gate rather than a
            # coin. A random epsilon fires mid-defence, when the elixir is
            # needed for the answer, and charges the loss to the card instead
            # of the moment.
            #
            # hog_advice returns the bridge cell when its conditions hold (our
            # half clear, solvent, opponent not banked), so this forces when
            # and where together. gate_mult is the opponent's elixir
            # multiplier: the gate reconstructs their bar from income, which
            # scales with it.
            advice = tactics.hog_advice(np.asarray(obs_l, np.float32),
                                        legal=hog_legal,
                                        multiplier=gate_mult)
            slot = ids.index(WIN_CONDITION_ID)
            if advice is not None and bool(mask[0, slot]):
                card = slot
                forced_now = True
                smart_cell = advice
                stats["forced"] += 1

        if (not forced_now) and force_prob > 0.0 and WIN_CONDITION_ID in ids:
            slot = ids.index(WIN_CONDITION_ID)
            # Only force what the engine accepts: an unaffordable slot makes
            # playCard fail silently.
            if bool(mask[0, slot]) and rng.random() < force_prob:
                card = slot
                forced_now = True
                stats["forced"] += 1

        if smart_cell is not None:
            px, py = float(smart_cell[0]), float(smart_cell[1])
            stats["advisor_placed"] += 1
        elif forced_now and advisor_cell:
            # Bypass the placement head, separating "the card is unviable" from
            # "the head has not learned where to put it". best_hog_cell rather
            # than hog_advice, since timing is forced externally here.
            ax, ay, _rank = tactics.best_hog_cell(
                np.asarray(obs_l, np.float32), legal=hog_legal)
            px, py = float(ax), float(ay)
            stats["advisor_placed"] += 1
        else:
            pl = net.placement_given_card(hx2, embeds, torch.tensor([card]),
                                          obs, spatial, hires_map=hires)
            c = int(pl.argmax(1))
            px, py = float(c % E.BOARD_WIDTH), float(c // E.BOARD_WIDTH)
        if card != NOOP:
            stats["plays"] += 1
            played_id = net.hand_card_ids(obs)[0].tolist()[card]
            stats["by_card"][played_id] = stats["by_card"].get(played_id, 0) + 1
        res = env.step(card, px, py, 10)
        obs_l = res.observation
        hx, cx = hx2, cx2
        if res.done:
            break

    stats["wincon_damage"] += float(env.get_damage_dealt_by_card(WIN_CONDITION_ID, 0))
    towers = env.get_towers_destroyed(0) if hasattr(env, "get_towers_destroyed") else None
    del towers
    r = float(res.reward)
    return 1.0 if r > 0 else (0.0 if r < 0 else 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join(HERE, "model_weights_selfplay.pth"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--force-prob", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smart-force", action="store_true",
                    help="arm triggered by the advisor TIMING gate rather "
                         "than a coin: forces WHEN and WHERE together")
    ap.add_argument("--advisor-cell", action="store_true",
                    help="third arm: force the Hog AND place it at the "
                         "advisor bridge cell, bypassing the head")
    args = ap.parse_args()

    print(f"win condition = card id {WIN_CONDITION_ID}")
    net = load(args.weights)
    root = CE.ClashRoyaleEnv(DEFAULT_DECK, DEFAULT_DECK, 3600)
    root.set_opponent_elixir_multiplier(args.opp_elixir)

    hog_legal = net._placement_legal[WIN_CONDITION_ID].numpy().astype(bool)

    # (force_prob, use_advisor_cell). All arms share one set of snapshot
    # openings.
    arms = {"baseline": (0.0, False),
            f"forced@{args.force_prob} net-cell": (args.force_prob, False)}
    if args.advisor_cell:
        arms[f"forced@{args.force_prob} ADVISOR-cell"] = (args.force_prob, True)
    if args.smart_force:
        arms["SMART-forced (gate+bridge)"] = ("smart", True)

    scores = {k: [] for k in arms}
    stats = {k: {"forced": 0, "plays": 0, "wincon_damage": 0.0,
                 "advisor_placed": 0, "by_card": {}} for k in arms}

    for i in range(args.n):
        root.reset()
        base = root.snapshot()
        for name, (prob, use_adv) in arms.items():
            # Same RNG stream per pair, so the forced arms draw their coin at
            # the same steps.
            rng = random.Random(args.seed * 100003 + i)
            smart = (prob == "smart")
            scores[name].append(play(net, base.snapshot(),
                                     0.0 if smart else prob, rng,
                                     stats[name], advisor_cell=use_adv,
                                     hog_legal=hog_legal, smart_force=smart,
                                     gate_mult=args.opp_elixir))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n}", flush=True)

    a = list(arms)[0]
    A = np.array(scores[a])
    print("\n" + "=" * 74)
    print(f"FORCED WIN-CONDITION USAGE  --  {os.path.basename(args.weights)}, "
          f"opp {args.opp_elixir}x, n={args.n} paired")
    print("=" * 74)
    for name in arms:
        s = stats[name]
        wc = s["wincon_damage"] / max(args.n, 1)
        hog = s["by_card"].get(WIN_CONDITION_ID, 0)
        print(f"  {name:<16} win {np.mean(scores[name]):.3f}   "
              f"plays/ep {s['plays']/args.n:5.1f}   "
              f"hog plays {hog:4d} ({100*hog/max(s['plays'],1):4.1f}%)   "
              f"wincon dmg/ep {wc:7.1f}   "
              f"adv-placed {s['advisor_placed']:4d}")

    from math import comb
    for name in list(arms)[1:]:
        B = np.array(scores[name])
        diff = B - A
        delta = float(diff.mean())
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        better = int((diff > 0).sum())
        worse = int((diff < 0).sum())
        print("")
        print(f"  {name} - baseline   {delta:+.4f}  "
              f"95% CI [{delta-1.96*se:+.4f}, {delta+1.96*se:+.4f}]")
        print(f"    {better} better / {worse} worse / "
              f"{len(diff)-better-worse} tied")
        if better + worse:
            n = better + worse
            kk = min(better, worse)
            pv = sum(comb(n, j) for j in range(kk + 1)) * 2 / (2 ** n)
            print(f"    exact sign test p = {min(pv,1.0):.4g}")

    print("\n  A NEGATIVE delta means the policy was RIGHT to avoid the card and")
    print("  no exploration fix is warranted. A positive one means it is stuck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
