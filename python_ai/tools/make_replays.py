"""Generate PAIRED replay logs for the viewer: original vs distilled, same opening.

Writes replays in the format `web/viewer.html` already reads (drop the JSON on
the page), annotated per tick with what the network chose and what its critic
thought the position was worth -- the same `annotate_replay_with_agent_info`
train.py uses, so the viewer shows the same overlays it always has.

WHY PAIRED
----------
Watching one agent play tells you very little: a single episode is one draw from
a distribution whose control arm alone measured 0.625 / 0.570 / 0.700 / 0.634
across four runs. Two agents playing the SAME opening is what makes a visual
comparison mean anything -- env.snapshot() hands both arms a bit-identical
start (same shuffled hand, same heuristic-opponent lane), so any divergence you
see on screen is the policy and not the deal.

Each opening produces two files with the same index:

    replay_<i>_A_original.json    model_weights_selfplay.pth
    replay_<i>_B_distilled.json   whatever --distilled points at

Scrub them side by side and the first tick where the two boards differ is the
first decision the distillation changed.

WHAT THESE REPLAYS ARE, AND ARE NOT
-----------------------------------
They are the C++ simulator playing against the built-in HeuristicOpponent at
1.5x elixir -- exactly the setting every measured number in CLAUDE.md came from.
They are NOT the live game, and nothing here is evidence about sim-to-real
transfer.

Also: the measured effect is +0.045 win rate. That is ~1 game in 22, so most
paired replays will look IDENTICAL or differ only in small ways. If you watch
three pairs and see no obvious difference, that is the expected outcome, not a
bug -- the effect is real but small, and a handful of episodes cannot show it.
Use `--n 10` and the printed divergence summary to find the pairs worth watching.

Run:
    python_ai/venv/Scripts/python.exe python_ai/make_replays.py --n 5
"""
import argparse
import time
from collections import Counter
import json
import os
import sys

import numpy as np
import torch

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.envs.gym_wrapper import DEFAULT_DECK  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402
from python_ai.models.policy_io import load_net, LSTM_HIDDEN  # noqa: E402
from python_ai.rl.replay import annotate_replay_with_agent_info  # noqa: E402
# These four were USED by play_and_log but never imported, so this script died
# with `NameError: LSTM_HIDDEN` on its first episode and could not be run at
# all. Lost in the 2026-08-20 package restructuring, the same way
# trainers/bc_pretrain.py lost its script bootstrap; found 2026-08-21 while
# adding --teacher-debug.
from python_ai.search.search import (  # noqa: E402
    policy_head, greedy_from_logits, search_action, outcome_score,
)
from python_ai.opponents.teacher import (  # noqa: E402
    TEACHER_STAGES, UtilityTeacher,
)
from python_ai.rl.teacher_debug import (  # noqa: E402
    CapturingTeacher, attach_teacher_debug,
)

CE = clash_royale_env.ClashRoyaleEnv
SKIP = 10  # matches rl.replay.REPLAY_SKIP_FRAMES; 1 decision -> 10 ticks


@torch.no_grad()
def play_and_log(net, env, device, path, cfg=None, use_search=False, max_steps=400):
    """Play one episode to completion, save the replay, annotate it with decisions."""
    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))
    obs = env.get_observation_for_team(0)
    decisions, plays = [], []
    reward, steps, done = 0.0, 0, False

    while not done and steps < max_steps:
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        card_logits, card_embeds, spatial_map, value, hidden_next = policy_head(net, obs_t, hidden)
        greedy = greedy_from_logits(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next)
        action = (greedy[0], greedy[1], greedy[2])
        if use_search:
            action, _, _ = search_action(net, env, obs_t, card_logits, card_embeds,
                                          spatial_map, hidden_next, action, cfg, device)

        # The hand is read BEFORE stepping: playCard consumes the slot and cycles
        # a new card in, so reading it after would name the replacement.
        hand = list(env.get_hand())
        idx = action[0]
        card_id = hand[idx] if idx < len(hand) else -1
        decisions.append({"stateValue": float(value.item()),
                          "actionCardId": int(card_id),
                          "actionX": float(action[1]),
                          "actionY": float(action[2])})
        if card_id != -1:
            plays.append((steps, int(card_id), round(action[1], 1), round(action[2], 1)))

        hidden = hidden_next
        result = env.step(action[0], action[1], action[2], SKIP)
        obs, reward, done = result.observation, float(result.reward), result.done
        steps += 1

    env.save_log(path)
    annotate_replay_with_agent_info(path, decisions, SKIP)
    return reward, steps, plays


@torch.no_grad()
def play_and_log_vs_teacher(net, env, device, path, stage=None, top_k=4,
                            seed=None, max_steps=400):
    """One episode of `net` against the UtilityTeacher, with the teacher's
    candidate rollouts recorded into the replay for the viewer.

    `stage` defaults to the TOP rung, derived rather than written as a literal:
    it was `stage=5`, which meant the final rung against the six-rung teacher
    table and means the middle of the eleven-rung one -- so the default would
    have silently started recording replays against a 4-second teacher instead
    of a 10-second one.

    A SEPARATE FUNCTION FROM play_and_log, because the opponent is genuinely
    different plumbing rather than a parameter: play_and_log calls env.step(),
    which runs the C++ HeuristicOpponent internally, and there is no teacher in
    that path to record. Here team 1's move comes from the teacher and both
    sides go through step_self_play.

    The teacher's (x, y) is passed through UNCONVERTED. It returns coordinates
    in its own mirrored frame and stepSelfPlay mirrors y back itself
    (realY1 = BOARD_HEIGHT - 1 - y1); converting here would double-mirror and
    put every opponent placement in its own back corner.
    """
    teacher = CapturingTeacher(list(DEFAULT_DECK), team=1, seed=seed, top_k=top_k)
    if stage is None:
        stage = len(TEACHER_STAGES) - 1
    teacher.set_stage(stage)
    teacher.reset()

    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))
    obs = env.get_observation_for_team(0)
    decisions, plays = [], []
    reward, steps, done = 0.0, 0, False

    while not done and steps < max_steps:
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        card_logits, card_embeds, spatial_map, value, hidden_next = policy_head(net, obs_t, hidden)
        action = greedy_from_logits(net, obs_t, card_logits, card_embeds,
                                    spatial_map, hidden_next)

        hand = list(env.get_hand())
        idx = action[0]
        card_id = hand[idx] if idx < len(hand) else -1
        decisions.append({"stateValue": float(value.item()),
                          "actionCardId": int(card_id),
                          "actionX": float(action[1]),
                          "actionY": float(action[2])})
        if card_id != -1:
            plays.append((steps, int(card_id), round(action[1], 1), round(action[2], 1)))

        obs1 = np.asarray(env.get_observation_for_team(1), dtype=np.float32)
        slot1, x1, y1 = teacher.act(env, obs1)

        hidden = hidden_next
        r = env.step_self_play(action[0], action[1], action[2],
                               slot1, x1, y1, SKIP)
        obs, reward, done = r.observation0, float(r.reward0), bool(r.done)
        steps += 1

    env.save_log(path)
    annotate_replay_with_agent_info(path, decisions, SKIP)
    attach_teacher_debug(path, teacher.debug_records, SKIP)
    return reward, steps, plays, teacher


def _run_teacher_debug(args, net, device, outdir):
    """--teacher-debug: one replay per opening, teacher reasoning recorded.

    Deliberately NOT paired. The paired A/B above exists to compare two nets
    against a fixed opponent; this mode exists to watch ONE net against an
    opponent whose reasoning is visible, which is a different question and a
    different artifact.
    """
    print(f"\nopponent : UtilityTeacher @ stage {args.teacher_stage}")
    print(f"net      : {args.original} (greedy)")
    print(f"capture  : top {args.teacher_top_k} candidates get a predicted board")
    print(f"out      : {outdir}\n")

    for i in range(args.n):
        env = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), args.max_ticks)
        env.seed(i)
        env.reset()
        path = os.path.join(outdir, f"replay_{i}_teacher_debug.json")
        t0 = time.time()
        reward, steps, plays, teacher = play_and_log_vs_teacher(
            net, env, device, path, stage=args.teacher_stage,
            top_k=args.teacher_top_k, seed=i)
        recs = teacher.debug_records
        kinds = Counter(r["kind"] for r in recs)
        boards = sum(1 for r in recs for c in r["candidates"] if "final" in c)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  replay {i}: {steps} decisions, {len(plays)} plays by the net | "
              f"teacher {dict(kinds)} | {boards} predicted boards | "
              f"{size_mb:.1f} MB | {time.time() - t0:.0f}s")

    print(f"\nOpen web/viewer.html, drop a JSON from {args.outdir} onto it,")
    print("and press T (or click the brain chip) for the Simulation View.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5, help="paired openings")
    ap.add_argument("--original", default="model_weights_selfplay.pth")
    ap.add_argument("--distilled", default="model_weights_selfplay.pth")
    ap.add_argument("--outdir", default="replays_e3")
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--search", action="store_true",
                    help="arm B uses 1-ply search instead of the distilled net")
    ap.add_argument("--teacher-debug", action="store_true",
                    help="play --original against the UtilityTeacher instead of "
                         "the heuristic, and record the teacher's candidate "
                         "rollouts into the replay for the viewer's Simulation "
                         "View. Not paired: one replay per opening.")
    ap.add_argument("--teacher-stage", type=int,
                    default=len(TEACHER_STAGES) - 1,
                    help="teacher competence rung for --teacher-debug (0-5)")
    ap.add_argument("--teacher-top-k", type=int, default=4,
                    help="candidates per decision that get a predicted board. "
                         "Each costs ~32 ms; the rest are recorded score-only.")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    here = python_ai.PACKAGE_DIR
    outdir = os.path.join(here, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    net_a = load_net(os.path.join(here, args.original), device)
    net_b = net_a if args.search else load_net(os.path.join(here, args.distilled), device)
    cfg = SearchCfg()
    label_b = "search" if args.search else "distilled"

    if args.teacher_debug:
        _run_teacher_debug(args, net_a, device, outdir)
        return

    print(f"\nopponent : HeuristicOpponent at {args.opp_elixir}x elixir")
    print(f"arm A    : {args.original} (greedy)")
    print(f"arm B    : {'same net + 1-ply search' if args.search else args.distilled + ' (greedy)'}")
    print(f"out      : {outdir}\n")

    rows = []
    for i in range(args.n):
        root = CE(list(DEFAULT_DECK), list(DEFAULT_DECK), args.max_ticks)
        root.set_opponent_elixir_multiplier(args.opp_elixir)
        root.reset()
        base = root.snapshot()  # the shared opening both arms play

        pa = os.path.join(outdir, f"replay_{i}_A_original.json")
        pb = os.path.join(outdir, f"replay_{i}_B_{label_b}.json")
        ra, sa, plays_a = play_and_log(net_a, base.snapshot(), device, pa)
        rb, sb, plays_b = play_and_log(net_b, base.snapshot(), device, pb,
                                       cfg=cfg, use_search=args.search)

        # First decision index where the two arms chose differently -- the tick
        # to scrub to if you want to see what actually changed.
        first_div = None
        for k in range(min(len(plays_a), len(plays_b))):
            if plays_a[k] != plays_b[k]:
                first_div = plays_a[k][0] * SKIP
                break
        rows.append((i, outcome_score(ra), outcome_score(rb), len(plays_a), len(plays_b), first_div))
        print(f"  pair {i}: A {outcome_score(ra):.1f} ({len(plays_a)} plays) | "
              f"B {outcome_score(rb):.1f} ({len(plays_b)} plays) | "
              f"first divergence @ tick {first_div if first_div is not None else '-- (identical)'}")

    a_mean = float(np.mean([r[1] for r in rows]))
    b_mean = float(np.mean([r[2] for r in rows]))
    print(f"\n  arm A won {a_mean:.2f}, arm B won {b_mean:.2f} over {len(rows)} pairs")
    print("  (n is far too small to mean anything -- the measured effect is +0.045,")
    print("   i.e. ~1 game in 22. This is for WATCHING, not for measuring.)")
    print(f"\nOpen web/viewer.html and drop a JSON from {args.outdir} onto it.")
    diverged = [r for r in rows if r[5] is not None]
    if diverged:
        print("Pairs where the policies actually diverged (watch these):")
        for r in diverged:
            print(f"  pair {r[0]} -- scrub to tick ~{r[5]}")
    else:
        print("No pair diverged. Re-run with a larger --n; at a 13.8% deviation rate")
        print("most openings play out identically.")


if __name__ == "__main__":
    main()
