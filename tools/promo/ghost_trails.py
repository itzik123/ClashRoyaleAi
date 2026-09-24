"""Ghost trails: the futures the lookahead search looked at, drawn on the board.

Plays one match with the network plus 1-ply search (python_ai/search/search.py,
the search behind "win rate 0.625 -> 0.944") against the built-in C++ bot at
1.5x elixir, the condition that number was measured in. At each decision the
search forks the game (env.snapshot()), plays every candidate move 4 seconds
ahead, scores each future with the network's critic, and picks the best.

For a few of those decisions this re-plays every candidate on its own
snapshot, records where the new units go, and renders a clip:

  1. the real match up to the decision;
  2. the board freezes and each candidate's future grows as a coloured trail
     from its placement, ending in the critic's score for it;
  3. the chosen future lights up, the rest fade, and the real match plays on
     with the move search actually made.

    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py
    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --seed 4 --clips 3
    python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --still

Clips land in tools/promo/out/ghost_trails/. The engine is deterministic, so a
re-played candidate is the same future the search scored (checked per clip).

The search runs as it was MEASURED (2026-08-11): greedy plus the top 3 cards'
top 2 cells, at most 7 futures. The widened proposals added on 2026-09-03 (a
48-cell sweep for any card whose placement head is flat) put ~97 futures on
one board, which is neither readable nor the search the win rate belongs to;
--wide-proposals shows them anyway.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time

from common import (BG, OUT_DIR, REPO_ROOT, TEAM, BoardPainter, Image, ImageDraw,
                    Replay, VideoWriter, composite, ease_in_out, engine, font,
                    load_arena, parse_size)

SKIP = 10
CLIP_DIR = OUT_DIR / "ghost_trails"
# Candidate colours: none is a team colour, all read on grass and water.
COLORS = [(250, 204, 21), (34, 211, 238), (244, 114, 182), (163, 230, 53),
          (251, 146, 60), (255, 255, 255), (167, 139, 250), (45, 212, 191)]


# --- the match ------------------------------------------------------------------
def play_match(args, workdir):
    """One searched match. Returns (replay path, [decision]), where a decision
    holds the snapshot taken before the search and what the search did."""
    E = engine()
    import numpy as np
    import torch
    from python_ai.envs.gym_wrapper import DEFAULT_DECK
    from python_ai.models.policy_io import LSTM_HIDDEN, load_net
    from python_ai.search.config import SearchCfg
    from python_ai.search.search import greedy_from_logits, policy_head, search_action

    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    weights = args.weights if os.path.isabs(args.weights) else \
        str(REPO_ROOT / "python_ai" / args.weights)
    net = load_net(weights, device)
    cfg = SearchCfg()

    env = E.ClashRoyaleEnv(list(DEFAULT_DECK), list(DEFAULT_DECK), args.max_ticks)
    env.set_opponent_elixir_multiplier(args.opp_elixir)
    env.seed(args.seed)
    hidden = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    obs = env.get_observation_for_team(0)
    decisions, done, steps, reward = [], False, 0, 0.0
    with torch.no_grad():
        while not done and steps < cfg.max_steps:
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            logits, embeds, spatial, _, hidden_next = policy_head(net, obs_t, hidden)
            greedy = greedy_from_logits(net, obs_t, logits, embeds, spatial, hidden_next)[:3]
            snap = env.snapshot()
            hand = list(env.get_hand())
            action, deviated, n, details = search_action(
                net, env, obs_t, logits, embeds, spatial, hidden_next, greedy, cfg,
                device, return_details=True)
            if details is not None:
                cands, scores = details
                decisions.append({
                    "tick": env.get_current_tick(), "snap": snap, "hand": hand,
                    "cands": [tuple(c) for c in cands],
                    "scores": [float(s) for s in scores],
                    "chosen": [tuple(c) for c in cands].index(tuple(action)),
                    "deviated": bool(deviated), "horizon": cfg.horizon})
            hidden = hidden_next
            r = env.step(action[0], action[1], action[2], SKIP)
            obs, reward, done = r.observation, float(r.reward), r.done
            steps += 1
    path = os.path.join(workdir, "match.json")
    env.save_log(path)
    outcome = "won" if reward > 0.5 else "lost" if reward < -0.5 else "drew"
    print(f"match: {steps} decisions, {len(decisions)} with a choice to make, "
          f"{sum(d['deviated'] for d in decisions)} where search overrode the "
          f"network; the searched agent {outcome}")
    return path, decisions


def futures(decision, workdir, tag):
    """Re-play each candidate on its own copy of the pre-search snapshot and
    return one Replay of its future per candidate."""
    from python_ai.search.search import rollout
    out = []
    for i, (card, x, y) in enumerate(decision["cands"]):
        sim = decision["snap"].snapshot()
        rollout(sim, card, x, y, decision["horizon"])
        path = os.path.join(workdir, f"{tag}_cand{i}.json")
        sim.save_log(path)
        out.append(Replay(path))
    # Determinism check: a second copy of candidate 0 must land identically,
    # or the trails would not be the futures that were scored.
    sim = decision["snap"].snapshot()
    card, x, y = decision["cands"][0]
    rollout(sim, card, x, y, decision["horizon"])
    check = os.path.join(workdir, f"{tag}_check.json")
    sim.save_log(check)
    again = Replay(check)
    if again.ticks[-1] != out[0].ticks[-1]:
        raise SystemExit("a re-played future differs from the first: the engine is "
                         "not deterministic from a snapshot, so the trails would lie")
    return out


def describe(decision, main):
    """Per candidate: name, placement, and the trail of each unit it adds."""
    E = engine()
    before = {e[0] for e in main.ticks[min(decision["tick"], len(main) - 1)]}
    rows = []
    for (card, x, y), fut in zip(decision["cands"], decision["futures"]):
        if card >= len(decision["hand"]):
            rows.append({"name": "Wait", "wait": True, "trails": [], "at": None})
            continue
        cid = decision["hand"][card]
        name = E.get_card_info(cid)["name"]
        paths = {}
        for tick in fut.ticks:
            for eid, ex, ey, _ in tick:
                b = fut.bodies[eid]
                if eid not in before and b.team == 0 and b.kind in ("troop", "building", "spell"):
                    paths.setdefault(eid, []).append((ex, ey))
        rows.append({"name": name, "wait": False, "at": (x, y), "trails": list(paths.values()),
                     "bodies": {eid: fut.bodies[eid] for eid in paths}})
    return rows


def pick(decisions, main, n, gap=80):
    """The decisions worth a clip: search overrode the network, and at least
    two candidates put a moving unit on the board. Spread-out in time."""
    scored = []
    for d in decisions:
        moving = sum(1 for c in d["cands"] if c[0] < len(d["hand"]))
        if len(d["cands"]) < 3 or moving < 2:
            continue
        spread = max(d["scores"]) - min(d["scores"])
        scored.append(((d["deviated"], moving, spread), d))
    scored.sort(key=lambda s: s[0], reverse=True)
    chosen = []
    for _, d in scored:
        if 60 <= d["tick"] <= len(main) - 60 and all(abs(d["tick"] - c["tick"]) >= gap for c in chosen):
            chosen.append(d)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda d: d["tick"])


# --- drawing --------------------------------------------------------------------
def score_text(s):
    if s >= 5:
        return "WIN"
    if s <= -5:
        return "LOSS"
    return f"{s:+.2f}"


def draw_futures(frame, painter, ox, oy, t, rows, decision, grow, reveal, focus,
                 only=None):
    """Overlay every candidate's future (or just candidate `only`).

    grow   0..1: how much of each 4-second future is drawn
    reveal 0..1: the score labels fading in
    focus  0..1: the chosen future brightening, the others fading back
    """
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    chosen = decision["chosen"]
    best = max(range(len(rows)), key=lambda i: decision["scores"][i])
    f_score = font(t * 0.62)
    f_small = font(t * 0.46)
    wait_slot = 0
    placed = []        # label boxes already drawn, for collision nudging
    named = []         # (name, x, y) of placement tags already drawn

    def free_box(box):
        """Shift `box` down until it overlaps no earlier label."""
        l, tp, r, b = box
        for _ in range(8):
            hit = next((p for p in placed if l < p[2] and r > p[0] and tp < p[3] and b > p[1]), None)
            if hit is None:
                break
            dy = hit[3] - tp + t * 0.08
            tp, b = tp + dy, b + dy
        placed.append((l, tp, r, b))
        return tp - box[1]

    order = sorted(range(len(rows)), key=lambda i: i == chosen)  # chosen on top
    for i in order:
        if only is not None and i != only:
            continue
        row, col = rows[i], COLORS[i % len(COLORS)]
        dim = 1.0 - 0.72 * focus if i != chosen else 1.0
        a = lambda v: max(0, min(255, round(v * dim)))
        width = max(2, round(t * (0.16 if i != chosen else 0.16 + 0.1 * focus)))
        label = score_text(decision["scores"][i]) + ("  BEST" if i == chosen and focus > 0.5 else "")

        if row["wait"]:
            # No unit to trail: a badge on the blue half's back line.
            x, y = painter.to_px(ox, oy, t, 1.2, 3.0 + 1.6 * wait_slot)
            wait_slot += 1
            text = f"Wait  {label}" if reveal > 0 else "Wait"
            l, tp, r, b = d.textbbox((x, y), text, font=f_score)
            pad = t * 0.25
            d.rounded_rectangle((l - pad, tp - pad, r + pad, b + pad), radius=pad * 2,
                                fill=(11, 11, 12, a(200 * max(grow, 0.3))),
                                outline=col + (a(255),), width=width)
            d.text((x, y), text, font=f_score, fill=col + (a(255),))
            continue

        px, py = painter.to_px(ox, oy, t, *row["at"])
        ring = t * 0.62
        d.ellipse((px - ring, py - ring, px + ring, py + ring), outline=col + (a(255),),
                  width=width, fill=(11, 11, 12, a(110)))
        d.text((px, py), str(i + 1), font=f_small, fill=col + (a(255),), anchor="mm")
        # One name tag per card per spot: two cells of the same card sit side
        # by side and their tags would print over each other.
        if i == chosen or not any(n == row["name"] and math.hypot(px - nx, py - ny) < t * 2.5
                                  for n, nx, ny in named):
            named.append((row["name"], px, py))
            tag = d.textbbox((px + ring * 1.25, py), row["name"], font=f_small, anchor="lm")
            dy = free_box(tag)
            d.text((px + ring * 1.25, py + dy), row["name"], font=f_small, anchor="lm",
                   fill=(255, 255, 255, a(230)), stroke_width=max(1, round(t * 0.05)),
                   stroke_fill=(0, 0, 0, a(200)))
        head = None
        for path in row["trails"]:
            n = max(2, round(len(path) * grow))
            pts = [painter.to_px(ox, oy, t, x, y) for x, y in path[:n]]
            for k in range(1, len(pts)):
                fade = 0.35 + 0.65 * k / max(1, len(pts) - 1)
                d.line((pts[k - 1], pts[k]), fill=col + (a(230 * fade),), width=width)
            if pts:
                hx, hy = pts[-1]
                r = max(t * 0.3, t * 0.45)
                d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=col + (a(150),),
                          outline=(255, 255, 255, a(220)), width=max(1, width // 2))
                if head is None or hy < head[1]:
                    head = (hx, hy)
        if reveal > 0 and head is not None:
            hx, hy = head
            text = label
            pad = t * 0.18
            l, tp, r, b = d.textbbox((hx + t * 0.7, hy), text, font=f_score, anchor="lm")
            dy = free_box((l - pad, tp - pad, r + pad, b + pad))
            d.rounded_rectangle((l - pad, tp - pad + dy, r + pad, b + pad + dy), radius=pad * 2,
                                fill=(11, 11, 12, a(210 * reveal)))
            d.text((hx + t * 0.7, hy + dy), text, font=f_score, anchor="lm",
                   fill=col + (a(255 * reveal),))
    frame.alpha_composite(layer)


def pill(frame, text, W, H):
    f = font(round(min(W, H) * 0.028))
    d = ImageDraw.Draw(frame, "RGBA")
    x, y = round(W * 0.05), round(H * 0.11)
    l, t, r, b = d.textbbox((x, y), text, font=f)
    pad = round(f.size * 0.45)
    d.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=pad * 2,
                        fill=(11, 11, 12, 200), outline=(255, 255, 255, 60), width=2)
    d.text((x, y), text, font=f, fill=(245, 245, 245, 255))


# --- the clip -------------------------------------------------------------------
def render_clip(main, decision, rows, out, args, painter, still=False):
    W, H = parse_size(args.size)
    SS = 2
    bw, bh = painter.size_tiles
    t = min(W * 0.96 / bw, H * 0.96 / bh) * SS
    ox, oy = (W * SS - bw * t) / 2, (H * SS - bh * t) / 2
    fps, T = args.fps, decision["tick"]
    n_opts = len(rows)
    title = f"Lookahead: {n_opts} options × {decision['horizon']} s"

    lead = round(args.lead * fps)        # the real match, up to the decision
    grow = round(1.8 * fps)              # futures grow
    reveal = round(0.7 * fps)            # scores appear
    focus = round(0.9 * fps)             # the chosen one lights up
    hold = round(0.8 * fps)
    follow = round(args.follow * fps)    # the real match plays on
    per_tick = 10.0 / fps                # 1x real time

    def board(q):
        img = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
        size = (round(bw * t), round(bh * t))
        img.paste(painter.background(t, size), (round(ox), round(oy)))
        painter.draw(img, round(ox), round(oy), t, main, main.at(q))
        return img

    frames = []
    for f in range(lead):
        frames.append(("play", T - (lead - f) * per_tick, None))
    think = grow + reveal + focus + hold
    for f in range(think):
        g = ease_in_out(f / grow) if f < grow else 1.0
        rv = ease_in_out((f - grow) / reveal) if f >= grow else 0.0
        fc = ease_in_out((f - grow - reveal) / focus) if f >= grow + reveal else 0.0
        frames.append(("think", T, (g, rv, fc)))
    for f in range(follow):
        frames.append(("follow", T + f * per_tick, (1.0, 1.0, 1.0, 1.0 - min(1.0, f / fps))))

    frozen = None

    def render(kind, q, p):
        nonlocal frozen
        if kind == "think":
            if frozen is None:
                frozen = board(T)
                shade = Image.new("RGBA", frozen.size, (0, 0, 0, 70))
                frozen.alpha_composite(shade)
            img = frozen.copy()
            draw_futures(img, painter, round(ox), round(oy), t, rows, decision, *p)
        else:
            img = board(q)
            if kind == "follow" and p[3] > 0:
                # The chosen future stays up briefly, fading, as the real one
                # plays out beside it.
                ghost = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw_futures(ghost, painter, round(ox), round(oy), t, rows, decision,
                             1.0, 1.0, 1.0, only=decision["chosen"])
                alpha = ghost.getchannel("A").point(lambda v: round(v * p[3] * 0.6))
                ghost.putalpha(alpha)
                img.alpha_composite(ghost)
        out_img = img.reduce(SS).convert("RGB")
        played = rows[decision["chosen"]]["name"]
        pill(out_img, {"play": "Real match", "think": title,
                       "follow": f"Played: {played}"}[kind], W, H)
        return out_img

    if still:
        path = os.path.splitext(out)[0] + ".png"
        render(*frames[lead + grow + reveal + focus]).save(path)
        return path
    with VideoWriter(out, (W, H), fps, args.ffmpeg, args.crf) as vid:
        for kind, q, p in frames:
            vid.write(render(kind, q, p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="model_weights_final_ep83128.pth",
                    help="checkpoint in python_ai/ (or an absolute path)")
    ap.add_argument("--seed", type=int, default=0, help="which match to play")
    ap.add_argument("--opp-elixir", type=float, default=1.5,
                    help="the built-in bot's elixir multiplier (1.5 = the "
                         "condition the 0.625 -> 0.944 result was measured in)")
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--clips", type=int, default=3, help="decisions to render")
    ap.add_argument("--size", default="vertical")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lead", type=float, default=2.0,
                    help="seconds of real match before each decision")
    ap.add_argument("--follow", type=float, default=3.0,
                    help="seconds of real match after it")
    ap.add_argument("--still", action="store_true",
                    help="one PNG per clip at the moment the choice lights up")
    ap.add_argument("--wide-proposals", action="store_true",
                    help="use today's widened candidate sweep (~97 futures) "
                         "instead of the measured top-k search (<= 7)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    if not args.wide_proposals:
        # Read by python_ai/search/config.py at import, which play_match does
        # lazily, so this lands before it. 0 means no head is ever "flat".
        os.environ["CLASH_WIDE_PROPOSAL_TOP1"] = "0"

    workdir = tempfile.mkdtemp(prefix="ghost-trails-")
    try:
        t0 = time.time()
        path, decisions = play_match(args, workdir)
        main_replay = Replay(path)
        chosen = pick(decisions, main_replay, args.clips)
        if not chosen:
            raise SystemExit("no decision in this match had two moving candidates "
                             "to compare; try another --seed")
        painter = BoardPainter(load_arena())
        os.makedirs(CLIP_DIR, exist_ok=True)
        print(f"played in {time.time() - t0:.0f}s; rendering {len(chosen)} clip(s)")
        for k, d in enumerate(chosen, 1):
            d["futures"] = futures(d, workdir, f"d{k}")
            rows = describe(d, main_replay)
            out = str(CLIP_DIR / f"seed{args.seed}_clip{k}_tick{d['tick']}.mp4")
            opts = ", ".join(f"{i + 1}:{r['name']} {score_text(s)}"
                             for i, (r, s) in enumerate(zip(rows, d["scores"])))
            print(f"  tick {d['tick']}: network wanted #1, search chose "
                  f"#{d['chosen'] + 1} ({rows[d['chosen']]['name']}) | {opts}")
            written = render_clip(main_replay, d, rows, out, args, painter, still=args.still)
            print(f"  wrote {written}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
