"""64 Clash Royale matches at once, as one MP4: the promo video's hook shot.

Plays N matches in the engine, each side piloted by the UtilityTeacher with a
real meta deck from python_ai/opponents/decks/meta_decks.json, then draws them
as a grid. By default the clip opens full-screen on the busiest match and pulls
back to reveal the rest.

    python_ai/venv/Scripts/python.exe tools/promo/mosaic.py
    python_ai/venv/Scripts/python.exe tools/promo/mosaic.py --size landscape
    python_ai/venv/Scripts/python.exe tools/promo/mosaic.py --still 4

The matches are cached in tools/promo/out/mosaic_matches/, so re-rendering with
a different camera, speed or size skips the simulation. --still writes one PNG
and needs no ffmpeg.

These are scripted teacher-vs-teacher matches, not the neural agent: the shot
shows the engine, not the AI's strength.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import (BG, OUT_DIR, BoardPainter, Image, Replay, VideoWriter,
                    ease_in_out, engine, load_arena, parse_size)

SKIP = 10            # ticks per decision, as in training
HOLD_TICKS = 15      # the finished board stays up this long before the next match
FLASH_TICKS = 30     # the winner's colour fades from the border over this many
GAP_TILES = 0.5      # space between boards, in tiles


# --- simulation --------------------------------------------------------------
def _play(job):
    """One teacher-vs-teacher match, saved as a replay. Runs in a worker."""
    path, deck0, deck1, rung, seed, max_ticks = job
    import numpy as np
    E = engine()
    from python_ai.opponents.teacher import UtilityTeacher

    env = E.ClashRoyaleEnv(list(deck0), list(deck1), max_ticks)
    env.seed(seed)
    teachers = []
    for team, deck in ((0, deck0), (1, deck1)):
        t = UtilityTeacher(list(deck), team=team, seed=seed * 2 + team)
        t.set_stage(rung)
        t.reset()
        teachers.append(t)
    for _ in range(max_ticks // SKIP + 10):
        # Each teacher answers in its own frame; step_self_play mirrors team
        # 1's y itself.
        a0 = teachers[0].act(env, np.asarray(env.get_observation_for_team(0), np.float32))
        a1 = teachers[1].act(env, np.asarray(env.get_observation_for_team(1), np.float32))
        if env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], SKIP).done:
            break
    env.save_log(path)
    return path


def simulate(args, count):
    """`count` replays in args.cache. Each match is keyed by everything that
    shapes it, so a cached one is reused and only new or changed ones run."""
    E = engine()
    from python_ai.opponents import deck_pool
    from python_ai.opponents.teacher import TEACHER_STAGES

    rung = len(TEACHER_STAGES) - 1 if args.rung is None else args.rung
    pool = deck_pool.load_pool()
    rng = random.Random(args.seed)
    pairs = [rng.sample(pool, 2) for _ in range(count)]
    pyd = E.__file__  # a rebuilt engine invalidates every match
    stamp = [os.path.getsize(pyd), int(os.path.getmtime(pyd))]
    os.makedirs(args.cache, exist_ok=True)
    paths = [os.path.join(args.cache, f"match_{i:03d}.json") for i in range(count)]
    keys = [[rung, args.seed, i, a.name, b.name, args.max_ticks, stamp]
            for i, (a, b) in enumerate(pairs)]
    mpath = os.path.join(args.cache, "manifest.json")
    try:
        with open(mpath, "r", encoding="utf-8") as fh:
            cached = json.load(fh).get("matches", {})
    except (OSError, ValueError, AttributeError):
        cached = {}

    todo = [i for i in range(count)
            if args.resimulate or cached.get(os.path.basename(paths[i])) != keys[i]
            or not os.path.exists(paths[i])]
    if todo:
        jobs = [(paths[i], pairs[i][0].card_ids, pairs[i][1].card_ids, rung,
                 args.seed * 1000 + i, args.max_ticks) for i in todo]
        workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
        print(f"simulating {len(todo)} matches at teacher rung {rung} on "
              f"{workers} workers ({count - len(todo)} cached)...")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_play, j) for j in jobs]):
                fut.result()
                done += 1
                if done % 8 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} ({time.time() - t0:.0f}s)")
        for i in todo:
            cached[os.path.basename(paths[i])] = keys[i]
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump({"matches": cached}, fh, indent=1)
    else:
        print(f"reusing {count} cached matches in {args.cache}")
    return paths, pairs, rung


# --- layout and camera -------------------------------------------------------
def grid(n, W, H, bw, bh):
    """(tile_px, cols, rows): the grid of n boards that makes boards biggest."""
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        t = min(W / (cols * bw + (cols + 1) * GAP_TILES),
                H / (rows * bh + (rows + 1) * GAP_TILES))
        if best is None or t > best[0]:
            best = (t, cols, rows)
    return best


def board_rects(n, W, H, bw, bh):
    t, cols, rows = grid(n, W, H, bw, bh)
    gap = GAP_TILES * t
    x0 = (W - cols * bw * t - (cols - 1) * gap) / 2
    y0 = (H - rows * bh * t - (rows - 1) * gap) / 2
    rects = [(x0 + (k % cols) * (bw * t + gap), y0 + (k // cols) * (bh * t + gap),
              bw * t, bh * t) for k in range(n)]
    return t, cols, rows, rects


def focus_viewport(rect, W, H, fill=0.92):
    """The viewport (x, y, w, h), in final-frame pixels, filled by one board."""
    x, y, w, h = rect
    vh = max(h / fill, (w / fill) * H / W)
    vw = vh * W / H
    return (x + w / 2 - vw / 2, y + h / 2 - vh / 2, vw, vh)


def camera(u, vp0, vp1):
    """Viewport between vp0 (u=0) and vp1 (u=1): the width moves geometrically
    and the centre in step with it, so the pull-back is a pure zoom about one
    fixed point rather than a zoom plus a pan."""
    if u <= 0:
        return vp0
    if u >= 1:
        return vp1
    vw = math.exp(math.log(vp0[2]) + (math.log(vp1[2]) - math.log(vp0[2])) * u)
    k = (vw - vp0[2]) / (vp1[2] - vp0[2]) if vp1[2] != vp0[2] else u
    c0 = (vp0[0] + vp0[2] / 2, vp0[1] + vp0[3] / 2)
    c1 = (vp1[0] + vp1[2] / 2, vp1[1] + vp1[3] / 2)
    cx, cy = c0[0] + (c1[0] - c0[0]) * k, c0[1] + (c1[1] - c0[1]) * k
    vh = vw * vp0[3] / vp0[2]
    return (cx - vw / 2, cy - vh / 2, vw, vh)


# --- a slot: one board showing a queue of matches back to back ----------------
class Slot:
    def __init__(self, replays, start):
        self.replays = replays
        self.start = start
        # Where each match's tick 0 falls on the slot's clock. The first is
        # entered mid-match, at `start`.
        self.starts, s = [], -float(start)
        for r in replays:
            self.starts.append(s)
            s += len(r) - 1 + HOLD_TICKS

    def at(self, c):
        """(replay, fractional tick, flash strength, flash team) at clock c."""
        j = 0
        while j + 1 < len(self.replays) and c >= self.starts[j + 1]:
            j += 1
        r = self.replays[j]
        q = c - self.starts[j]
        flash, team = 0.0, None
        ended = [(self.starts[i] + len(self.replays[i]) - 1, self.replays[i].winner)
                 for i in range(j + 1)]
        end, winner = ended[-1] if q >= len(r) - 1 else (ended[-2] if j else (None, None))
        if end is not None and winner is not None and 0 <= c - end < FLASH_TICKS:
            flash, team = 1.0 - (c - end) / FLASH_TICKS, winner
        return r, min(q, len(r) - 1.0), flash, team

    def shows_jitter(self, c_end):
        """True if a unit vibrates in place anywhere this slot shows during
        clock [0, c_end] (Replay.jitter_spans)."""
        for r, s in zip(self.replays, self.starts):
            lo, hi = max(0.0, -s), min(len(r) - 1.0, c_end - s)
            if hi < lo:
                continue
            if any(a <= hi and b >= lo for a, b in r.jitter_spans()):
                return True
        return False

    def activity(self, c0, span, step=5, fights=False):
        """Bodies on the board summed over clock [c0, c0 + span), first match.

        fights=True scores the smaller side instead, so a board where both
        teams are committed beats one side walking in unopposed.
        """
        first = self.replays[0]
        total = 0
        for d in range(0, span, step):
            t = int(min(self.start + c0 + d, len(first) - 1))
            total += (min(first.units(t, 0), first.units(t, 1)) if fights
                      else first.units(t))
        return total


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=64, help="boards in the grid")
    ap.add_argument("--queue", type=int, default=2,
                    help="matches per board: when one ends the next starts, "
                         "so no board freezes")
    ap.add_argument("--spares", type=int, default=16,
                    help="extra boards' worth of matches, swapped in for any "
                         "board where a unit is stuck vibrating at a bridge "
                         "mouth (an engine defect, UPSTREAM_REQUESTS item 31)")
    ap.add_argument("--rung", type=int, default=None,
                    help="teacher difficulty 0-10 (default: the top rung)")
    ap.add_argument("--seed", type=int, default=0,
                    help="change for a different set of matches")
    ap.add_argument("--max-ticks", type=int, default=3600)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resimulate", action="store_true",
                    help="ignore the cached matches")
    ap.add_argument("--cache", default=str(OUT_DIR / "mosaic_matches"))

    ap.add_argument("--size", default="vertical",
                    help="vertical (1080x1920, Shorts), landscape (1920x1080), "
                         "square (1080x1080), or WxH")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=8.0, help="clip length")
    ap.add_argument("--speed", type=float, default=4.0,
                    help="game seconds per video second at the start")
    ap.add_argument("--speed-end", type=float, default=None,
                    help="ramp to this speed after the zoom (default: no ramp)")
    ap.add_argument("--start", default="auto",
                    help="match tick the clip starts at, or 'auto' for the "
                         "busiest window")
    ap.add_argument("--no-zoom", action="store_true",
                    help="show the whole grid from the first frame")
    ap.add_argument("--zoom-hold", type=float, default=1.2,
                    help="seconds on the single board before pulling back")
    ap.add_argument("--zoom-seconds", type=float, default=2.8,
                    help="length of the pull-back")
    ap.add_argument("--focus", type=int, default=None,
                    help="board to open on (default: the busiest)")
    ap.add_argument("--supersample", type=int, default=2,
                    help="draw at this multiple and downscale, for smooth edges")

    ap.add_argument("--out", default=str(OUT_DIR / "mosaic.mp4"))
    ap.add_argument("--still", type=float, default=None,
                    help="write one PNG at this many seconds instead of a video")
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg.exe")
    ap.add_argument("--crf", type=int, default=18,
                    help="H.264 quality: lower is better and bigger (16-23)")
    args = ap.parse_args()

    W, H = parse_size(args.size)
    SS = max(1, args.supersample)
    fps = args.fps
    frames = max(1, round(args.seconds * fps))
    zoom = not args.no_zoom
    hold, zsec = (args.zoom_hold, args.zoom_seconds) if zoom else (0.0, 0.0)
    speed_end = args.speed if args.speed_end is None else args.speed_end

    arena = load_arena()
    painter = BoardPainter(arena)
    bw, bh = painter.size_tiles

    paths, pairs, rung = simulate(args, (args.n + args.spares) * args.queue)
    print("loading replays...")
    replays = [Replay(p) for p in paths]

    # The clock: ticks elapsed at each frame, integrating a speed that may ramp
    # up once the pull-back is over.
    ramp0 = hold + zsec
    clock, c = [], 0.0
    for f in range(frames):
        clock.append(c)
        tau = f / fps
        u = (tau - ramp0) / max(1e-6, args.seconds - ramp0)
        c += (args.speed + (speed_end - args.speed) * ease_in_out(u)) * 10.0 / fps

    all_queues = [replays[k * args.queue:(k + 1) * args.queue]
                  for k in range(args.n + args.spares)]
    all_pairs = [pairs[k * args.queue] for k in range(args.n + args.spares)]
    queues, spare_ids = all_queues[:args.n], list(range(args.n, args.n + args.spares))
    pairs_shown = all_pairs[:args.n]
    look =int(min(clock[-1], clock[min(frames - 1, round((ramp0 + 2) * fps))])) + 1
    if args.start == "auto":
        # Capped at 900: later windows are busier (double elixir starts at
        # 1200) but by then most boards have lost towers, which at grid scale
        # reads as empty lawn.
        best = None
        for s in range(200, 901, 20):
            score = sum(Slot(q, min(s, max(0, len(q[0]) - 200))).activity(0, look)
                        for q in queues)
            if best is None or score > best[0]:
                best = (score, s)
        start = best[1]
    else:
        start = int(args.start)
    def slot_for(q):
        return Slot(q, min(start, max(0, len(q[0]) - 200)))

    slots = [slot_for(q) for q in queues]

    # A board where a unit vibrates at a bridge mouth reads as lag, so it is
    # swapped for a spare match that shows none.
    c_end = clock[-1] + 1
    swapped = 0
    for k in range(args.n):
        if not slots[k].shows_jitter(c_end):
            continue
        while spare_ids:
            j = spare_ids.pop(0)
            cand = slot_for(all_queues[j])
            if not cand.shows_jitter(c_end):
                slots[k], pairs_shown[k] = cand, all_pairs[j]
                swapped += 1
                break
        else:
            print(f"WARNING: board {k} shows a unit stuck at a bridge mouth and no "
                  f"clean spare is left; raise --spares")
    if swapped:
        print(f"swapped {swapped} board(s) with a unit stuck at a bridge mouth "
              f"for clean spares")

    t_world, cols, rows, rects = board_rects(args.n, W, H, bw, bh)
    focus_clock = max(1, int(clock[min(frames - 1, round(ramp0 * fps))]))
    focus = args.focus
    if focus is None:
        focus = max(range(args.n),
                    key=lambda k: slots[k].activity(0, focus_clock, fights=True))
    # Open on the centre of the grid, so the pull-back grows evenly outward.
    centre = (rows // 2) * cols + cols // 2 - (1 if cols % 2 == 0 else 0)
    centre = min(centre, args.n - 1)
    slots[focus], slots[centre] = slots[centre], slots[focus]
    pairs_shown[focus], pairs_shown[centre] = pairs_shown[centre], pairs_shown[focus]

    vp_grid = (0.0, 0.0, float(W), float(H))
    vp_focus = focus_viewport(rects[centre], W, H) if zoom else vp_grid

    print(f"{args.n} boards as {cols}x{rows}, {t_world:.2f} px/tile | "
          f"start tick {start} | {frames} frames at {fps} fps | "
          f"speed {args.speed}x -> {speed_end}x")
    print(f"opening board: {pairs_shown[centre][0].name} (blue) vs "
          f"{pairs_shown[centre][1].name} (red)")

    static_bg = {}

    def render(f):
        tau = f / fps
        vp = camera(ease_in_out((tau - hold) / zsec) if zoom and zsec > 0 else 1.0,
                    vp_focus, vp_grid)
        z = W / vp[2] * SS
        t = t_world * z
        key = tuple(round(v, 3) for v in vp)
        boxes = []
        for k, (x, y, w, h) in enumerate(rects):
            sx, sy = (x - vp[0]) * z, (y - vp[1]) * z
            if sx > W * SS or sy > H * SS or sx + w * z < 0 or sy + h * z < 0:
                continue
            boxes.append((k, sx, sy, w * z, h * z))
        frame = static_bg.get(key)
        if frame is None:
            frame = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
            for k, sx, sy, sw, sh in boxes:
                bx, by = round(sx), round(sy)
                size = (max(1, round(sw)), max(1, round(sh)))
                vx0, vy0 = max(0, bx), max(0, by)
                vx1 = min(W * SS, bx + size[0])
                vy1 = min(H * SS, by + size[1])
                if vx1 <= vx0 or vy1 <= vy0:
                    continue
                frame.paste(painter.background(
                    t, size, (vx0 - bx, vy0 - by, vx1 - bx, vy1 - by)), (vx0, vy0))
            if len(static_bg) < 2 and key == tuple(round(v, 3) for v in vp_grid):
                static_bg[key] = frame
        frame = frame.copy()
        for k, sx, sy, sw, sh in boxes:
            r, q, flash, team = slots[k].at(clock[f])
            painter.draw(frame, round(sx), round(sy), t, r, r.at(q), flash, team)
        return frame.reduce(SS) if SS > 1 else frame

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Which match each board shows, and over which ticks: provenance for the
    # footage, and what an independent check can re-scan.
    boards = []
    for k, slot in enumerate(slots):
        shown = []
        for r, s in zip(slot.replays, slot.starts):
            lo, hi = max(0.0, -s), min(len(r) - 1.0, c_end - s)
            if hi >= lo:
                blue, red = pairs[paths.index(r.path)]
                shown.append({"replay": os.path.basename(r.path),
                              "blue": blue.name, "red": red.name,
                              "ticks": [int(lo), int(math.ceil(hi))]})
        boards.append({"board": k, "row": k // cols, "col": k % cols, "shows": shown})
    with open(os.path.splitext(args.out)[0] + ".boards.json", "w", encoding="utf-8") as fh:
        json.dump({"opening_board": centre, "boards": boards}, fh, indent=1)

    if args.still is not None:
        f = min(frames - 1, max(0, round(args.still * fps)))
        path = os.path.splitext(args.out)[0] + f"_still_{args.still:g}s.png"
        render(f).convert("RGB").save(path)
        print(f"wrote {path}")
        return

    t0 = time.time()
    with VideoWriter(args.out, (W, H), fps, args.ffmpeg, args.crf) as vid:
        for f in range(frames):
            vid.write(render(f))
            if f % fps == 0:
                print(f"  frame {f}/{frames} ({time.time() - t0:.0f}s)")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({frames / fps:.1f}s, {W}x{H}, {size_mb:.1f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
