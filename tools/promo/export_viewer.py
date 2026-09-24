"""A replay, rendered by web/viewer.html itself, as an MP4.

Drives the real viewer in headless Chrome (or Edge) and screenshots every
frame, so the footage is exactly what the viewer shows. Two layouts:

  board   vertical 1080x1920 (Shorts): the arena alone, optionally cropped to
          a region and zoomed. The default.
  full    the whole viewer UI, 1920x1080: brain gauge, hands, towers, events.

    python_ai/venv/Scripts/python.exe tools/promo/export_viewer.py replays/replay_ep115422.json
    python_ai/venv/Scripts/python.exe tools/promo/export_viewer.py REPLAY --layout full --start 600 --end 900
    python_ai/venv/Scripts/python.exe tools/promo/export_viewer.py REPLAY --crop 9,11,17,22 --label "Recorded before the fix"

Motion is smooth at any speed: the viewer interpolates positions between the
replay's ticks for export (__viewerTests.gotoTickFrac), so 1x playback at
30 fps does not hold each tick for three frames.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from pathlib import Path

from common import (BG, OUT_DIR, REPO_ROOT, Image, ImageDraw, VideoWriter, font,
                    parse_size)
from cdp import Browser

VIEWER = REPO_ROOT / "web" / "viewer.html"
READY = "document.readyState === 'complete' && !!window.__viewerTests"
MAX_SCALE = 8.0         # device scale cap: the 420x772 CSS arena at 8x is ~21 Mpx
MARGIN = 0.02           # of the frame's short side, around the board

# Board layout: hide everything but the arena and pin it to the top-left at
# its logical size, so the page is exactly as big as the arena.
BOARD_ONLY_CSS = """
body.promo-board { background: transparent !important; overflow: hidden !important; }
body.promo-board * { visibility: hidden !important; }
body.promo-board #gameCanvas {
  visibility: visible !important; position: fixed !important;
  left: 0 !important; top: 0 !important; z-index: 2147483647 !important;
  max-width: none !important; max-height: none !important; border-radius: 0 !important;
}
"""


def _parse_crop(spec):
    if not spec:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit("--crop takes x0,y0,x1,y1 in board tiles, e.g. 9,11,17,22")
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


class ViewerSession:
    """The viewer loaded with one replay, ready to render any fractional tick."""

    def __init__(self, replay_path, layout, size, crop=None, browser=None):
        self.layout = layout
        self.size = size
        with open(replay_path, "r", encoding="utf-8") as fh:
            self.text = fh.read()
        data = json.loads(self.text)
        self.ticks = len(data["ticks"])
        self.board_w, self.board_h = data["boardWidth"], data["boardHeight"]
        self.name = os.path.basename(replay_path)

        if layout == "board":
            self.crop = crop
            # Load once to ask the viewer its arena geometry (CELL_SIZE and
            # PADDING, never restated here), then size the page to the arena
            # and pick the scale that gives the captured region at least as
            # many pixels as it will fill.
            self.browser = Browser(800, 1000, 1.0, browser)
            self._load()
            cell, pad = self.browser.eval("[__viewerTests.cellSize(), __viewerTests.padding()]")
            self.cell, self.pad = cell, pad
            self.css_w = self.board_w * cell + 2 * pad
            self.css_h = self.board_h * cell + 2 * pad
            self.clip = self._clip_rect()
            fit = min((size[0] * (1 - 2 * MARGIN)) / self.clip[2],
                      (size[1] * (1 - 2 * MARGIN)) / self.clip[3])
            scale = min(MAX_SCALE, max(1.0, math.ceil(fit * 4) / 4))
            self.browser.set_viewport(self.css_w, self.css_h, scale)
            self._load()
            self.scale = scale
        else:
            self.crop = None
            self.browser = Browser(size[0], size[1], 1.0, browser)
            self._load()
            self.scale = 1.0

    def _load(self):
        b = self.browser
        b.open(VIEWER.as_uri(), READY)
        b.eval("document.fonts.ready.then(() => true)")
        b.eval(f"__viewerTests.loadData(JSON.parse({json.dumps(self.text)}), "
               f"{json.dumps(self.name)}), true")
        if self.layout == "board":
            b.eval("(() => { const s = document.createElement('style'); "
                   f"s.textContent = {json.dumps(BOARD_ONLY_CSS)}; "
                   "document.head.appendChild(s); "
                   "document.body.classList.add('promo-board'); return true; })()")

    def _clip_rect(self):
        """(x, y, w, h) in CSS pixels: the whole arena, or the crop's region
        mapped the way the viewer's gameToCanvas maps tiles."""
        if not self.crop:
            return (0, 0, self.css_w, self.css_h)
        x0, y0, x1, y1 = self.crop
        cs, pad, H = self.cell, self.pad, self.board_h
        left = pad + x0 * cs + cs / 2
        right = pad + x1 * cs + cs / 2
        top = pad + (H - 1 - y1) * cs + cs / 2
        bottom = pad + (H - 1 - y0) * cs + cs / 2
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.css_w, right), min(self.css_h, bottom)
        return (left, top, right - left, bottom - top)

    def select(self, entity_id):
        self.browser.eval(f"__viewerTests.select({int(entity_id)}), true")

    def sim_view(self, on=True):
        self.browser.eval(f"__viewerTests.toggleSimView({'true' if on else 'false'}), true")

    def frame(self, tick):
        """The viewer at fractional `tick`, composed into one output frame."""
        tick = max(0.0, min(tick, self.ticks - 1.0))
        self.browser.eval(f"__viewerTests.gotoTickFrac({tick:.4f}), true")
        if self.layout == "board":
            shot = Image.open(io.BytesIO(self.browser.screenshot(self.clip))).convert("RGB")
            return compose_board(shot, self.size)
        shot = Image.open(io.BytesIO(self.browser.screenshot())).convert("RGB")
        return shot if shot.size == self.size else shot.resize(self.size, Image.LANCZOS)

    def close(self):
        self.browser.close()


def compose_board(shot, size):
    """The arena image centred on the frame, as large as the margins allow,
    with rounded corners like the viewer's canvas wrapper."""
    W, H = size
    m = round(min(W, H) * MARGIN)
    k = min((W - 2 * m) / shot.width, (H - 2 * m) / shot.height)
    w, h = max(1, round(shot.width * k)), max(1, round(shot.height * k))
    board = shot.resize((w, h), Image.LANCZOS) if (w, h) != shot.size else shot
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1),
                                           radius=round(min(W, H) * 0.02), fill=255)
    frame = Image.new("RGB", size, BG)
    frame.paste(board, ((W - w) // 2, (H - h) // 2), mask)
    return frame


def burn_label(frame, text):
    """A small pill in the top-left, below the platform's own header: the
    honesty label ("Recreated in the engine") that should never be dropped
    in the edit."""
    if not text:
        return frame
    W, H = frame.size
    f = font(round(min(W, H) * 0.028))
    d = ImageDraw.Draw(frame, "RGBA")
    x, y = round(W * 0.05), round(H * (0.11 if H > W else 0.05))
    l, t, r, b = d.textbbox((x, y), text, font=f)
    pad = round(f.size * 0.45)
    d.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=pad * 2,
                        fill=(11, 11, 12, 200), outline=(255, 255, 255, 60), width=2)
    d.text((x, y), text, font=f, fill=(245, 245, 245, 255))
    return frame


def frame_ticks(start, end, speed, fps):
    """The replay tick each played frame shows. `speed` is game seconds per
    video second: a number, or (tick, speed) keyframes interpolated linearly
    across ticks, for a slow-motion ramp."""
    if isinstance(speed, (int, float)):
        per_frame = speed * 10.0 / fps
        return [start + f * per_frame for f in range(max(1, int((end - start) / per_frame) + 1))]
    keys = sorted((float(t), float(s)) for t, s in speed)

    def at(t):
        for (t0, s0), (t1, s1) in zip(keys, keys[1:]):
            if t < t1:
                return s0 if t <= t0 else s0 + (s1 - s0) * (t - t0) / (t1 - t0)
        return keys[-1][1]

    ticks, t = [], float(start)
    while t <= end:
        ticks.append(t)
        t += at(t) * 10.0 / fps
    return ticks or [float(start)]


def export(replay, out, *, layout="board", size=None, start=0.0, end=None, speed=1.0,
           fps=30, crop=None, select=None, label=None, sim_view=False, hold=0.8,
           still=None, ffmpeg=None, crf=18, browser=None, quiet=False):
    """Render `replay` from tick `start` to `end` at `speed` (see frame_ticks).
    Returns the path written."""
    size = parse_size(size or ("vertical" if layout == "board" else "landscape"))
    t0 = time.time()
    sess = ViewerSession(replay, layout, size, crop, browser)
    try:
        if select is not None:
            sess.select(select)
        if sim_view:
            sess.sim_view(True)
        end = sess.ticks - 1 if end is None else min(float(end), sess.ticks - 1)
        ticks = frame_ticks(start, end, speed, fps)
        n_play = len(ticks)
        n_hold = round(hold * fps)
        if not quiet:
            pace = f"{speed:g}x" if isinstance(speed, (int, float)) else "a speed ramp"
            print(f"{os.path.basename(replay)}: ticks {start:g}-{end:g} at {pace} -> "
                  f"{(n_play + n_hold) / fps:.1f}s, {layout} {size[0]}x{size[1]}, "
                  f"render scale {sess.scale:g}")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if still is not None:
            path = os.path.splitext(out)[0] + f"_still_{still:g}s.png"
            burn_label(sess.frame(ticks[min(n_play - 1, round(still * fps))]), label).save(path)
            return path
        with VideoWriter(out, size, fps, ffmpeg, crf) as vid:
            for f in range(n_play + n_hold):
                vid.write(burn_label(sess.frame(ticks[min(f, n_play - 1)]), label))
    finally:
        sess.close()
    if not quiet:
        print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("replay", help="a replay JSON (python_ai/replays/, replays/, ...)")
    ap.add_argument("--layout", choices=("board", "full"), default="board")
    ap.add_argument("--size", default=None,
                    help="vertical, landscape, square or WxH (default: vertical "
                         "for board, landscape for full)")
    ap.add_argument("--start", type=float, default=0.0, help="first tick")
    ap.add_argument("--end", type=float, default=None, help="last tick (default: the end)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="game seconds per video second (1 = real time)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crop", default=None,
                    help="board layout: zoom to x0,y0,x1,y1 in board tiles")
    ap.add_argument("--select", type=int, default=None,
                    help="entity id to ring in yellow (ids are in the replay JSON)")
    ap.add_argument("--label", default=None,
                    help='small burned-in note, e.g. "Recreated in the engine"')
    ap.add_argument("--sim-view", action="store_true",
                    help="full layout: open the Simulation View (teacher replays)")
    ap.add_argument("--hold", type=float, default=0.8,
                    help="seconds to hold the last frame")
    ap.add_argument("--out", default=None)
    ap.add_argument("--still", type=float, default=None,
                    help="write one PNG at this many seconds instead of a video")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--browser", default=None, help="path to chrome.exe or msedge.exe")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    out = args.out or str(OUT_DIR / (Path(args.replay).stem + f"_{args.layout}.mp4"))
    export(args.replay, out, layout=args.layout, size=args.size, start=args.start,
           end=args.end, speed=args.speed, fps=args.fps, crop=_parse_crop(args.crop),
           select=args.select, label=args.label, sim_view=args.sim_view, hold=args.hold,
           still=args.still, ffmpeg=args.ffmpeg, crf=args.crf, browser=args.browser)


if __name__ == "__main__":
    main()
