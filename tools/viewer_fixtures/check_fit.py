"""Check that web/viewer.html's arena fits its window, in headless Chrome/Edge.

The arena canvas is backed at devicePixelRatio and its CSS size pinned to the
board's logical size (CELL_SIZE and PADDING), so the promo export can render it
at exactly that size. Pinning the HEIGHT made `max-height: 100%` inert: on any
window shorter than the board (772 CSS px) the arena overflowed its panel and
was clipped, which read as "the board is zoomed in". This checks the three
cases that pin has to satisfy at once:

  short window   the arena shrinks to fit its panel
  tall window    the arena shows at its logical size, not DPR times larger
  export         with the promo board-only CSS, exactly the logical size

Run (any Python with Pillow, which the export module imports):
    python_ai/venv/Scripts/python.exe tools/viewer_fixtures/check_fit.py
Exits non-zero on any failure.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "promo"))

from cdp import Browser  # noqa: E402
from export_viewer import BOARD_ONLY_CSS, READY, VIEWER  # noqa: E402

FIXTURE = REPO / "web" / "fixtures" / "modern.json"
TOL = 0.5  # CSS px of sub-pixel layout slack

# (label, CSS viewport width, height, device scale factor)
WINDOWS = [
    ("short, 125% DPI", 950, 717, 1.25),
    ("short, 100% DPI", 1280, 680, 1.0),
    ("short, 200% DPI", 900, 600, 2.0),
    ("tall, 125% DPI", 1400, 1000, 1.25),
    ("tall, 100% DPI", 1400, 1000, 1.0),
]

MEASURE = """(() => {
  const r = document.getElementById('gameCanvas').getBoundingClientRect();
  const p = document.querySelector('.arena-panel');
  const pr = p.getBoundingClientRect(), cs = getComputedStyle(p);
  return {canvas: [r.left, r.top, r.width, r.height],
          panel: [pr.left + parseFloat(cs.paddingLeft),
                  pr.top + parseFloat(cs.paddingTop),
                  pr.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
                  pr.height - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)],
          dpr: devicePixelRatio};
})()"""


def load(b, text):
    b.open(VIEWER.as_uri(), READY)
    b.eval(f"__viewerTests.loadData(JSON.parse({json.dumps(text)}), 'fit'), true")


def main():
    text = FIXTURE.read_text(encoding="utf-8")
    data = json.loads(text)
    failures = []

    def check(ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    with Browser(800, 600, 1.0, os.environ.get("CLASH_BROWSER")) as b:
        for label, w, h, dpr in WINDOWS:
            b.set_viewport(w, h, dpr)
            load(b, text)
            cell, pad = b.eval("[__viewerTests.cellSize(), __viewerTests.padding()]")
            lw = data["boardWidth"] * cell + 2 * pad
            lh = data["boardHeight"] * cell + 2 * pad
            m = b.eval(MEASURE)
            cx, cy, cw, ch = m["canvas"]
            px, py, pw, ph = m["panel"]
            print(f"{label}: {w}x{h} @ {m['dpr']}  canvas {cw:.0f}x{ch:.0f}  "
                  f"panel {pw:.0f}x{ph:.0f}  board {lw}x{lh}")
            check(cy >= py - TOL and cy + ch <= py + ph + TOL,
                  f"{label}: arena inside its panel vertically")
            check(cx >= px - TOL and cx + cw <= px + pw + TOL,
                  f"{label}: arena inside its panel horizontally")
            check(cw <= lw + TOL and ch <= lh + TOL,
                  f"{label}: arena no larger than its logical size")
            if ph >= lh and pw >= lw + 220:  # room for the board and the gauge
                check(abs(cw - lw) <= TOL and abs(ch - lh) <= TOL,
                      f"{label}: arena at exactly its logical size when it fits")

        # The promo export's board-only layout, at its own sizing.
        b.set_viewport(800, 1000, 4.0)
        load(b, text)
        b.eval("(() => { const s = document.createElement('style'); "
               f"s.textContent = {json.dumps(BOARD_ONLY_CSS)}; "
               "document.head.appendChild(s); "
               "document.body.classList.add('promo-board'); return true; })()")
        cw, ch = b.eval(MEASURE)["canvas"][2:]
        print(f"export board-only @ 4.0: canvas {cw:.2f}x{ch:.2f}  board {lw}x{lh}")
        check(abs(cw - lw) <= TOL and abs(ch - lh) <= TOL,
              "export: arena at exactly its logical size")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
