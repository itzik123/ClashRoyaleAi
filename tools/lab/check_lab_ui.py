"""Headless checks for the Reflex Lab page (web/lab/), in Chrome or Edge.

Drives REAL input events through the DevTools protocol, so the page's pointer
handling is what is tested, not a JavaScript shortcut:

  mouse drag     tray card -> a chosen legal cell: placed on exactly that cell
  touch drag     the same with touch events; the aim point floats 60 px above
                 the finger, so the finger is released 60 px below the cell
  cancel         released off the board: nothing placed, the card returns
  illegal        released over the enemy half: refused, nothing placed
  layout         no horizontal scroll, board and tray on screen, at phone and
                 desktop sizes
  training       tries advance, the curve fills, dragging the attacker moves
                 the preview; results render a verdict

Starts its own dev server (tools/lab/dev_server.mjs, the native engine) unless
--url is given, e.g. a WASM build served elsewhere.

Run:  python_ai/venv/Scripts/python.exe tools/lab/check_lab_ui.py
Exits non-zero on any failure.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "promo"))
from cdp import Browser  # noqa: E402

TOUCH_AIM_OFFSET = 60     # web/lab/lab.js TOUCH_AIM_OFFSET
failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait(b, expr, timeout=15.0, what=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if b.eval(expr):
                return True
        except RuntimeError:
            pass
        time.sleep(0.05)
    check(False, f"timed out waiting for {what or expr}")
    return False


def js(b, expr):
    return b.eval(f"(() => {{ {expr} }})()")


def mouse(b, kind, x, y, buttons=1):
    b.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left",
           buttons=buttons if kind != "mouseReleased" else 0, clickCount=1)


def mouse_drag(b, start, end, steps=12):
    mouse(b, "mouseMoved", *start, buttons=0)
    mouse(b, "mousePressed", *start)
    for i in range(1, steps + 1):
        t = i / steps
        mouse(b, "mouseMoved", start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        time.sleep(0.01)
    mouse(b, "mouseReleased", *end)


def touch_drag(b, start, end, steps=12):
    pt = lambda x, y: [{"x": x, "y": y, "id": 1, "radiusX": 8, "radiusY": 8, "force": 1}]
    b.call("Input.dispatchTouchEvent", type="touchStart", touchPoints=pt(*start))
    for i in range(1, steps + 1):
        t = i / steps
        b.call("Input.dispatchTouchEvent", type="touchMove",
               touchPoints=pt(start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
        time.sleep(0.01)
    b.call("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])


def open_lab(b, url):
    b.open(url, "!!window.__lab && !!window.__lab.state.matchup")
    b.eval("document.fonts.ready.then(() => true)")


def tray_center(b):
    return js(b, "const r = document.getElementById('trayCard').getBoundingClientRect();"
                 "return [r.left + r.width / 2, r.top + r.height / 2];")


def cell_center(b, x, y):
    return js(b, f"const p = __lab.cellCenterClient({x}, {y}); return [p.x, p.y];")


def start_round(b):
    js(b, "document.getElementById('btnRound').scrollIntoView({block: 'center'});"
          "document.getElementById('btnRound').click(); return true;")
    wait(b, "__lab.state.roundLive === true", what="the round to go live")
    js(b, "document.getElementById('trayCard').scrollIntoView({block: 'end'}); return true;")
    time.sleep(0.2)


def finish_round(b):
    wait(b, "!__lab.state.roundLive && document.getElementById('btnRound').disabled === false",
         timeout=40, what="the round to finish")


def placement_checks(b, url, touch):
    label = "touch" if touch else "mouse"
    open_lab(b, url)
    js(b, "document.getElementById('btnStart').click(); return true;")
    # A legal cell in front of the King, from the engine's own list.
    target = js(b, "const c = __lab.state.matchup.cells.find(c => c[0] === 9 && c[1] === 9)"
                   " || __lab.state.matchup.cells[Math.floor(__lab.state.matchup.cells.length / 2)];"
                   "return c;")
    want = js(b, f"return __lab.state.cellAt.get('{target[0]},{target[1]}');")

    # 1. a drop on a legal cell lands on exactly that cell
    start_round(b)
    end = cell_center(b, *target)
    if touch:
        end = [end[0], end[1] + TOUCH_AIM_OFFSET]
    drag = touch_drag if touch else mouse_drag
    drag(b, tray_center(b), end)
    wait(b, "__lab.state.attempts[0] && __lab.state.attempts[0].cell != null", timeout=5,
         what=f"{label} placement to register")
    got = js(b, "return __lab.state.attempts[0] && __lab.state.attempts[0].cell;")
    check(got == want, f"{label} drag placed on the aimed cell {target} (index {want}, got {got})")
    finish_round(b)
    score = js(b, "return __lab.state.attempts[0].score;")
    check(isinstance(score, (int, float)), f"{label} round 1 scored ({score})")

    # 2. released off the board: cancelled, nothing placed, avatar hidden
    start_round(b)
    off = js(b, "const r = document.querySelector('.control-panel').getBoundingClientRect();"
                "return [r.left + r.width / 2, r.top + 40];")
    drag(b, tray_center(b), off)
    time.sleep(0.4)
    placed = js(b, "return __lab.state.placed;")
    hidden = js(b, "return document.getElementById('dragAvatar').hidden;")
    check(placed is False and hidden is True, f"{label} drop off the board cancels (placed={placed}, avatar hidden={hidden})")

    # 3. released over the enemy half: refused
    enemy = cell_center(b, 9, 26)
    if touch:
        enemy = [enemy[0], enemy[1] + TOUCH_AIM_OFFSET]
    drag(b, tray_center(b), enemy)
    time.sleep(0.4)
    placed = js(b, "return __lab.state.placed;")
    toast = js(b, "const t = document.getElementById('toast'); return !t.hidden && t.textContent;")
    check(placed is False, f"{label} drop on the enemy half is refused (toast: {toast!r})")
    finish_round(b)


def layout_checks(b, url, w, h, dpr, mobile):
    b.set_viewport(w, h, dpr)
    if mobile:
        b.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
    open_lab(b, url)
    m = js(b, "const s = document.scrollingElement, c = document.getElementById('board').getBoundingClientRect();"
              "const i = document.querySelector('.intro-card').getBoundingClientRect();"
              "return {sw: s.scrollWidth, iw: innerWidth, bl: c.left, br: c.right, bw: c.width, bh: c.height,"
              " il: i.left, ir: i.right, it: i.top, ib: i.bottom, ih: innerHeight};")
    tag = f"{w}x{h}"
    check(m["sw"] <= m["iw"], f"{tag}: no horizontal scroll (scrollWidth {m['sw']} <= {m['iw']})")
    check(m["bl"] >= 0 and m["br"] <= m["iw"] and m["bw"] > 200, f"{tag}: board within the width ({m['bw']:.0f}x{m['bh']:.0f})")
    check(m["il"] >= 0 and m["ir"] <= m["iw"] and m["it"] >= 0 and m["ib"] <= m["ih"], f"{tag}: intro card fits the screen")
    js(b, "document.getElementById('btnStart').click(); return true;")
    t = js(b, "const r = document.getElementById('trayCard').getBoundingClientRect(); return [r.top, r.bottom, innerHeight];")
    check(t[1] <= t[2] + 1, f"{tag}: the card tray is on screen at the start of the challenge ({t[0]:.0f}-{t[1]:.0f} of {t[2]})")


def training_checks(b, url):
    open_lab(b, url)
    js(b, "document.getElementById('btnStart').click(); document.getElementById('btnSkip').click();"
          "document.querySelector('[data-speed=full]').click(); document.getElementById('btnTrain').click(); return true;")
    time.sleep(4)
    st = js(b, "const s = __lab.state; return {tries: s.tries, curve: s.curve.length, heat: s.heat && s.heat.length,"
               " cells: s.matchup.cells.length, preview: s.preview};")
    check(st["tries"] > 500, f"training advances ({st['tries']} tries in 4 s at full speed)")
    check(st["curve"] > 2, f"the learning curve fills ({st['curve']} points)")
    check(st["heat"] == st["cells"], f"heatmap covers every legal cell ({st['heat']} of {st['cells']})")
    # Drag the attacker from one lane to the other.
    a = js(b, "const s = __lab.state.matchup.spawns[__lab.state.preview]; const p = __lab.cellCenterClient(s[0], s[1]); return [p.x, p.y, s[0]];")
    other_x = 14 if a[2] < 9 else 3
    dest = cell_center(b, other_x, 20)
    mouse_drag(b, a[:2], dest)
    time.sleep(0.5)
    moved = js(b, "return __lab.state.matchup.spawns[__lab.state.preview][0];")
    check(abs(moved - other_x) <= 1, f"dragging the attacker moves the preview to the other lane (x {a[2]} -> {moved})")
    js(b, "document.getElementById('btnResults').click(); return true;")
    wait(b, "__lab.state.stage === 'results'", what="the results stage")
    v = js(b, "return document.getElementById('verdict').textContent;")
    check("learner saved" in v, f"results verdict renders: {v!r}")
    errs = js(b, "return window.__labErrors || [];")
    check(not errs, f"no page errors ({errs})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="lab URL (default: start tools/lab/dev_server.mjs)")
    args = ap.parse_args()
    server = None
    url = args.url
    if not url:
        port = free_port()
        server = subprocess.Popen(["node", str(REPO / "tools" / "lab" / "dev_server.mjs"), str(port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}/lab/"
        time.sleep(1.0)
    try:
        with Browser(1440, 900, 1.0, os.environ.get("CLASH_BROWSER")) as b:
            b.call("Runtime.enable")
            print("desktop mouse, 1440x900")
            placement_checks(b, url, touch=False)
            print("phone touch, 390x844")
            b.set_viewport(390, 844, 3.0)
            b.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
            placement_checks(b, url, touch=True)
            b.call("Emulation.setTouchEmulationEnabled", enabled=False)
            print("layout")
            layout_checks(b, url, 1440, 900, 1.0, False)
            layout_checks(b, url, 1280, 720, 1.25, False)
            layout_checks(b, url, 390, 844, 3.0, True)
            b.call("Emulation.setTouchEmulationEnabled", enabled=False)
            print("training and results, 1440x900")
            b.set_viewport(1440, 900, 1.0)
            training_checks(b, url)
    finally:
        if server:
            server.terminate()
    print(f"\n{len(failures)} failure(s)" if failures else "\nall lab UI checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
