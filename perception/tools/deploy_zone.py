"""Read the real game's deployable region straight off the screen.

    perception/.venv/Scripts/python.exe perception/tools/deploy_zone.py --slots 0,1,2,3

Checking placements against the simulator's `is_valid_placement` is circular:
the bot's mask is built from that predicate. The game answers the question
itself whenever a card is selected, by tinting the region you may not deploy
into red. So

    screenshot with no card selected
    tap a hand slot
    screenshot with the card selected
    subtract

reads the real rule for 288 cells at once, at zero elixir, with no policy or
detector in the loop. The tint raises redness (R - (G+B)/2) by 55-107 inside
the forbidden region and by 0.0-0.2 outside, so the threshold is not a tuning
parameter.

The tint is sampled at `engine_tile_centre(x, y)`, the pixel the actuator would
tap, so a cell reads illegal if the game forbids it or our tile grid maps it
outside the allowed area: the composition that decides whether a card deploys.

One baseline, then switch selection; never deselect. A second tap on a selected
slot does not cancel the selection, so a later baseline would already contain
the tint. Tapping a different slot switches the selection.

Every failure mode of this measurement answers "everything is legal": an
unaffordable card selects nothing, a stale baseline already holds the tint, a
tap pixel outside the arena is untinted. CONTROL_ROWS and `in_arena` guard
against that.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "python_ai") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "python_ai"))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from live.actuator import ADB, card_centre, engine_tile_centre  # noqa: E402
from live.adapter import TILE_Y_OFFSET  # noqa: E402

# Redness delta above which a pixel counts as tinted. The populations sit at
# ~0.1 and ~60, so anything in 10..40 gives the same answer.
TINT_THRESHOLD = 15.0

# Half-width of the patch sampled around a tile centre, in display pixels. A
# tile is 37.4 x 28.8, so +/-6 stays inside it and gives 169 samples, enough
# that a unit, a tower edge or an HP bar cannot flip the median.
PATCH = 6

# The engine's own half for team 0 is rows 0..15 (18 x 16 = 288 cells), what
# the net's placement_mask offers a troop. Derived from the binding when
# importable; this is the fallback.
FALLBACK_OWN_HALF_ROWS = 16

# The internal control. An unaffordable card selects nothing, so no tint
# appears and every cell reads legal. A band deep in the enemy half must be
# tinted whenever a card really is selected; if it is not, the frame is
# discarded.
CONTROL_ROWS = (20, 28)          # DETECTOR rows, well past the river
CONTROL_MIN_DELTA = 20.0


def screencap(adb: Path, path: Path | None = None) -> Image.Image:
    # The adb daemon-start banner can precede the PNG on stdout;
    # match_nav.decode_screencap owns that knowledge. The offset is resolved
    # before the write, so `path` never receives a corrupt .png.
    from match_nav import _PNG_MAGIC, decode_screencap  # noqa: PLC0415

    p = subprocess.run([str(adb), "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=60)
    img = decode_screencap(p.stdout, p.returncode, p.stderr)
    if path is not None:
        path.write_bytes(p.stdout[p.stdout.find(_PNG_MAGIC):])
    return img


def redness(img: Image.Image) -> np.ndarray:
    """R - (G+B)/2. The tint is a red multiply, so it moves this and little else,
    while the grass itself varies frame to frame.
    """
    a = np.asarray(img, dtype=np.float32)
    return a[:, :, 0] - 0.5 * (a[:, :, 1] + a[:, :, 2])


def control_tint(delta: np.ndarray) -> float:
    """Mean redness delta deep in the enemy half: did the card actually get
    selected? See CONTROL_ROWS.
    """
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        DISPLAY_HEIGHT,
        TILE_HEIGHT,
        TILE_INIT_X,
        TILE_INIT_Y,
        TILE_WIDTH,
    )
    lo, hi = CONTROL_ROWS
    y0 = DISPLAY_HEIGHT - TILE_INIT_Y - (hi + 1) * TILE_HEIGHT
    y1 = DISPLAY_HEIGHT - TILE_INIT_Y - lo * TILE_HEIGHT
    band = delta[int(y0):int(y1), int(TILE_INIT_X):int(TILE_INIT_X + 18 * TILE_WIDTH)]
    return float(band.mean())


def wait_for_elixir(detector, adb: Path, want: float = 10.0,
                    timeout_s: float = 40.0, serial: str | None = None):
    """Block until the bar reads `want`, so every slot is affordable."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        img = screencap(adb)
        state = detector.run(img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT),
                                        Image.LANCZOS))
        if state is None:
            time.sleep(0.5)
            continue
        if state.screen.name != "in_game":
            return state
        last = state
        if state.numbers.elixir.number >= want:
            return state
        time.sleep(1.0)
    return last


def tap(adb: Path, x: int, y: int, serial: str | None = None) -> None:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    subprocess.run(cmd + ["shell", "input", "tap", str(x), str(y)],
                   check=True, capture_output=True, timeout=20)


def in_arena(px: int, py: int) -> bool:
    """Does this pixel lie inside the arena rectangle? Not redundant with the tint
    test: the tint exists only inside the arena, so a tap below the board
    (where engine row 0 lands) is untinted, and off the board is not
    permission.
    """
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        DISPLAY_HEIGHT,
        TILE_HEIGHT,
        TILE_INIT_X,
        TILE_INIT_Y,
        TILE_WIDTH,
    )
    x0, x1 = TILE_INIT_X, TILE_INIT_X + 18 * TILE_WIDTH
    y1 = DISPLAY_HEIGHT - TILE_INIT_Y                 # our own back edge
    y0 = y1 - 32 * TILE_HEIGHT                        # the enemy back edge
    return x0 <= px <= x1 and y0 <= py <= y1


def tile_is_clear(delta: np.ndarray, px: int, py: int) -> tuple[bool, float]:
    """Is the tap pixel for this cell inside the arena and outside the tint? A
    median over a small patch, so the verdict belongs to the tile rather than
    whatever stands on it.
    """
    h, w = delta.shape
    if not (0 <= py < h and 0 <= px < w) or not in_arena(px, py):
        return False, float("nan")
    patch = delta[max(0, py - PATCH):py + PATCH + 1,
                  max(0, px - PATCH):px + PATCH + 1]
    med = float(np.median(patch))
    return med < TINT_THRESHOLD, med


def fit_arena_rect(delta: np.ndarray) -> dict:
    """Fit the tile grid to the arena rectangle the game draws for itself.

    The forbidden tint is a solid rectangle over the enemy half, so its edges
    are board edges, readable to the pixel:

        top    the arena's far edge          -- detector row boundary 32
        bottom the river / own-half edge     -- detector row boundary 15
        left   the arena's left edge         -- column boundary 0
        right  the arena's right edge        -- column boundary 18

    Two y edges 17 row-boundaries apart give TILE_HEIGHT directly, with no
    landmark whose tile index must be assumed. The arena's bottom edge is
    untinted, so it is extrapolated 15 rows down from the river edge; a tap
    scan puts the last deployable pixel at y=981 +/- 1 against 984 predicted
    (the last ~3 px sit under the King's HP bar).
    """
    ys = delta[:, 200:520].mean(1)          # mid-board columns only: the side
    xs = delta[200:500, :].mean(0)          # borders and towers are not edges

    # Longest contiguous run, not min/max: selecting a card also highlights it
    # in the tray, a shorter separate red run far below the arena.
    def longest_run(profile) -> tuple[int, int]:
        best = cur = None
        for i, v in enumerate(profile):
            if v > TINT_THRESHOLD:
                cur = (i, i) if cur is None else (cur[0], i)
                if best is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
            else:
                cur = None
        if best is None:
            raise RuntimeError("no tint found -- no card is selected")
        return best

    y_lo, y_hi = longest_run(ys)
    x_lo, x_hi = longest_run(xs)
    # Edges sit half a pixel outside the last tinted pixel.
    top, river = y_lo - 0.5, y_hi + 0.5
    left, right = x_lo - 0.5, x_hi + 0.5

    tile_h = (river - top) / (32 - 15)
    tile_w = (right - left) / 18
    bottom = river + 15 * tile_h
    return {
        "TILE_WIDTH": tile_w, "TILE_HEIGHT": tile_h,
        "TILE_INIT_X": left, "TILE_INIT_Y": 1280.0 - bottom,
        "_arena": {"left": left, "right": right, "top": top,
                   "river": river, "bottom": bottom},
    }


def engine_legality(card_id: int, cells_wide: int, rows: int) -> np.ndarray | None:
    """The simulator's verdict for the same cells, or None without the binding.
    For differencing only: where they disagree, the game is right.
    """
    try:
        import clash_royale_env as engine  # noqa: PLC0415
    except Exception:                                       # noqa: BLE001
        return None
    from live.unit_to_card import UNKNOWN_CARD_SIM_ID  # noqa: PLC0415
    if card_id == UNKNOWN_CARD_SIM_ID:
        return None
    deck = [10, 1, 41, 25, 7, 2, 6, 5]
    env = engine.ClashRoyaleEnv(deck, deck, 100)
    env.reset()
    out = np.zeros((rows, cells_wide), dtype=bool)
    for y in range(rows):
        for x in range(cells_wide):
            out[y, x] = env.is_valid_placement(card_id, float(x), float(y), 0)
    return out


def render(mask: np.ndarray, rows: int, width: int, mark=None) -> str:
    """ASCII map, our own back line at the bottom, as the screen shows it."""
    lines = ["      " + "".join(f"{x % 10}" for x in range(width))]
    for y in range(rows - 1, -1, -1):
        row = "".join(
            ("#" if mark is not None and mark[y, x] else
             ("." if mask[y, x] else "X"))
            for x in range(width))
        lines.append(f"  y{y:2d} {row}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slots", default="0,1,2,3",
                    help="hand slots to probe, comma separated")
    ap.add_argument("--adb", type=Path, default=ADB)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--settle", type=float, default=0.7,
                    help="seconds after a tap before the screen is read")
    ap.add_argument("--out", type=Path, default=Path("deploy_zone.json"))
    ap.add_argument("--save-frames", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=None,
                    help="engine rows to report (default: the own half)")
    ap.add_argument("--fit-grid", action="store_true",
                    help="report the tile grid fitted to the arena rectangle "
                         "the tint marks out, instead of a legality map")
    ap.add_argument("--ensure-match", action="store_true",
                    help="navigate into a Training Camp match first")
    args = ap.parse_args()

    try:
        import clash_royale_env as engine  # noqa: PLC0415
        width = engine.ClashRoyaleEnv.BOARD_WIDTH
        probe = engine.ClashRoyaleEnv([10, 1, 41, 25, 7, 2, 6, 5],
                                      [10, 1, 41, 25, 7, 2, 6, 5], 10)
        own_rows = int(probe.get_own_half_max_y()) + 1
    except Exception as exc:                                # noqa: BLE001
        print(f"  (engine binding unavailable: {exc})")
        width, own_rows = 18, FALLBACK_OWN_HALF_ROWS
    rows = args.rows or own_rows

    # The detector only names the cards and refuses to run outside a match; the
    # measurement is pure pixels.
    from clashroyalebuildabot.detectors.detector import Detector  # noqa: PLC0415
    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415
    deck = [Cards.VALKYRIE, Cards.ARCHERS, Cards.MINIONS, Cards.CANNON,
            Cards.FIREBALL, Cards.GIANT, Cards.MUSKETEER, Cards.MINIPEKKA]
    detector = Detector(deck)

    if args.ensure_match:
        from match_nav import ensure_in_match  # noqa: PLC0415
        state = ensure_in_match(detector, args.adb, args.serial)
    else:
        base_img = screencap(args.adb)
        state = detector.run(base_img.resize(
            (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS))
    if state is None or state.screen.name != "in_game":
        print(f"not in a match (screen="
              f"{None if state is None else state.screen.name}) -- aborting")
        return 1
    hand = [c.name for c in state.cards[1:5]]
    elixir0 = state.numbers.elixir.number
    print(f"in_game   elixir {elixir0}   hand {hand}   ready {sorted(state.ready)}")
    print(f"board {width} wide, reporting engine rows 0..{rows - 1}   "
          f"TILE_Y_OFFSET={TILE_Y_OFFSET}\n")

    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    # One clean baseline, then switch selection slot to slot without
    # deselecting (module docstring): four taps, no elixir. Waiting for full
    # elixir first makes all four slots affordable at once, keeping baseline
    # and measurements seconds apart.
    st = wait_for_elixir(detector, args.adb, 10.0, serial=args.serial)
    if st is None or st.screen.name != "in_game":
        print("left the match while waiting for elixir -- aborting")
        return 1
    hand = [c.name for c in st.cards[1:5]]
    print(f"probing at elixir {st.numbers.elixir.number}, hand {hand}")
    baseline = redness(screencap(
        args.adb, None if args.save_frames is None else
        args.save_frames / "zone_baseline.png"))

    results = {}
    for slot in [int(s) for s in args.slots.split(",") if s.strip() != ""]:
        name = hand[slot]
        card = card_centre(slot)

        tap(args.adb, card.x, card.y, args.serial)
        time.sleep(args.settle)
        sel_img = screencap(args.adb,
                            None if args.save_frames is None else
                            args.save_frames / f"zone_slot{slot}_{name}.png")
        delta = redness(sel_img) - baseline
        ctrl = control_tint(delta)
        if ctrl < CONTROL_MIN_DELTA:
            print(f"  slot {slot} ({name}): UNMEASURED -- control band delta "
                  f"{ctrl:.1f} < {CONTROL_MIN_DELTA}, so no card is selected\n")
            continue

        if args.fit_grid:
            from clashroyalebuildabot import constants as C  # noqa: PLC0415
            fit = fit_arena_rect(delta)
            a = fit["_arena"]
            print(f"  slot {slot}  {name}: arena rectangle measured from the "
                  f"game's own tint")
            print(f"    left {a['left']:.1f}  right {a['right']:.1f}  "
                  f"top {a['top']:.1f}  river {a['river']:.1f}  "
                  f"bottom(extrapolated) {a['bottom']:.1f}")
            for key, live in (("TILE_WIDTH", C.TILE_WIDTH),
                              ("TILE_HEIGHT", C.TILE_HEIGHT),
                              ("TILE_INIT_X", C.TILE_INIT_X),
                              ("TILE_INIT_Y", C.TILE_INIT_Y)):
                print(f"    {key:<12} fitted {fit[key]:8.3f}   in use "
                      f"{live:8.3f}   delta {fit[key] - live:+7.3f}")
            print()
            continue

        allowed = np.zeros((rows, width), dtype=bool)
        offscreen = np.zeros((rows, width), dtype=bool)
        med = np.full((rows, width), np.nan)
        for y in range(rows):
            for x in range(width):
                t = engine_tile_centre(x, y)
                ok, m = tile_is_clear(delta, t.x, t.y)
                allowed[y, x] = ok
                med[y, x] = m
                offscreen[y, x] = not np.isfinite(m)

        sim_id = hand_card_id_for(name)
        sim = engine_legality(sim_id, width, rows)
        results[name] = {
            "slot": slot, "sim_id": sim_id,
            "real_allowed": allowed.tolist(),
            "offscreen": offscreen.tolist(),
            "engine_allowed": None if sim is None else sim.tolist(),
        }

        print(f"  slot {slot}  {name}  (sim id {sim_id})   "
              f"control-band tint {ctrl:.0f}")
        print(f"  REAL GAME says {int(allowed.sum())}/{allowed.size} cells legal")
        print(render(allowed, rows, width, mark=offscreen))
        if sim is not None:
            print(f"  ENGINE says    {int(sim.sum())}/{sim.size}")
            bad = sim & ~allowed          # engine permits, game refuses
            miss = allowed & ~sim         # game permits, engine refuses
            print(f"  ENGINE PERMITS BUT GAME REFUSES: {int(bad.sum())} cells")
            if bad.any():
                print(render(bad, rows, width))
                for y in range(rows):
                    xs = [x for x in range(width) if bad[y, x]]
                    if xs:
                        print(f"     engine row {y}: x={xs}")
            print(f"  game permits but engine refuses: {int(miss.sum())} cells")
        print()

    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
