"""Read the REAL game's deployable region straight off the screen.

    perception/.venv/Scripts/python.exe perception/tools/deploy_zone.py --slots 0,1,2,3

WHY THIS IS THE ORACLE THE PROJECT HAS BEEN MISSING
---------------------------------------------------
Every previous attempt to decide whether a placement was legal asked the
SIMULATOR (`ClashRoyaleEnv.is_valid_placement`). The mask the bot plays with is
built from that same predicate, so the check was circular and could only ever
return "all legal" -- see handoff.md section 0. The game kept refusing taps
anyway.

Clash Royale answers the question itself, for free, every time a card is
selected: it tints the region you may NOT deploy into red. So

    screenshot with no card selected
    tap a hand slot
    screenshot with the card selected
    subtract

is a direct, dense, per-pixel readout of the real rule -- 288 cells at once,
costing zero elixir, with no policy, no detector and no inference in the loop.

Measured on a Training Camp match: the tint raises "redness" (R - (G+B)/2) by
55-107 inside the forbidden region and by 0.0-0.2 outside it. Two orders of
magnitude of separation, so the threshold is not a tuning parameter.

WHAT IT TESTS THAT A PURE RULE COMPARISON CANNOT
------------------------------------------------
It samples the tint at `engine_tile_centre(x, y)` -- the exact pixel the
actuator would tap for that cell. So a cell is reported illegal if EITHER the
game forbids that area OR our tile grid maps the cell to a pixel outside it.
Legality and calibration are the same question from the actuator's point of
view, and this measures the composition of the two, which is what actually
decides whether a card deploys.

ONE BASELINE, THEN SWITCH SELECTION -- NEVER DESELECT
-----------------------------------------------------
The obvious loop is select / measure / deselect / repeat, and it does not work.
A second tap on an already-selected slot does NOT cancel the selection, so
every baseline after the first was captured with the previous card's tint still
on screen and the difference came out at ~0 everywhere -- which reads as "the
whole board is legal".

Tapping a DIFFERENT slot switches the selection (the card tray is not the
arena, so it cannot place a card), so one clean baseline taken before any card
is touched serves every slot.

BOTH FAILURE MODES OF THIS MEASUREMENT ARE "EVERYTHING IS LEGAL"
----------------------------------------------------------------
An unaffordable card selects nothing; a stale baseline already contains the
tint; a tap pixel outside the arena is untinted because there is no board
there. All three produce a confident, maximally permissive, entirely wrong
answer -- the same shape of error as validating a mask against the predicate
that generated it. Hence CONTROL_ROWS and `in_arena`, which are not belt and
braces but the only things standing between this tool and that answer.
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

from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from live.actuator import ADB, card_centre, engine_tile_centre  # noqa: E402
from live.adapter import TILE_Y_OFFSET  # noqa: E402

# Redness delta above which a pixel counts as tinted. The two populations sit at
# ~0.1 and ~60, so anything in 10..40 gives the same answer; 15 is the midpoint
# of the gap on a log scale and is quoted in the module docstring.
TINT_THRESHOLD = 15.0

# Half-width of the patch sampled around a tile centre, in DISPLAY pixels. A
# tile is 37.4 x 28.8, so +/-6 stays well inside one tile while giving 169
# samples to take a median over -- enough that a unit standing on the tile, a
# tower edge or the HP-bar overlay cannot flip the verdict.
PATCH = 6

# The engine's own half for team 0 is rows 0..15 (18 x 16 = 288 cells), which is
# what model.py's placement_mask offers a troop. Derived below from the binding
# when it is importable; this is the fallback for a machine without the .pyd.
FALLBACK_OWN_HALF_ROWS = 16

# THE INTERNAL CONTROL, and the reason the first run of this tool produced a
# confident, entirely wrong "288/288 cells legal".
#
# Tapping a card the player cannot afford selects nothing, so no tint appears,
# so every cell reads as untinted, so every cell reads as LEGAL. The failure
# mode of this measurement is silently maximal permissiveness -- exactly the
# shape of error handoff.md section 0 is about, arrived at from the other side.
#
# So the tint is verified against a region that MUST be forbidden whenever a
# card really is selected: deep in the enemy half. If that band is not tinted,
# the selection did not happen and the frame is discarded rather than believed.
CONTROL_ROWS = (20, 28)          # DETECTOR rows, well past the river
CONTROL_MIN_DELTA = 20.0


def screencap(adb: Path, path: Path | None = None) -> Image.Image:
    # The adb daemon-start banner lands on STDOUT ahead of the PNG on the first
    # call of a session -- see match_nav.decode_screencap, which owns the one
    # copy of that knowledge. Saving `raw` unstripped would also write a
    # corrupt .png to `path`, so the offset is resolved BEFORE the write.
    from match_nav import _PNG_MAGIC, decode_screencap  # noqa: PLC0415

    p = subprocess.run([str(adb), "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=60)
    img = decode_screencap(p.stdout, p.returncode, p.stderr)
    if path is not None:
        path.write_bytes(p.stdout[p.stdout.find(_PNG_MAGIC):])
    return img


def redness(img: Image.Image) -> np.ndarray:
    """R - (G+B)/2. The tint is a red multiply, so it moves this and little else.

    Chosen over a plain per-channel difference because the arena's own grass
    varies frame to frame (the checkerboard breathes, units cast shadows) while
    its redness does not.
    """
    a = np.asarray(img, dtype=np.float32)
    return a[:, :, 0] - 0.5 * (a[:, :, 1] + a[:, :, 2])


def control_tint(delta: np.ndarray) -> float:
    """Mean redness delta deep in the enemy half -- the "did the card actually
    get selected" check. See CONTROL_ROWS."""
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
    """Block until the bar reads `want`, so every slot is affordable.

    Probing at whatever elixir happens to be there is what made the first run
    useless: the two 4-cost slots were unaffordable, selected nothing, and
    reported every cell legal.
    """
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
    """Does this pixel lie inside the arena rectangle at all?

    NOT redundant with the tint test, and leaving it out cost the first clean
    run a wrong answer. The tint only exists inside the arena, so a tap pixel
    that lands BELOW the board -- in the dead strip above the card tray, which
    is exactly where engine row 0 lands -- is untinted, and "untinted" was being
    read as "the game allows it". Off the board is not permission; it is a tap
    that reaches nothing.
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
    """Is the tap pixel for this cell inside the arena AND outside the tint?

    Median over a small patch, not the single centre pixel: one pixel can sit on
    a unit, a tower edge or an HP bar, and those move. The median of 169 makes
    the verdict a property of the tile rather than of whatever happened to be
    standing on it.
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

    The forbidden tint is a solid rectangle covering the enemy half, so its
    edges ARE board edges and can be read to the pixel:

        top    the arena's far edge          -- detector row boundary 32
        bottom the river / own-half edge     -- detector row boundary 15
        left   the arena's left edge         -- column boundary 0
        right  the arena's right edge        -- column boundary 18

    Two y edges 17 row-boundaries apart give TILE_HEIGHT directly, with no
    landmark whose tile index has to be assumed. That assumption is exactly
    what the 2026-08-05 refit got wrong: it scaled y off "the two princess HP
    bars are 21.0 tiles apart" and x off "the two river gaps are 10.0 tiles
    apart", and both counts were short by one, inflating TILE_HEIGHT by 4% and
    TILE_WIDTH by 10%.

    The arena's BOTTOM edge is not in the tint (our own half is untinted), so it
    is extrapolated 15 rows down from the river edge rather than measured. A tap
    scan puts the last deployable pixel at y=981 +/- 1 against 984 predicted
    here; the last ~3 px sit under the King's HP bar and are swallowed by the
    UI, which does not move any tile centre.
    """
    ys = delta[:, 200:520].mean(1)          # mid-board columns only: the side
    xs = delta[200:500, :].mean(0)          # borders and towers are not edges

    # LONGEST CONTIGUOUS RUN, not min/max. Selecting a card also highlights the
    # card in the tray, which is red and sits at y>1050 -- far below the arena.
    # Taking the extreme tinted row therefore put the "river" edge at y=1265
    # and produced a TILE_HEIGHT of 68. The arena block is ~470 rows tall and
    # the tray highlight is a much shorter separate run, so the longest run is
    # the board and the rule needs no hardcoded search window.
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
    """The SIMULATOR's verdict for the same cells, or None without the binding.

    Present only so the two can be differenced. It is explicitly NOT the
    reference: where they disagree the game is right by construction.
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
    """ASCII map, our own back line at the BOTTOM, the way the screen shows it."""
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

    # The detector is only needed to name the cards and to refuse to run outside
    # a match. The measurement itself is pure pixels.
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

    # ONE clean baseline, then SWITCH selection from slot to slot without ever
    # deselecting.
    #
    # The obvious design -- select, measure, deselect, repeat -- does not work,
    # and failed in a way worth recording: a second tap on an already-selected
    # slot does NOT cancel the selection, so every baseline after the first was
    # captured with the previous card's tint still on screen. The delta then
    # came out at ~0 everywhere, which reads as "the whole board is legal".
    # Same silently-maximal-permissiveness failure as an unaffordable card.
    #
    # Tapping a DIFFERENT slot switches the selection (the card tray is not the
    # arena, so it cannot place), so one baseline serves every slot and the
    # whole sweep costs four taps and no elixir. Waiting for full elixir first
    # makes all four slots affordable at once, which keeps the baseline and the
    # measurements seconds apart instead of a minute.
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
