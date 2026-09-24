"""Shared plumbing for the promo-video tools in this folder.

  engine()        the compiled simulator, with a readable error on the wrong Python
  load_arena()    river and bridge cells, derived from the engine bindings
  Replay          one GameLogger JSON, parsed into per-tick entity tuples
  BoardPainter    draws one board at any scale, in web/viewer.html's palette
  VideoWriter     pipes frames into ffmpeg as an H.264 MP4

The palette mirrors web/viewer.html's CSS tokens, so footage from these tools
and a screen capture of the viewer look like one product. Geometry is NOT copied
from the viewer: it comes from the bindings, and the two values no binding
reaches are named against their headers.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
TRAINING_PYTHON = REPO_ROOT / "python_ai" / "venv" / "Scripts" / "python.exe"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _wrong_python(exc):
    script = " ".join(sys.argv) or "tools/promo/<script>.py"
    return SystemExit(
        f"{exc}\n\nThe engine (clash_royale_env.pyd) is built for Python 3.11, and "
        f"this is Python {sys.version.split()[0]}.\nRun the script with the "
        f"training venv's Python instead:\n\n  {TRAINING_PYTHON} {script}\n")


try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as _exc:  # a system Python without Pillow
    raise _wrong_python(_exc) from _exc


def engine():
    """The compiled simulator module."""
    try:
        import python_ai  # noqa: F401  -- puts the .pyd's directory on sys.path
        import clash_royale_env
    except ImportError as exc:
        raise _wrong_python(exc) from exc
    return clash_royale_env


# --- arena geometry ----------------------------------------------------------
# Board.h: riverY_end. No binding reaches it (perception/geometry.py pins the
# same value against the header).
RIVER_Y_END = 17.5
# GameManager.h: OWN_HALF_RIVER_BUFFER, from the own-half limit to the river.
OWN_HALF_RIVER_BUFFER = 0.5
# Board.h: BRIDGE_HALF_WIDTH.
BRIDGE_HALF_WIDTH = 1.0


@dataclass(frozen=True)
class Arena:
    width: int
    height: int
    river_rows: tuple   # game rows that are water (cell r covers [r-0.5, r+0.5])
    bridge_cols: tuple  # (first, last) column of each bridge deck

    def river_row(self):
        """The river row as W/B, e.g. WWBBWWWWWWWWWWBBWW."""
        deck = {c for c0, c1 in self.bridge_cols for c in range(c0, c1 + 1)}
        return "".join("B" if c in deck else "W" for c in range(self.width))


def load_arena():
    E = engine()
    from python_ai.deck import SHIPPED_DECK
    CE = E.ClashRoyaleEnv
    probe = CE(list(SHIPPED_DECK), list(SHIPPED_DECK), 10)
    river_start = probe.get_own_half_max_y() + OWN_HALF_RIVER_BUFFER
    rows = tuple(r for r in range(CE.BOARD_HEIGHT) if river_start <= r < RIVER_Y_END)
    bridges = []
    for bx in (E.ARENA_LEFT_BRIDGE_X, E.ARENA_RIGHT_BRIDGE_X):
        cols = [c for c in range(CE.BOARD_WIDTH)
                if bx - BRIDGE_HALF_WIDTH <= c < bx + BRIDGE_HALF_WIDTH]
        bridges.append((cols[0], cols[-1]))
    return Arena(CE.BOARD_WIDTH, CE.BOARD_HEIGHT, rows, tuple(bridges))


# --- replays -----------------------------------------------------------------
# Tower card ids as GameLogger writes them, and their drawn size in tiles (the
# real game's King is 4x4 and Princess 3x3, as web/viewer.html draws them).
KING_CARD_ID, PRINCESS_CARD_ID = -2, -3
TOWER_SIZE = {KING_CARD_ID: 4.0, PRINCESS_CARD_ID: 3.0}
BUILDING_SIZE = 2.0


@dataclass(frozen=True)
class Body:
    """Per-entity drawing facts, fixed for the entity's lifetime."""
    kind: str           # tower | building | troop | spell | projectile
    team: int
    max_hp: float
    radius: float       # tiles: troop body radius, or spell disc radius
    size: float         # tiles: side of a tower or building
    roll: tuple         # (width, range) of a rolling spell, else (0, 0)
    origin: tuple       # (x, y) where it first appeared
    flying: bool


def troop_radius(max_hp):
    """Drawn radius in tiles, from max HP: a Skeleton reads small and a Golem
    big. The engine's placement radius is 0.4 for every troop, so it cannot
    tell them apart; web/viewer.html uses a per-symbol table instead, which
    leaves most of the 173 cards on its default.
    """
    return min(0.62, max(0.22, 0.22 + 0.09 * math.log(max(max_hp, 80.0) / 80.0)))


def classify(ent, card_meta):
    cid = int(ent.get("cardId", -1))
    meta = card_meta.get(str(cid)) or {}
    team = int(ent["team"])
    first_hp = float(ent["hp"])
    max_hp = float(meta.get("maxHp") or 0) or max(first_hp, 1.0)
    origin = (float(ent["x"]), float(ent["y"]))
    flying = bool(ent.get("isFlying", meta.get("isFlying", False)))
    if cid in TOWER_SIZE:
        return Body("tower", team, first_hp, 0, TOWER_SIZE[cid], (0, 0), origin, False)
    if ent["symbol"] == "-":
        return Body("projectile", team, 1, 0.12, 0, (0, 0), origin, flying)
    if meta.get("isSpell"):
        roll = (float(meta.get("rollWidth", 0)), float(meta.get("rollRange", 0)))
        # web/viewer.html's disc: 4 tiles for Arrows ('*'), 2.5 otherwise.
        radius = 4.0 if ent["symbol"] == "*" else 2.5
        return Body("spell", team, 1, radius, 0, roll, origin, False)
    if meta.get("isBuilding"):
        return Body("building", team, max_hp, 0, BUILDING_SIZE, (0, 0), origin, False)
    return Body("troop", team, max_hp, troop_radius(max_hp), 0, (0, 0), origin, flying)


class Replay:
    """One replay JSON (GameLogger::save), parsed for drawing.

    ticks[t] is a list of (id, x, y, hp) tuples; bodies[id] carries what does
    not change. towers lists the towers alive at tick 0, so a destroyed one can
    be drawn as rubble.
    """

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        card_meta = data.get("cardMeta", {})
        self.path = str(path)
        self.result = data.get("result", {})
        self.ticks = []
        self.bodies = {}
        for tick in data["ticks"]:
            row = []
            for ent in tick["entities"]:
                eid = int(ent["id"])
                if eid not in self.bodies:
                    self.bodies[eid] = classify(ent, card_meta)
                row.append((eid, float(ent["x"]), float(ent["y"]), float(ent["hp"])))
            self.ticks.append(row)
        self.towers = [(eid, x, y) for eid, x, y, _ in self.ticks[0]
                       if self.bodies[eid].kind == "tower"]
        self._dicts = {}
        self._jitter = None

    def jitter_spans(self, min_run=6):
        """[(first_tick, last_tick)] where some unit flips between the same two
        points every tick for at least `min_run` ticks.

        That is the bridge-mouth overshoot (perception/UPSTREAM_REQUESTS.md
        item 31): a unit on the bank line steps past the bridge mouth, turns
        back and steps past it again, forever. On screen it vibrates in place.
        Measured over 128 matches, every run of 6+ ticks was on a bank line,
        while ordinary collision nudges reverse once or twice and stay below.
        """
        if self._jitter is None:
            paths = {}
            for t, row in enumerate(self.ticks):
                for eid, x, y, _ in row:
                    if self.bodies[eid].kind == "troop":
                        paths.setdefault(eid, []).append((t, x, y))
            spans = []
            for pts in paths.values():
                run = 0
                for i in range(2, len(pts) + 1):
                    flip = False
                    if i < len(pts):
                        (ta, xa, ya), (_, xb, yb), (tc, xc, yc) = pts[i - 2], pts[i - 1], pts[i]
                        flip = (tc - ta == 2 and abs(xc - xa) < 1e-4 and abs(yc - ya) < 1e-4
                                and (abs(xc - xb) > 1e-3 or abs(yc - yb) > 1e-3))
                    if flip:
                        run += 1
                    elif run:
                        if run >= min_run:
                            spans.append((pts[i - run - 2][0], pts[i - 1][0]))
                        run = 0
            self._jitter = spans
        return self._jitter

    def __len__(self):
        return len(self.ticks)

    @property
    def winner(self):
        """0 or 1, or None for a draw."""
        loser = self.result.get("loserTeam", -1)
        return None if loser not in (0, 1) else 1 - loser

    def units(self, t, team=None):
        """How many non-tower, non-projectile bodies are on the board at t."""
        return sum(1 for e in self.ticks[t]
                   if self.bodies[e[0]].kind in ("troop", "building", "spell")
                   and (team is None or self.bodies[e[0]].team == team))

    def at(self, q):
        """Entities at fractional tick q, positions interpolated by id."""
        q = min(max(q, 0.0), len(self.ticks) - 1.0)
        a = int(q)
        alpha = q - a
        if alpha <= 1e-6 or a + 1 >= len(self.ticks):
            return self.ticks[a]
        nxt = self._dicts.get(a + 1)
        if nxt is None:
            if len(self._dicts) > 64:
                self._dicts.clear()
            nxt = self._dicts[a + 1] = {e[0]: e for e in self.ticks[a + 1]}
        out = []
        for eid, x, y, hp in self.ticks[a]:
            b = nxt.get(eid)
            if b is not None:
                x += (b[1] - x) * alpha
                y += (b[2] - y) * alpha
            out.append((eid, x, y, hp))
        return out


# --- palette: web/viewer.html's :root tokens ----------------------------------
def _hex(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


BG = _hex("#0B0B0C")               # --bg-primary
SURROUND = _hex("#2E4A1C")         # --arena-surround
GRASS = (_hex("#93C756"), _hex("#84BC46"))   # --grass-light / --grass-dark
LANE = _hex("#CFC08F")             # --lane-path
RIVER = _hex("#37BEDC")            # --river
RIVER_EDGE = _hex("#2AA5C2")       # --river-edge
BRIDGE = _hex("#B4813F")           # --bridge-deck
BRIDGE_SEAM = _hex("#8A5F2B")      # --bridge-seam
TEAM = {0: _hex("#4a9eff"), 1: _hex("#ff5252")}        # --blue-team / --red-team
TEAM_DIM = {0: _hex("#1a3a66"), 1: _hex("#661a1a")}    # drawEntity's dimColor
TOWER_FACE = {0: _hex("#3E6E9E"), 1: _hex("#9E4444")}  # drawEntity's gradient top
HP_COLORS = (_hex("#4ade80"), _hex("#facc15"), _hex("#ff5252"))
RUBBLE = (112, 138, 70)   # a scorched patch of lawn, not a dark building


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


TEAM_BODY = {k: _mix(TEAM[k], TEAM_DIM[k], 0.45) for k in TEAM}

_FONT_FILES = ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf")
_fonts = {}


def font(px):
    px = max(6, int(px))
    f = _fonts.get(px)
    if f is None:
        for name in _FONT_FILES:
            for base in ("", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")):
                try:
                    f = ImageFont.truetype(os.path.join(base, name), px)
                    break
                except OSError:
                    continue
            if f is not None:
                break
        if f is None:
            f = ImageFont.load_default(size=px)
        _fonts[px] = f
    return f


def composite(img, patch, x0, y0):
    """alpha_composite `patch` onto RGBA `img` at (x0, y0), clipped to img."""
    w, h = patch.size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(img.width, x0 + w), min(img.height, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    if (cx0, cy0, cx1, cy1) != (x0, y0, x0 + w, y0 + h):
        patch = patch.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    img.alpha_composite(patch, (cx0, cy0))


class BoardPainter:
    """Draws one arena at any scale onto an RGBA frame.

    Coordinates follow web/viewer.html's gameToCanvas: tile (x, y) is centred
    at pad + (x + 0.5) * t across and pad + (H - 1 - y + 0.5) * t down, with y
    growing toward the red side at the top.
    """

    #: Background pyramid, in pixels per tile. A board is resized down from
    #: the next level up, so any zoom stays sharp without re-painting 612
    #: lawn tiles per board per frame.
    LEVELS = (8, 16, 32, 64, 128)

    def __init__(self, arena, pad_tiles=0.35):
        self.arena = arena
        self.pad = pad_tiles
        self._levels = {}
        self._gradients = {}

    @property
    def size_tiles(self):
        return (self.arena.width + 2 * self.pad, self.arena.height + 2 * self.pad)

    def to_px(self, ox, oy, t, x, y):
        return (ox + (self.pad + x + 0.5) * t,
                oy + (self.pad + (self.arena.height - 1 - y) + 0.5) * t)

    # -- background --------------------------------------------------------
    def _paint_background(self, t):
        A, pad = self.arena, self.pad * t
        bw, bh = self.size_tiles
        img = Image.new("RGB", (round(bw * t), round(bh * t)), SURROUND)
        d = ImageDraw.Draw(img)

        def span(i):  # pixel edges of tile i along one axis
            return round(pad + i * t), round(pad + (i + 1) * t) - 1

        for x in range(A.width):
            x0, x1 = span(x)
            for row in range(A.height):  # canvas row, 0 at the top
                y0, y1 = span(row)
                d.rectangle((x0, y0, x1, y1), fill=GRASS[(x + row) % 2])
        top, bottom = round(pad), round(pad + A.height * t) - 1
        for c0, c1 in A.bridge_cols:  # lane paths, under the river
            d.rectangle((span(c0)[0], top, span(c1)[1], bottom), fill=LANE)
        r_top = span(A.height - 1 - max(A.river_rows))[0]
        r_bot = span(A.height - 1 - min(A.river_rows))[1]
        d.rectangle((round(pad), r_top, round(pad + A.width * t) - 1, r_bot), fill=RIVER)
        d.rectangle((round(pad), r_top, round(pad + A.width * t) - 1,
                     r_top + max(1, round(t * 0.12)) - 1), fill=RIVER_EDGE)
        for c0, c1 in A.bridge_cols:
            bl, br = span(c0)[0], span(c1)[1]
            d.rectangle((bl, r_top, br, r_bot), fill=BRIDGE)
            for k in (1, 2, 3):
                py = round(r_top + (r_bot - r_top) * k / 4)
                d.line((bl, py, br, py), fill=BRIDGE_SEAM, width=max(1, round(t * 0.06)))
        if t >= 16:  # the viewer's faint tile grid
            grid = Image.new("RGBA", img.size, (0, 0, 0, 0))
            g = ImageDraw.Draw(grid)
            for x in range(A.width + 1):
                px = round(pad + x * t)
                g.line((px, top, px, bottom), fill=(0, 0, 0, 14))
            for y in range(A.height + 1):
                py = round(pad + y * t)
                g.line((round(pad), py, round(pad + A.width * t), py), fill=(0, 0, 0, 14))
            img = Image.alpha_composite(img.convert("RGBA"), grid).convert("RGB")
        return img

    def background(self, t, size, visible=None):
        """The board's static layer at `t` px/tile, scaled to `size`.

        `visible` = (x0, y0, x1, y1), in that scaled board's pixels, returns
        only that part: a zoomed-in board is mostly off-screen, and resizing
        all of it cost ~2 s a frame.
        """
        level = next((lv for lv in self.LEVELS if lv >= t), self.LEVELS[-1])
        src = self._levels.get(level)
        if src is None:
            src = self._levels[level] = self._paint_background(level)
        x0, y0, x1, y1 = visible or (0, 0, size[0], size[1])
        if src.size == size and visible is None:
            return src
        fx, fy = src.width / size[0], src.height / size[1]
        return src.resize((x1 - x0, y1 - y0),
                          Image.BOX if size[0] < src.width else Image.BILINEAR,
                          box=(x0 * fx, y0 * fy, x1 * fx, y1 * fy))

    # -- bodies ------------------------------------------------------------
    def _block(self, team, px, radius):
        """A tower/building face: the viewer's top-to-bottom gradient, rounded."""
        key = (team, px)
        img = self._gradients.get(key)
        if img is None:
            if len(self._gradients) > 96:
                self._gradients.clear()
            grad = Image.linear_gradient("L").resize((px, px))
            img = ImageOps.colorize(grad, TOWER_FACE[team], TEAM_DIM[team]).convert("RGBA")
            mask = Image.new("L", (px, px), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, px - 1, px - 1), radius=radius, fill=255)
            img.putalpha(mask)
            self._gradients[key] = img
        return img

    def draw(self, img, ox, oy, t, replay, ents, flash=0.0, flash_team=None):
        """Draw `ents` (from Replay.at) for one board whose top-left is (ox, oy)
        on RGBA `img`, at `t` px per tile.
        """
        d = ImageDraw.Draw(img)
        bodies = replay.bodies
        alive = {e[0] for e in ents}
        detail = t >= 22          # HP bars read from here up
        # On a small board a Skeleton would be a pixel; floor the radius so a
        # swarm still shows up as a blob.
        min_r = 0.38 * t if t < 24 else 1.2

        # Rubble where a tower stood.
        for eid, x, y in replay.towers:
            if eid not in alive:
                cx, cy = self.to_px(ox, oy, t, x, y)
                h = bodies[eid].size * t * 0.28
                d.rounded_rectangle((cx - h, cy - h, cx + h, cy + h),
                                    radius=max(1, t * 0.2), fill=RUBBLE)

        spells, blocks, ground, air, shots = [], [], [], [], []
        for e in ents:
            kind = bodies[e[0]].kind
            if kind == "spell":
                spells.append(e)
            elif kind in ("tower", "building"):
                blocks.append(e)
            elif kind == "projectile":
                shots.append(e)
            elif bodies[e[0]].flying:
                air.append(e)
            else:
                ground.append(e)

        for eid, x, y, _ in spells:
            self._spell(img, ox, oy, t, bodies[eid], x, y)

        for eid, x, y, hp in blocks:
            b = bodies[eid]
            cx, cy = self.to_px(ox, oy, t, x, y)
            px = max(2, round(b.size * t))
            face = self._block(b.team, px, max(1, round(t * 0.18)))
            composite(img, face, round(cx - px / 2), round(cy - px / 2))
            d.rounded_rectangle((cx - px / 2, cy - px / 2, cx + px / 2, cy + px / 2),
                                radius=max(1, t * 0.18), outline=TEAM[b.team],
                                width=max(1, round(t * 0.09)))
            if detail:
                self._hp_bar(d, cx, cy - px / 2 - t * 0.35, b.size * 0.8 * t, t, hp / b.max_hp)

        for group in (ground, air):
            for eid, x, y, hp in group:
                b = bodies[eid]
                cx, cy = self.to_px(ox, oy, t, x, y)
                r = max(min_r, b.radius * t)
                if b.flying and detail:  # a shadow below it: in the air
                    s = 0.8 * r
                    sh = Image.new("RGBA", (round(2 * s + 2), round(2 * s + 2)), (0, 0, 0, 0))
                    ImageDraw.Draw(sh).ellipse((0, 0, sh.width - 1, sh.height - 1),
                                               fill=(0, 0, 0, 60))
                    composite(img, sh, round(cx - s - 1 + t * 0.1),
                              round(cy - s - 1 + t * 0.3))
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=TEAM_BODY[b.team],
                          outline=TEAM[b.team], width=max(1, round(t * 0.07)))
                if detail:
                    self._hp_bar(d, cx, cy - r - t * 0.3, 2.2 * r, t, hp / b.max_hp)

        for eid, x, y, _ in shots:
            cx, cy = self.to_px(ox, oy, t, x, y)
            r = max(1.2, 0.12 * t)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=TEAM[bodies[eid].team])

        if flash > 0 and flash_team is not None:
            bw, bh = self.size_tiles
            w = max(2, round(t * 0.9 * flash))
            d.rectangle((ox, oy, ox + bw * t - 1, oy + bh * t - 1),
                        outline=TEAM[flash_team], width=w)

    def _spell(self, img, ox, oy, t, b, x, y):
        col = TEAM[b.team]
        width, rng = b.roll
        if width > 0 and rng > 0:
            # A rolling spell: the swept corridor, faint, and the body at its
            # leading edge. Team 0 rolls toward +y (AreaSpell::configureRoll).
            direction = 1 if b.team == 0 else -1
            x0, y_tip = self.to_px(ox, oy, t, b.origin[0] - width / 2, y)
            x1, y_org = self.to_px(ox, oy, t, b.origin[0] + width / 2, b.origin[1])
            _, y_body = self.to_px(ox, oy, t, 0, y - direction * 0.8)
            patch_w = round(x1 - x0) + 2
            top = round(min(y_tip, y_org, y_body)) - 1
            bot = round(max(y_tip, y_org, y_body)) + 1
            patch = Image.new("RGBA", (max(1, patch_w), max(1, bot - top)), (0, 0, 0, 0))
            p = ImageDraw.Draw(patch)
            lx = round(x0) - 1

            def rect(ya, yb, **kw):
                p.rectangle((0, min(ya, yb) - top, patch_w - 1, max(ya, yb) - top), **kw)

            rect(y_org, y_tip, fill=col + (40,))
            rect(y_body, y_tip, fill=col + (215,), outline=col + (140,))
            composite(img, patch, lx, top)
            return
        r = b.radius * t
        patch = Image.new("RGBA", (round(2 * r) + 3, round(2 * r) + 3), (0, 0, 0, 0))
        ImageDraw.Draw(patch).ellipse((1, 1, patch.width - 2, patch.height - 2),
                                      fill=col + (34,), outline=col + (90,),
                                      width=max(1, round(t * 0.06)))
        cx, cy = self.to_px(ox, oy, t, x, y)
        composite(img, patch, round(cx - patch.width / 2), round(cy - patch.height / 2))

    @staticmethod
    def _hp_bar(d, cx, top, width, t, ratio):
        ratio = min(1.0, max(0.0, ratio))
        h = max(2, t * 0.14)
        x0 = cx - width / 2
        d.rectangle((x0 - 1, top - 1, x0 + width + 1, top + h + 1), fill=(0, 0, 0))
        col = HP_COLORS[0] if ratio > 0.5 else HP_COLORS[1] if ratio > 0.25 else HP_COLORS[2]
        if ratio > 0:
            d.rectangle((x0, top, x0 + width * ratio, top + h), fill=col)


# --- video -------------------------------------------------------------------
FFMPEG_HELP = """ffmpeg was not found. Install it once, then open a NEW terminal:

    winget install --id Gyan.FFmpeg -e

or pass its location with --ffmpeg C:\\path\\to\\ffmpeg.exe.
"""


def find_ffmpeg(explicit=None):
    """ffmpeg's path: --ffmpeg, $FFMPEG, PATH, imageio-ffmpeg, or a winget
    install the current terminal has not picked up yet.
    """
    for cand in (explicit, os.environ.get("FFMPEG")):
        if cand:
            if not os.path.isfile(cand):
                raise SystemExit(f"ffmpeg not found at {cand}")
            return cand
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    local = os.environ.get("LOCALAPPDATA", "")
    hits = sorted(glob.glob(os.path.join(
        local, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "*", "bin", "ffmpeg.exe")))
    if hits:
        return hits[-1]
    raise SystemExit(FFMPEG_HELP)


class VideoWriter:
    """Frames in, an H.264 MP4 out (yuv420p, +faststart: plays everywhere,
    including YouTube Shorts, X and Reddit)."""

    def __init__(self, path, size, fps, ffmpeg=None, crf=18):
        self.path = str(path)
        self.size = size
        exe = find_ffmpeg(ffmpeg)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        cmd = [exe, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, img):
        if img.size != self.size:
            raise ValueError(f"frame is {img.size}, video is {self.size}")
        self.proc.stdin.write(img.convert("RGB").tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()


SIZES = {"vertical": (1080, 1920), "landscape": (1920, 1080), "square": (1080, 1080)}


def parse_size(spec):
    if spec in SIZES:
        return SIZES[spec]
    try:
        w, h = (int(v) for v in spec.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size {spec!r}: use vertical, landscape, square or WxH")
    if w % 2 or h % 2:
        raise SystemExit("--size: width and height must be even for H.264")
    return w, h


def ease_in_out(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)
