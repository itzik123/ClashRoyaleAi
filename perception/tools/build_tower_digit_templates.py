"""Cut tower-HP digit templates from the recordings, labelled by cluster.

    # 1. harvest glyph crops and cluster them into contact sheets
    perception/.venv/Scripts/python.exe \
        perception/tools/build_tower_digit_templates.py cluster \
        --out perception/out/tower_glyphs \
        --recordings perception/assets/recordings

    # 2. read the sheets by eye, write cluster -> digit into a labels file
    #    (the one that produced the committed set is labels_549x976.json)

    # 3. build and save
    perception/.venv/Scripts/python.exe \
        perception/tools/build_tower_digit_templates.py build \
        --out perception/out/tower_glyphs \
        --labels perception/tools/labels_549x976.json \
        --templates perception/config/templates/tower_549x976

Unlike the clock (build_digit_templates.py), whose monotone countdown labels
every glyph from one reading, tower HP drops by arbitrary amounts, so numerals
must be read by eye. Clustering makes that tractable: thousands of crops reduce
to ~30 coherent groups, and an incoherent group is visible on the sheet and
dropped whole.

The recordings, not a live capture: live captures show full-health towers only
(three distinct digits), while the recordings walk towers down through dozens
of values. The numeral renders identically in both (same font, weight and
position relative to the crown badge); only the arena's placement in the frame
differs.

The numeral is located by its whiteness: it is drawn near-white with a dark
outline, the one property independent of the arena skin. A per-channel minimum
above 200 isolates it (tan floor 170-185, grass 65-75, glyph 218-230). Neither
CRBAB's NUMBER_CONFIG (the recordings' crop sits ~9 px higher than live) nor
colour-matching the HP bar (the pink skin's floor matches the bar's dark track)
locates it reliably.

The white mask also segments better than Otsu, which must fit a three-level
histogram (outline / floor / glyph) with two classes and bridges adjacent
digits.

Read-only over assets/recordings/: frames are decoded and dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# clashroyalebuildabot/__init__.py pulls in the whole live bot (keyboard,
# onnxruntime, adb). Only constants.py is needed, so the package is registered
# as a bare namespace.
_pkg = types.ModuleType("clashroyalebuildabot")
_pkg.__path__ = [str(_ROOT / "clashroyalebuildabot")]
sys.modules.setdefault("clashroyalebuildabot", _pkg)

from clashroyalebuildabot.constants import (  # noqa: E402
    ENEMY_PRINCESS_HP_Y, HP_HEIGHT, HP_WIDTH, LEFT_PRINCESS_HP_X,
    RIGHT_PRINCESS_HP_X, SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH,
)
from readers.clock import DigitTemplates, GLYPH_SHAPE, normalise_glyph  # noqa: E402
from readers.tower_numerals import ink_channel, numeral_roi  # noqa: E402

# The emulator window inside the 1920x1080 desktop capture, per the recording
# spec in perception/README.md.
WINDOW = (686, 40, 1236, 1012)
DETECTOR = (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)
WHITE = 200                  # per-channel minimum; see the module docstring
SEARCH = (-14, 22)           # rows around the derived ROI to search
MARGIN = 2                   # rows kept around the located band

# Enemy towers only. The ally numeral is drawn on its bar rather than above it,
# so NUMERAL_OFFSET lands on the tower roof there. See
# perception/UPSTREAM_REQUESTS.md.
BARS = {"L": LEFT_PRINCESS_HP_X, "R": RIGHT_PRINCESS_HP_X}

MIN_INK_HEIGHT_FRAC = 0.80   # the glyph must fill the tile it was scaled into

# Merged pairs are rejected on the raw cell's aspect: normalise_glyph squashes
# anything wider than 18 columns, so after normalisation a merged "90" and a
# wide "0" look alike, and a width filter there would reject every round digit.
MAX_ASPECT = 1.15


# --- harvesting ---

def numeral_band(native, bar_x):
    """(top, bottom, x0, x1) of the numeral, located by its own whiteness."""
    nh, nw = native.shape[:2]
    bar = (bar_x, ENEMY_PRINCESS_HP_Y, bar_x + HP_WIDTH,
           ENEMY_PRINCESS_HP_Y + HP_HEIGHT)
    x0, y0, x1, y1 = numeral_roi(bar, (nw, nh), DETECTOR)
    a, b = max(0, y0 + SEARCH[0]), min(nh, y1 + SEARCH[1])
    if b <= a or x1 <= x0:
        return None

    mask = native[a:b, x0:x1].min(axis=2) > WHITE
    rows = np.where(mask.sum(axis=1) >= 3)[0]
    if rows.size < 6:
        return None
    # One contiguous band: a gap means two things were caught (the numeral and
    # a unit's HP badge, say).
    if rows[-1] - rows[0] + 1 != rows.size:
        return None
    return (a + rows[0] - MARGIN, a + rows[-1] + 1 + MARGIN, x0, x1)


def split_on_white(patch):
    """Column-segment on the near-white mask rather than on Otsu."""
    m = patch.min(axis=2) > WHITE if patch.ndim == 3 else patch > WHITE
    if not m.size:
        return []
    h = m.shape[0]
    runs, start = [], None
    for x, a in enumerate(m.sum(axis=0) >= 2):
        if a and start is None:
            start = x
        elif not a and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, m.shape[1]))

    cells = []
    for a, b in runs:
        rows = np.where(m[:, a:b].any(axis=1))[0]
        # A digit spans most of the band's height; shorter runs are a highlight
        # or a unit's HP badge.
        if rows.size and (rows[-1] - rows[0] + 1) >= h * 0.55:
            cells.append(patch[:, a:b])
    return cells


def cells_from(native, bar_x):
    band = numeral_band(native, bar_x)
    if band is None:
        return []
    t, b, x0, x1 = band
    return split_on_white(native[max(0, t):b, x0:x1])


def harvest(video: Path, every_s: float, tag: str):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * every_s)))
    wx0, wy0, wx1, wy1 = WINDOW

    tiles, n, kept = [], 0, 0
    while cap.grab():
        if n % stride:
            n += 1
            continue
        ok, frame = cap.retrieve()
        n += 1
        if not ok:
            continue
        native = frame[wy0:wy1, wx0:wx1]
        for side, bar_x in BARS.items():
            cells = cells_from(native, bar_x)
            # A Princess Tower's HP has 3 or 4 digits; any other count is a
            # segmentation failure, and a merged cell would poison its
            # template. Cheap to drop.
            if not 3 <= len(cells) <= 4:
                continue
            kept += 1
            for i, cell in enumerate(cells):
                # ink_channel then normalise_glyph: byte for byte what
                # TowerNumeralReader does before matching.
                h_, w_ = cell.shape[:2]
                tiles.append((f"{tag}_{n:06d}_{side}{i}",
                              normalise_glyph(ink_channel(cell), GLYPH_SHAPE),
                              w_ / max(1, h_)))
    cap.release()
    print(f"  {video.name}: {kept} numerals, {len(tiles)} cells")
    return tiles


# --- filtering and clustering ---

def usable(glyph, aspect) -> bool:
    if aspect > MAX_ASPECT:
        return False
    m = glyph > WHITE
    rows = np.where(m.any(axis=1))[0]
    if not rows.size:
        return False
    return (rows[-1] - rows[0] + 1) >= MIN_INK_HEIGHT_FRAC * GLYPH_SHAPE[0]


def kmeans(x: np.ndarray, k: int, iters: int = 80, seed: int = 0):
    """k-means++ with incremental seeding: the naive form allocates n x k x d
    floats at the last seeding step for what is a running minimum. Seeded so a
    rerun reproduces the cluster numbering the labels file names.
    """
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    best = ((x - centres[0]) ** 2).sum(1)
    for _ in range(k - 1):
        centres.append(x[rng.choice(len(x), p=best / best.sum())])
        best = np.minimum(best, ((x - centres[-1]) ** 2).sum(1))
    c = np.array(centres)
    for _ in range(iters):
        lab = np.argmin((x ** 2).sum(1)[:, None] - 2 * x @ c.T
                        + (c ** 2).sum(1)[None], axis=1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def sheets(cells, lab, k, outdir: Path, scale=5, members=8, per_sheet=10):
    """One row per cluster: median first, then members, then the count. The median
    is what gets labelled (it is the template); the members show the cluster is
    coherent enough for one label.
    """
    gh, gw = GLYPH_SHAPE
    pad = 4
    rows = []
    for j in range(k):
        m = cells[lab == j]
        if not len(m):
            continue
        med = np.median(m, axis=0).astype(np.uint8)
        picks = m[np.linspace(0, len(m) - 1, min(members, len(m))).astype(int)]
        row = np.full((gh * scale + 20, (gw * scale + pad) * (members + 2)),
                      30, np.uint8)
        for i, g in enumerate([med] + list(picks)):
            row[0:gh * scale,
                i * (gw * scale + pad):i * (gw * scale + pad) + gw * scale] = \
                cv2.resize(g, (gw * scale, gh * scale),
                           interpolation=cv2.INTER_NEAREST)
        row[0:gh * scale, gw * scale + 1:gw * scale + 3] = 200
        cv2.putText(row, f"c{j}  n={len(m)}", (2, gh * scale + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        rows.append(row)

    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        w = max(r.shape[1] for r in chunk)
        canvas = np.full((sum(r.shape[0] + 3 for r in chunk), w), 30, np.uint8)
        y = 0
        for r in chunk:
            canvas[y:y + r.shape[0], :r.shape[1]] = r
            y += r.shape[0] + 3
        cv2.imwrite(str(outdir / f"clusters_{s // per_sheet}.png"), canvas)


def cmd_cluster(args) -> int:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.recordings).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no recordings in {args.recordings}")

    tiles = []
    for i, v in enumerate(videos):
        tiles += harvest(v, args.every, f"r{i}")
    cells = np.stack([g for _k, g, _a in tiles])
    aspect = np.array([a for _k, _g, a in tiles], np.float32)

    mask = np.array([usable(g, a) for g, a in zip(cells, aspect)])
    print(f"{len(cells)} cells -> {mask.sum()} usable, {(~mask).sum()} dropped")
    cells = cells[mask]

    lab = kmeans(cells.reshape(len(cells), -1).astype(np.float32) / 255.0,
                 args.k)
    np.save(outdir / "cells.npy", cells)
    np.save(outdir / "cluster.npy", lab)
    sheets(cells, lab, args.k, outdir)
    print(f"wrote {outdir}/clusters_*.png -- label them into a labels file")
    for j in range(args.k):
        print(f"  c{j:2d}  {(lab == j).sum():4d}")
    return 0


# --- building ---

def cmd_build(args) -> int:
    src = Path(args.out)
    cells = np.load(src / "cells.npy")
    lab = np.load(src / "cluster.npy")
    labels = json.loads(Path(args.labels).read_text("utf-8"))["clusters"]

    pools: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}
    for j, digit in labels.items():
        if digit is None:
            continue
        pools[digit].extend(cells[lab == int(j)])

    templates = {}
    for digit, crops in sorted(pools.items()):
        if len(crops) < 3:
            raise SystemExit(f"digit {digit}: only {len(crops)} samples")
        # Per-pixel median, as the clock builder uses: discards crops corrupted
        # by a passing troop or damage flash without detecting them.
        templates[digit] = np.median(np.stack(crops), axis=0).astype(np.uint8)
        print(f"  digit {digit}: {len(crops):5d} samples")

    DigitTemplates(templates).save(Path(args.templates))
    print(f"wrote {args.templates}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cluster", help="harvest crops and write contact sheets")
    c.add_argument("--out", required=True)
    c.add_argument("--recordings", default=str(_ROOT / "assets" / "recordings"))
    c.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    c.add_argument("--k", type=int, default=30)
    c.set_defaults(fn=cmd_cluster)

    b = sub.add_parser("build", help="apply a labels file and save templates")
    b.add_argument("--out", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--templates", required=True)
    b.set_defaults(fn=cmd_build)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
