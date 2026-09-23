"""Crop card icon templates for our own deck, from the recordings.

    perception/.venv/Scripts/python.exe perception/tools/build_icon_templates.py \
        --cluster                # step 1: propose 8 clusters, write a montage
    ... look at the montage, then ...
        --label Fireball,Giant,Musketeer,...    # step 2: name them, in order

The hand shows one of eight known cards in four fixed rectangles, so this is
grouping, not recognition: crop every slot from every sampled frame, cluster,
and each cluster is a card. The icons are identical pixels each time;
affordability dimming is a brightness offset that k-means on normalised crops
ignores.

The one human step is naming the clusters from a montage. Inferring names from
costs would have to guess (costs are not unique within a deck), and mapping/
refuses to guess.

Once the hand can be read, readers/hand.infer_play_from_hand_change turns each
of our placements into a labelled event (card and frame) for free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calib.homography import load_profile  # noqa: E402
from track.cycle import DECK_SIZE  # noqa: E402
from capture.video import VideoSource  # noqa: E402
from readers.hand import (  # noqa: E402
    ICON_SHAPE,
    has_cost_badge as _has_cost_badge,
    normalise_icon as _normalise,
)



def measure_game_rect(frames) -> tuple[int, int, int, int]:
    """The emulator's game area inside a desktop capture, measured per recording.

    The emulator sits wherever its window was that day: the July batch at (686,
    40, 1236, 1012), the 2026-09-02 recording at (665, 41, 1214, 1018). A fifth
    of a card slot is enough to clip every cost badge and make ScreenDetector
    report `unknown`. The game area is the one large bright block inside the
    black letterbox; a median over several frames stops a dark moment from
    moving an edge.

    Returns (left, top, right, bottom), the PIL crop box.
    """
    grey = np.median(np.stack(frames), axis=0).mean(axis=2)

    def widest_run(mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise RuntimeError("no bright region found -- is this a game capture?")
        runs, start = [], idx[0]
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        return max(runs, key=lambda r: r[1] - r[0])

    h = grey.shape[0]
    x0, x1 = widest_run(grey[int(h * 0.3):int(h * 0.75), :].mean(axis=0) > 40)
    y0, y1 = widest_run(grey[:, x0 + 20:x1 - 20].mean(axis=1) > 40)
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)


def collect_video_crbab(videos: list[Path], sample_fps: float = 2.0,
                        rect: tuple[int, int, int, int] | None = None):
    """Crops from a recording, in the live loop's 368x652 CARD_CONFIG frame:
    measure the game area, crop it out of the desktop frame, resize to 368x652,
    take CARD_CONFIG. The same geometry as `collect_frames`, from an mp4.

    Gated on `in_game` and filtered on the cost badge (see `collect_frames` and
    `has_cost_badge`). The badge filter also drops unaffordable cards on
    purpose: templates come from clean full-contrast exemplars, and contrast
    normalisation handles the dimmed render at read time (these templates score
    100% on dimmed slots).
    """
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        CARD_CONFIG, SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH,
    )
    from clashroyalebuildabot.detectors.screen_detector import (  # noqa: PLC0415
        ScreenDetector,
    )

    screens = ScreenDetector()
    crops, features = [], []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / sample_fps)))

        box = rect
        if box is None:
            probes = []
            for f in (0.2, 0.35, 0.5, 0.65, 0.8):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * f))
                ok, fr = cap.read()
                if ok:
                    probes.append(fr)
            box = measure_game_rect(probes)
            print(f"  {video.name}: game area {box} "
                  f"({box[2]-box[0]}x{box[3]-box[1]})")

        kept = seen = 0
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                continue
            seen += 1
            small = Image.fromarray(fr[:, :, ::-1]).crop(box).resize(
                (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
            if screens.run(small).name != "in_game":
                continue
            for slot in range(1, 5):
                crop = np.ascontiguousarray(
                    np.array(small.crop(CARD_CONFIG[slot]))[:, :, ::-1])
                if not _has_cost_badge(crop):
                    continue
                crops.append(crop)
                features.append(_normalise(crop).ravel())
            kept += 1
        print(f"  {video.name}: {kept}/{seen} sampled frames in_game")
        cap.release()
    return crops, np.array(features, dtype=np.float32)


def collect_frames(frames_dir: Path, stride: int = 1):
    """Same collection, from a dumped PNG frame directory.

    The geometry is CRBAB's, not the calibration profile's. `collect()` crops
    with `profile.rois`, the frame `readers/hand.py` works in; the live loop
    (`live/mvp_loop.py`) resizes to 368x652 and crops `CARD_CONFIG`. A template
    built in one frame and matched in the other is off by a resize and a few
    pixels of framing, which no matcher recovers.
    """
    from PIL import Image  # noqa: PLC0415
    from clashroyalebuildabot.constants import (  # noqa: PLC0415
        CARD_CONFIG, SCREENSHOT_HEIGHT, SCREENSHOT_WIDTH,
    )
    from clashroyalebuildabot.detectors.screen_detector import (  # noqa: PLC0415
        ScreenDetector,
    )

    # Gate on in_game. k-means gets exactly DECK_SIZE centres, so every cluster
    # spent on a non-card costs a card: out-of-match screens contribute a red
    # "7" badge on a wooden panel that passes the magenta badge test. The badge
    # test answers "is there a card here", not "are we in a match".
    screens = ScreenDetector()

    paths = sorted(Path(frames_dir).glob("f*.png"))[::stride]
    crops, features = [], []
    for path in paths:
        image = Image.open(path).convert("RGB").resize(
            (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
        if screens.run(image).name != "in_game":
            continue
        for slot in range(1, 5):  # CARD_CONFIG[0] is the small "next" preview
            crop = np.array(image.crop(CARD_CONFIG[slot]))[:, :, ::-1]  # RGB->BGR
            if not _has_cost_badge(crop):
                continue
            crops.append(np.ascontiguousarray(crop))
            features.append(_normalise(crop).ravel())
    return crops, np.array(features, dtype=np.float32)


def collect(videos: list[Path], profile_path: Path, sample_fps: float = 1.0):
    profile = load_profile(profile_path)
    slots = [profile.rois[f"hand_slot_{i}"] for i in range(4)]

    crops, features = [], []
    for video in videos:
        source = VideoSource(video)
        for frame in source.sample_every(sample_fps):
            for x, y, w, h in slots:
                crop = frame.image[y:y + h, x:x + w]
                if crop.shape[0] != h or crop.shape[1] != w:
                    continue
                if not _has_cost_badge(crop):
                    continue
                crops.append(crop)
                features.append(_normalise(crop).ravel())
        source.close()
    return crops, np.array(features, dtype=np.float32)


def cluster(crops, features, out_dir: Path, k: int = DECK_SIZE):
    """Group the crops, and write one median PNG per group.

    Over-cluster, then merge by name. `k` defaults to DECK_SIZE, which works
    when every card appears a similar number of times in one render state. When
    they do not, k-means spends a cluster on a render state (a selected card's
    highlight) and merges two real cards to pay for it, since it splits large
    groups before separating small ones. A larger `k` with repeated names costs
    one more glance at the montage; `label` merges by name, keeping the largest
    group per card.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    _compact, labels, _centres = cv2.kmeans(
        features, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    counts = []
    for j in range(k):
        members = [c for c, lab in zip(crops, labels) if lab == j]
        counts.append(len(members))
        # Per-pixel median over the cluster: rejects frames where a drag
        # animation or the cursor covered the slot, without detecting them.
        stack = np.stack([cv2.resize(m, (ICON_SHAPE[1], ICON_SHAPE[0])) for m in members])
        template = np.median(stack, axis=0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"cluster_{j}.png"), template)
        tiles.append(cv2.copyMakeBorder(template, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                        value=(0, 255, 0)))

    (out_dir / "cluster_sizes.json").write_text(json.dumps(counts), encoding="utf-8")

    montage = cv2.resize(np.hstack(tiles), None, fx=3, fy=3,
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "clusters.png"), montage)
    print(f"wrote {out_dir / 'clusters.png'}")
    print("cluster sizes (left to right):", counts)
    print(f"\nNow look at clusters.png and re-run with {k} names, in that same "
          f"order:")
    print("  --label <name0>,<name1>,...")
    if k > DECK_SIZE:
        print(f"  ({k} groups for {DECK_SIZE} cards -- repeat a name wherever two "
              f"groups show the same card; the largest group wins.)")
    return labels


def label(names: list[str], out_dir: Path) -> None:
    """Name the groups. Repeats are allowed; the largest group per card wins.

    Two groups for one card are usually its normal and selected render, and
    averaging them smears both. The matcher handles the selected state by
    searching a vertical window (`live/deck_hand.LIFT_SEARCH`), so the template
    should be the dominant clean render.
    """
    import mapping

    sizes_path = out_dir / "cluster_sizes.json"
    sizes = json.loads(sizes_path.read_text(encoding="utf-8")) \
        if sizes_path.exists() else [1] * len(names)
    if len(names) != len(sizes):
        raise SystemExit(
            f"got {len(names)} names for {len(sizes)} clusters -- one name per "
            f"cluster, in montage order")

    best: dict[str, tuple[int, int]] = {}   # real_name -> (size, cluster index)
    entries = {}
    for j, name in enumerate(names):
        entry = mapping.resolve(name)  # raises on an unknown card
        if not entry.in_simulator:
            raise SystemExit(f"{name!r} has no simulator id -- see mapping/")
        entries[entry.real_name] = entry
        if entry.real_name not in best or sizes[j] > best[entry.real_name][0]:
            best[entry.real_name] = (sizes[j], j)

    if len(best) != DECK_SIZE:
        raise SystemExit(
            f"named {len(best)} distinct cards, expected a full deck of "
            f"{DECK_SIZE}: {sorted(best)}. A card with no group of its own has "
            f"no template, and the live loop would read it as whichever of the "
            f"others it least mismatches -- raise --k and re-cluster.")

    index = {}
    for real_name, (size, j) in sorted(best.items()):
        entry = entries[real_name]
        source = out_dir / f"cluster_{j}.png"
        if not source.exists():
            raise SystemExit(f"missing {source}; run --cluster first")
        target = out_dir / f"card_{entry.sim_id}.png"
        cv2.imwrite(str(target), cv2.imread(str(source)))
        index[str(entry.sim_id)] = {"file": target.name, "name": entry.real_name}
        dupes = [i for i, n in enumerate(names)
                 if mapping.resolve(n).real_name == real_name]
        extra = f"  (from {len(dupes)} groups, kept the largest)" if len(dupes) > 1 else ""
        print(f"  cluster {j} (n={size}) -> {entry.real_name} "
              f"(sim id {entry.sim_id}){extra}")

    (out_dir / "icons.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'icons.json'}")


def load_icons(directory: Path) -> dict[int, np.ndarray]:
    """Icon templates keyed by simulator card id, for readers/hand.py."""
    index_path = Path(directory) / "icons.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"no icon templates at {directory}. Build them with "
            "perception/tools/build_icon_templates.py."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    out = {}
    for sim_id, meta in index.items():
        image = cv2.imread(str(Path(directory) / meta["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {meta['file']}")
        out[int(sim_id)] = image
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, nargs="*", default=None)
    parser.add_argument("--frames", type=Path, default=None,
                        help="dumped PNG frame directory; crops in CRBAB "
                             "368x652 CARD_CONFIG geometry, which is what "
                             "the LIVE loop matches in")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--video-crbab", type=Path, nargs="*", default=None,
                        help="recording(s) to cut templates from IN THE "
                             "LIVE LOOP's 368x652 geometry; the emulator "
                             "game area is measured per file")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--rect", type=int, nargs=4, default=None,
                        help="override the measured game area: L T R B")
    parser.add_argument("--profile", type=Path,
                        default=_ROOT / "config" / "profile_gpg_1920x1080.json")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "config" / "templates" / "icons")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--k", type=int, default=DECK_SIZE,
                        help="number of k-means groups; raise it "
                             "above the deck size when a card "
                             "gets no group of its own")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    if args.label:
        label([n.strip() for n in args.label.split(",")], args.out)
        return 0

    if args.video_crbab:
        crops, features = collect_video_crbab(
            args.video_crbab, args.sample_fps,
            tuple(args.rect) if args.rect else None)
        print(f"collected {len(crops)} slot crops from "
              f"{len(args.video_crbab)} recording(s), CRBAB geometry")
    elif args.frames:
        crops, features = collect_frames(args.frames, args.stride)
        print(f"collected {len(crops)} slot crops from {args.frames}")
    else:
        videos = args.video or sorted((_ROOT / "assets" / "recordings").glob("*.mp4"))
        crops, features = collect(videos, args.profile)
        print(f"collected {len(crops)} slot crops from {len(videos)} recording(s)")
    cluster(crops, features, args.out, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
