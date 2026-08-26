"""Crop card icon templates for our own deck, from the recordings.

    perception/.venv/Scripts/python.exe perception/tools/build_icon_templates.py \
        --cluster                # step 1: propose 8 clusters, write a montage
    ... look at the montage, then ...
        --label Fireball,Giant,Musketeer,...    # step 2: name them, in order

WHY CLUSTERING RATHER THAN A CLASSIFIER
---------------------------------------
The hand shows one of exactly eight known cards in four fixed rectangles. That
is not recognition, it is grouping: crop every slot from every sampled frame,
cluster into eight, and each cluster IS a card. No labels are needed to get
that far, and the clusters are near-perfect because the icons are identical
pixels every time they appear -- the only variation is the affordability
dimming, which is a brightness offset that k-means on normalised crops
ignores.

The one human step is naming the eight clusters, which is a single glance at
a montage. That is deliberate: the alternative is inferring names from card
costs, and the costs in this deck are not unique (four cards cost 4), so it
would have to guess. See mapping/ for why guessing is refused everywhere here.

WHAT THIS UNLOCKS
-----------------
Once the hand can be read, readers/hand.infer_play_from_hand_change turns
every one of OUR placements into a labelled event for free -- the card and the
frame it happened on, exactly, with no annotation. That is the training set
for the opponent-side detector in stage 3, which is the last blocked stage.
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

ICON_SHAPE = (48, 40)  # (h, w) that crops are normalised to before clustering


def _has_cost_badge(crop: np.ndarray) -> bool:
    """True if this slot actually holds a card.

    Every card icon carries a magenta elixir-cost badge low-centre. Nothing
    else in the tray does, which makes it a far better "is this a card"
    test than any brightness or variance heuristic.

    That distinction cost a run. Filtering on `crop.std() < 18` let the empty
    between-match tray through -- it has enough texture to pass -- and 1525
    of those crops then formed a cluster of their own. With only eight
    clusters available, that displaced a real card: Archers vanished from the
    deck entirely, and the result looked plausible enough that only counting
    the cards in the montage caught it.
    """
    h, w = crop.shape[:2]
    badge = crop[int(h * 0.62):, int(w * 0.2):int(w * 0.8)]
    if badge.size == 0:
        return False
    hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
    magenta = cv2.inRange(hsv, np.array([135, 90, 90]), np.array([175, 255, 255]))
    return float(magenta.mean()) / 255.0 > 0.04


def _normalise(crop: np.ndarray) -> np.ndarray:
    """Grey, resized, contrast-normalised.

    Contrast normalisation is what makes a dimmed (unaffordable) icon cluster
    with its bright twin instead of forming a ninth group of its own.
    """
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (ICON_SHAPE[1], ICON_SHAPE[0]), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


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


def cluster(crops, features, out_dir: Path):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    _compact, labels, _centres = cv2.kmeans(
        features, DECK_SIZE, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    counts = []
    for k in range(DECK_SIZE):
        members = [c for c, lab in zip(crops, labels) if lab == k]
        counts.append(len(members))
        # Per-pixel median over the cluster: rejects the frames where a drag
        # animation or the cursor was over the slot, without detecting them.
        stack = np.stack([cv2.resize(m, (ICON_SHAPE[1], ICON_SHAPE[0])) for m in members])
        template = np.median(stack, axis=0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"cluster_{k}.png"), template)
        tiles.append(cv2.copyMakeBorder(template, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                        value=(0, 255, 0)))

    montage = cv2.resize(np.hstack(tiles), None, fx=3, fy=3,
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "clusters.png"), montage)
    print(f"wrote {out_dir / 'clusters.png'}")
    print("cluster sizes (left to right):", counts)
    print("\nNow look at clusters.png and re-run with, in that same order:")
    print("  --label <name0>,<name1>,...,<name7>")
    return labels


def label(names: list[str], out_dir: Path) -> None:
    import mapping

    if len(names) != DECK_SIZE:
        raise SystemExit(f"expected {DECK_SIZE} names, got {len(names)}")

    index = {}
    for k, name in enumerate(names):
        entry = mapping.resolve(name)  # raises on an unknown card
        if not entry.in_simulator:
            raise SystemExit(f"{name!r} has no simulator id -- see mapping/")
        source = out_dir / f"cluster_{k}.png"
        if not source.exists():
            raise SystemExit(f"missing {source}; run --cluster first")
        target = out_dir / f"card_{entry.sim_id}.png"
        cv2.imwrite(str(target), cv2.imread(str(source)))
        index[str(entry.sim_id)] = {"file": target.name, "name": entry.real_name}
        print(f"  cluster {k} -> {entry.real_name} (sim id {entry.sim_id})")

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
    parser.add_argument("--profile", type=Path,
                        default=_ROOT / "config" / "profile_gpg_1920x1080.json")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "config" / "templates" / "icons")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    if args.label:
        label([n.strip() for n in args.label.split(",")], args.out)
        return 0

    videos = args.video or sorted((_ROOT / "assets" / "recordings").glob("*.mp4"))
    crops, features = collect(videos, args.profile)
    print(f"collected {len(crops)} slot crops from {len(videos)} recording(s)")
    cluster(crops, features, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
