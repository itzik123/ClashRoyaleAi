"""Sample a recording, run CRBAB's unit detector, project to engine tiles.

The detector wants a 720x1280 emulator screenshot; the recording is a
1920x1080 desktop capture with the emulator at EMU. So: crop, resize, detect,
map the bbox back to desktop pixels, then use perception's OWN calibrated
homography for the tile mapping -- never CRBAB's TILE_* constants, which were
measured on a different emulator.
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

_PERCEPTION = Path(__file__).resolve().parent.parent
if str(_PERCEPTION) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION))

# CRBAB's package __init__ imports its Bot, which imports `keyboard` -- a
# dependency that exists only to drive a live emulator and has nothing to do
# with running the detector over a file. Stubbing the package lets
# `clashroyalebuildabot.detectors...` resolve without it, rather than adding a
# requirement to this venv for a module that is never called.
sys.modules.setdefault("keyboard", types.ModuleType("keyboard"))
if "clashroyalebuildabot" not in sys.modules:
    _pkg = types.ModuleType("clashroyalebuildabot")
    _pkg.__path__ = [str(_PERCEPTION / "clashroyalebuildabot")]
    sys.modules["clashroyalebuildabot"] = _pkg

from clashroyalebuildabot.constants import MODELS_DIR, DETECTOR_UNITS
from clashroyalebuildabot.detectors.unit_detector import UnitDetector
from calib.homography import load_profile, homography_from_profile

EMU = (686, 40, 1236, 1012)          # desktop pixels of the emulator window
DET_W, DET_H = 720, 1280             # what the model was trained on

PROFILE = _PERCEPTION / "config" / "profile_gpg_1920x1080.json"


class _AllCards:
    """UnitDetector uses `cards` only to decide which names CAN be ours.

    We do not know the deck up front, so allow every unit and let the side
    detector make the call from pixels for all of them.
    """
    def __init__(self):
        self.units = list(DETECTOR_UNITS)


def build_detector():
    return UnitDetector(str(Path(MODELS_DIR) / "units_M_480x352.onnx"),
                        [_AllCards()])


def detect_frame(det, frame_bgr, homo):
    crop = frame_bgr[EMU[1]:EMU[3], EMU[0]:EMU[2]]
    ch, cw = crop.shape[:2]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((DET_W, DET_H))
    ally, enemy = det.run(img)
    out = []
    for side, group in (("ally", ally), ("enemy", enemy)):
        for d in group:
            b = d.position.bbox
            # detector space -> emulator crop -> desktop pixels
            px = (b[0] + b[2]) / 2 * cw / DET_W + EMU[0]
            py = b[3] * ch / DET_H + EMU[1]           # feet, not centre
            tx, ty = homo.screen_to_tile(np.array([[px, py]]))[0]
            out.append(dict(name=d.unit.name, side=side, conf=round(float(d.position.conf), 3),
                            px=round(float(px), 1), py=round(float(py), 1),
                            x=round(float(tx), 2), y=round(float(ty), 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    a = ap.parse_args()

    # The homography is used ONLY to keep a px->tile field in the output for
    # reference. Everything downstream re-maps from the pixels through
    # arena_anchor, because this profile's tile frame is stale -- see
    # UPSTREAM_REQUESTS.md item 28.
    homo = homography_from_profile(load_profile(PROFILE), (1080, 1920))
    det = build_detector()

    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / a.fps))
    print(f"src {src_fps:.4f} fps, {total} frames, sampling every {step}", flush=True)

    frames, i, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        if i % step == 0 and a.start <= t <= a.end:
            frames.append(dict(t=round(t, 3), units=detect_frame(det, frame, homo)))
            if len(frames) % 25 == 0:
                print(f"  t={t:6.1f}s  {len(frames)} sampled  "
                      f"{time.time()-t0:.0f}s elapsed", flush=True)
        i += 1
    cap.release()

    Path(a.out).write_text(json.dumps(
        dict(video=a.video, fps=a.fps, src_fps=src_fps, frames=frames), indent=0))
    print(f"wrote {a.out}: {len(frames)} frames, "
          f"{sum(len(f['units']) for f in frames)} detections")


if __name__ == "__main__":
    main()
