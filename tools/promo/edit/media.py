"""Video in: probe a clip and read its frames in order, through ffmpeg."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from PIL import Image


@dataclass
class ClipInfo:
    width: int
    height: int
    fps: float
    duration: float


def _banner(ffmpeg, path):
    return subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True,
                          text=True, errors="replace").stderr


def _duration(err):
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    return None if m is None else int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def probe(ffmpeg, path):
    """A clip's size, frame rate and length, read from ffmpeg's banner (the
    winget build ships ffprobe, but imageio-ffmpeg does not)."""
    err = _banner(ffmpeg, path)
    size = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", err)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", err)
    dur = _duration(err)
    if size is None or dur is None:
        raise SystemExit(f"cannot read {path} as a video:\n{err.strip()[-400:]}")
    return ClipInfo(int(size[1]), int(size[2]), float(fps[1]) if fps else 30.0, dur)


def audio_duration(ffmpeg, path):
    dur = _duration(_banner(ffmpeg, path))
    if dur is None:
        raise SystemExit(f"cannot read {path} as audio")
    return dur


class ClipReader:
    """One clip's frames, scaled and centre-cropped to fill `size`, read
    forward from `start` seconds. frame(t) takes clip time and must never go
    backwards; before the first frame it shows the first, past the end the last.
    """

    def __init__(self, ffmpeg, path, info, size, start=0.0):
        self.w, self.h = size
        self.fps = info.fps
        self.start = min(max(0.0, start), max(0.0, info.duration - 1.0 / info.fps))
        vf = (f"scale={self.w}:{self.h}:force_original_aspect_ratio=increase:flags=bicubic,"
              f"crop={self.w}:{self.h}")
        cmd = [ffmpeg, "-v", "error", "-ss", f"{self.start:.3f}", "-i", str(path), "-an",
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.nbytes = self.w * self.h * 3
        self.index = -1
        self.image = None
        self.path = path

    def frame(self, t):
        want = max(0, int((t - self.start) * self.fps + 1e-6))
        while self.index < want:
            buf = self.proc.stdout.read(self.nbytes)
            if len(buf) < self.nbytes:
                break
            self.image = Image.frombytes("RGB", (self.w, self.h), buf)
            self.index += 1
        if self.image is None:
            raise SystemExit(f"ffmpeg returned no frames from {self.path} at {self.start:.2f} s")
        return self.image

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.stdout.close()
        self.proc.wait()


class StillClip:
    """Stands in for a clip that is not there yet."""

    def __init__(self, image):
        self.image = image

    def frame(self, t):
        return self.image

    def close(self):
        pass
