"""Cut the promo clips into the finished Short, timed to your voiceover.

Everything the edit does is in short.json beside this script: the script you
read (one entry per line), which clip plays when, the zooms, shakes, call-outs
and sound effects, and the caption style. Change it and run again.

    tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/edit_short.py
    ...edit_short.py --draft              half size and fast, for checking the cut
    ...edit_short.py --from 20 --to 32    only that stretch
    ...edit_short.py --still 14.5         one frame, as a PNG
    ...edit_short.py --timings            when every line, shot and effect lands

The voiceover (assets/voiceover.mp3) is the clock. faster-whisper finds when
each word is spoken, and every time in short.json can name a word instead of
a number ("cannon_drop.behind"), so a new take re-times the whole edit by
itself. With no voiceover yet, the edit is timed at a normal speaking pace and
rendered silent, as a preview.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]    # this folder, and tools/promo for common.py

from common import VideoWriter, ease_in_out, find_ffmpeg  # noqa: E402
from PIL import Image  # noqa: E402

import align  # noqa: E402
import media  # noqa: E402
import mix  # noqa: E402
from captions import Typesetter, back_out, parse_line, put, rgba, scaled, with_alpha  # noqa: E402
from overlays import SHAPES, arrow, ring  # noqa: E402

CACHE = HERE / "cache"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")

DEFAULT_STYLE = {
    "font": "assets/Montserrat-Black.ttf",
    "uppercase": True,
    "strip_punctuation": True,
    "caption_size": 84,
    "caption_y": 0.64,
    "caption_max_width": 0.8,
    "caption_color": "#FFFFFF",
    "highlight_color": "#FFD400",
    "stroke": 9,
    "stroke_color": "#000000",
    "shadow": 0.6,
    "max_words": 3,
    "caption_lead": 0.05,
    "caption_linger": 0.35,
    "pop_seconds": 0.14,
    "pop_from": 0.6,
    "callout_size": 150,
    "callout_y": 0.36,
    "callout_color": "#FFD400",
    "label_size": 30,
    "label_y": 0.115,
    "push": [1.0, 1.06],
    "punch_scale": 1.15,
    "sfx_db": -10,
    "cut_sfx_offset": -0.2,
    "sfx": {"cut": "whoosh", "callout": "pop", "counter": "ding", "shake": "impact",
            "punch": None, "flash": None, "freeze": None, "caption": None, "mark": None},
}
DEFAULT_MUSIC = {"below_voice_db": 10, "fade_out": 1.5, "swell_db": 4,
                 "duck": {"threshold": 0.04, "ratio": 5, "attack_ms": 20, "release_ms": 350}}
EFFECT_DEFAULTS = {
    "punch": {"duration": 0.35},
    "shake": {"duration": 0.4, "strength": 14},
    "flash": {"duration": 0.18, "color": "#FFFFFF"},
    "freeze": {"duration": 0.8},
    "callout": {"duration": 1.2},
    "counter": {"duration": 1.0, "hold": 1.0, "format": "{:.0f}"},
    "mark": {"duration": 1.5, "shape": "ring", "size": 160, "color": "#FF3B30"},
    "sfx": {"duration": 0.0},
}

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f"  ! {msg}")


# --- config ----------------------------------------------------------------------

def load_config(path):
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no config at {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}, line {exc.lineno}, column {exc.colno}: {exc.msg}\n"
                         "(usually a missing or extra comma, or a missing quote)")
    return _no_comments(cfg)


def _no_comments(v):
    """Keys starting with "_" are notes for people, not settings."""
    if isinstance(v, dict):
        return {k: _no_comments(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [_no_comments(x) for x in v]
    return v


def _merged(defaults, given):
    out = dict(defaults)
    for k, v in (given or {}).items():
        out[k] = _merged(defaults[k], v) if isinstance(defaults.get(k), dict) and isinstance(v, dict) else v
    return out


def _find_audio(path):
    """A path as written, or the same name with any audio extension (so
    voiceover.mp3 in the config also finds voiceover.wav)."""
    if path.exists():
        return path
    return next((p for p in (path.with_suffix(e) for e in AUDIO_EXTS) if p.exists()), None)


# --- the script and its timing -----------------------------------------------------

@dataclass
class Line:
    id: str
    words: list
    chunks: list
    caption: bool
    pause: float = 0.0
    times: list = field(default_factory=list)
    heard: list = field(default_factory=list)

    @property
    def start(self):
        return self.times[0][0]

    @property
    def end(self):
        return self.times[-1][1]


def load_lines(cfg, style):
    lines, seen = [], set()
    for n, entry in enumerate(cfg.get("lines") or [], 1):
        lid, text = entry.get("id"), entry.get("text")
        if not isinstance(lid, str) or not re.fullmatch(r"[a-z0-9_]+", lid):
            raise SystemExit(f"lines[{n}]: \"id\" must be lower-case letters, digits and _ "
                             f"(got {lid!r})")
        if lid in seen or lid == "end":
            raise SystemExit(f"lines[{n}]: the id {lid!r} is " + ("reserved" if lid == "end" else "used twice"))
        if not text or not text.strip():
            raise SystemExit(f"lines[{n}] ({lid}): no \"text\"")
        seen.add(lid)
        words, chunks = parse_line(text, style["max_words"])
        lines.append(Line(lid, words, chunks, entry.get("caption", True),
                          float(entry.get("pause_before", 0.0))))
    if not lines:
        raise SystemExit("short.json has no \"lines\"")
    return lines


def time_lines(lines, cfg, voice, log):
    """Word times from the voiceover, or estimated. Returns a one-line report."""
    words = [ln.words for ln in lines]
    if voice is None:
        for ln, times in zip(lines, align.estimate(words, [ln.pause for ln in lines])):
            ln.times, ln.heard = times, [False] * len(times)
        return "no voiceover yet: timed at a normal speaking pace"
    w = cfg.get("whisper", {})
    heard = align.transcribe(voice, w.get("model", "base.en"), w.get("language", "en"),
                             w.get("prompt"), CACHE, log)
    times, found = align.match(words, heard)
    for ln, t, f in zip(lines, times, found):
        ln.times, ln.heard = t, f
        if len(f) >= 3 and sum(f) < len(f) / 2:
            warn(f"line {ln.id!r}: only {sum(f)} of {len(f)} words were heard in the voiceover; "
                 "its timing is a guess (was it re-worded or cut?)")
    total, ok = sum(len(f) for f in found), sum(sum(f) for f in found)
    if total and ok < total * 0.3:
        warn("the voiceover barely matches the script: is it the right file?")
    return f"{ok} of {total} script words heard in the voiceover ({100 * ok / max(1, total):.0f}%)"


def apply_pauses(lines, recorded):
    """Make every "pause_before" real: the silence before that line is at
    least that long. In a recording that falls short, silence is inserted
    (so you can read straight through) and everything after it moves later.

    Returns ([(time in the ORIGINAL recording, seconds to insert)], the pauses
    as (start, end) on the edit's timeline, where the music swells)."""
    cuts, swells, shift = [], [], 0.0
    for i in range(1, len(lines)):
        prev, ln = lines[i - 1], lines[i]
        if ln.pause <= 0:
            continue
        gap = max(0.0, ln.start - prev.end)
        add = max(0.0, ln.pause - gap) if recorded else 0.0
        if add > 0:
            cuts.append((prev.end + gap / 2 - shift, add))
            for later in lines[i:]:
                later.times = [(a + add, b + add) for a, b in later.times]
            shift += add
        swells.append((prev.end, ln.start))
    return cuts, swells


class Clock:
    """Turns a time in short.json into seconds on the voiceover.

    12.5              seconds
    "hook"            when line "hook" starts; "hook.end" when it ends
    "hook.64"         when the word "64" is said in it ("hook.the#2": the 2nd "the")
    "end"             the end of the video
    any of those +/- an offset: "hook.64-0.2", "end-3"
    """

    def __init__(self, lines, end):
        self.lines = {ln.id: ln for ln in lines}
        self.end = end

    def __call__(self, spec, what):
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return float(spec)
        if not isinstance(spec, str) or not spec.strip():
            raise SystemExit(f"{what}: {spec!r} is not a time (a number, or \"line\" / \"line.word\")")
        m = re.fullmatch(r"(.*?)([+-]\d+(?:\.\d+)?)?", spec.strip())
        ref, off = m[1], float(m[2] or 0)
        if ref == "end":
            return self.end + off
        lid, _, word = ref.partition(".")
        line = self.lines.get(lid)
        if line is None:
            raise SystemExit(f"{what}: no line {lid!r}; the lines are: {', '.join(self.lines)}")
        if not word:
            return line.start + off
        if word == "end":
            return line.end + off
        word, _, nth = word.partition("#")
        key, n = align.normalize(word), int(nth or 1)
        hits = [i for i, w in enumerate(line.words) if align.normalize(w) == key]
        if len(hits) < n:
            raise SystemExit(f"{what}: line {lid!r} has no word {word!r}"
                             f"{f' #{n}' if nth else ''}; it says: {' '.join(line.words)}")
        return line.times[hits[n - 1]][0] + off


# --- shots, effects, captions ------------------------------------------------------

@dataclass
class Effect:
    kind: str
    at: float
    dur: float
    p: dict
    spec: str
    sfx: str | None
    sfx_db: float


@dataclass
class Shot:
    n: int
    name: str
    path: Path | None
    info: media.ClipInfo | None
    start: float
    end: float = 0.0
    src_in: float = 0.0
    speed: float = 1.0
    zoom: tuple = (1.0, 1.0)
    focus: tuple = ((0.5, 0.5), (0.5, 0.5))
    overscan: float = 0.0
    label: str | None = None
    sfx: str | None = None
    spec: dict = field(default_factory=dict)
    freezes: list = field(default_factory=list)

    def held(self, t):
        return sum(min(max(0.0, t - a), d) for a, d in self.freezes)

    def src_time(self, t):
        return self.src_in + self.speed * (t - self.start - self.held(t))


@dataclass
class Caption:
    start: float
    end: float
    words: list


def load_effects(cfg, clock, style):
    out = []
    for n, e in enumerate(cfg.get("effects") or [], 1):
        kind = e.get("type")
        what = f"effects[{n}] ({kind})"
        if kind not in EFFECT_DEFAULTS:
            raise SystemExit(f"effects[{n}]: \"type\" must be one of {', '.join(EFFECT_DEFAULTS)}")
        p = {**EFFECT_DEFAULTS[kind], **e}
        at = clock(e.get("at"), what)
        dur = clock(e["until"], what) - at if "until" in e else float(p["duration"])
        if dur < 0:
            raise SystemExit(f"{what}: \"until\" is before \"at\"")
        need = {"callout": ["text"], "counter": ["from", "to"], "sfx": ["sound"], "mark": ["pos"]}
        if kind == "mark":
            if p["shape"] not in SHAPES:
                raise SystemExit(f"{what}: \"shape\" must be one of {', '.join(SHAPES)}")
            if p["shape"] == "arrow":
                need["mark"].append("from")
        for key in need.get(kind, []):
            if key not in e:
                raise SystemExit(f"{what}: needs \"{key}\"")
        sfx = e["sound"] if kind == "sfx" else e.get("sfx", style["sfx"].get(kind))
        out.append(Effect(kind, at, dur, p, str(e.get("at")), sfx, float(e.get("sfx_db", style["sfx_db"]))))
    return sorted(out, key=lambda x: x.at)


def _sidecar(path, suffix):
    side = path.with_name(path.stem + suffix)
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    return None


def load_shots(cfg, clock, duration, clips_dir, ffmpeg, style, effects):
    raw = cfg.get("shots") or []
    if not raw:
        raise SystemExit("short.json has no \"shots\"")
    shots = []
    for n, s in enumerate(raw, 1):
        what = f"shots[{n}] ({s.get('clip')})"
        if "clip" not in s:
            raise SystemExit(f"shots[{n}]: needs \"clip\" (a file in {clips_dir})")
        start = 0.0 if n == 1 else clock(s.get("at"), what)
        path = (clips_dir / s["clip"]).resolve()
        info = None
        if not path.exists():
            warn(f"{what}: {path} does not exist yet; a placeholder card stands in")
        else:
            try:
                info = media.probe(ffmpeg, path)
            except SystemExit:
                warn(f"{what}: {path} cannot be read (still rendering, or damaged?); "
                     "a placeholder card stands in")
        zoom = s.get("zoom", style["push"])
        zoom = (float(zoom), float(zoom)) if isinstance(zoom, (int, float)) else tuple(map(float, zoom))
        focus = s.get("focus", [0.5, 0.5])
        focus = (tuple(focus[0]), tuple(focus[1])) if isinstance(focus[0], list) else (tuple(focus), tuple(focus))
        cont = s.get("in") == "continue"
        if cont and "sync" in s and s["sync"].get("fit") != "stretch":
            raise SystemExit(f"{what}: \"in\": \"continue\" and a shifting \"sync\" both set where "
                             "the clip starts (use \"fit\": \"stretch\" to sync a continued shot)")
        shots.append(Shot(n, s["clip"], path if info else None, info, start,
                          src_in=0.0 if cont else float(s.get("in", 0.0)),
                          speed=float(s.get("speed", 1.0)),
                          zoom=zoom, focus=focus, overscan=float(s.get("overscan", 0.0)), spec=s,
                          sfx=s.get("sfx", style["sfx"]["cut"]) if n > 1 else None))
    for a, b in zip(shots, shots[1:]):
        if b.start <= a.start:
            raise SystemExit(f"shots[{b.n}] starts at {b.start:.2f} s, not after shots[{a.n}] "
                             f"({a.start:.2f} s): shots must be in time order")
    for a, b in zip(shots, shots[1:] + [None]):
        a.end = b.start if b else duration
    for e in effects:
        if e.kind == "freeze":
            shot = shot_at(shots, e.at)
            shot.freezes.append((e.at, e.dur))
    for prev, s in zip([None] + shots[:-1], shots):
        if s.spec.get("in") == "continue":
            # The same clip, picked up where the last shot left it: a jump cut
            # to a new framing that does not replay anything.
            if prev is None or prev.name != s.name:
                raise SystemExit(f"shots[{s.n}]: \"in\": \"continue\" needs the shot before "
                                 "it to show the same clip")
            s.src_in = prev.src_time(s.start)
        _sync(s, clock)
        _label(s)
        if s.info:
            over = s.src_time(s.end) - s.info.duration
            if over > 0.3:
                warn(f"shots[{s.n}] ({s.name}) runs {over:.1f} s past the end of its clip, "
                     "which holds its last frame (start the next shot sooner, or slow this one)")
    return shots


def _sync(shot, clock):
    """"sync": {"beat": "cannon_drop" | "clip": 3.67, "to": "line.word"}: land that
    moment of the clip on that moment of the voiceover, by shifting "in"
    (default) or, with "fit": "stretch", by setting "speed"."""
    sync = shot.spec.get("sync")
    if not sync:
        return
    what = f"shots[{shot.n}] ({shot.name}) sync"
    if "beat" in sync:
        beats = _sidecar(shot.path, ".beats.json") if shot.path else None
        if beats is None:
            if shot.path:
                warn(f"{what}: no {shot.path.stem}.beats.json beside the clip; not synced")
            return
        if sync["beat"] not in beats:
            raise SystemExit(f"{what}: no beat {sync['beat']!r}; the clip has "
                             f"{', '.join(k for k in beats if k != 'fps')}")
        beat = float(beats[sync["beat"]]["seconds"])
    elif "clip" in sync:
        beat = float(sync["clip"])
    else:
        raise SystemExit(f"{what}: needs \"beat\" (a name from the clip's beats file) or \"clip\" (seconds)")
    to = clock(sync.get("to"), what)
    if not shot.start <= to <= shot.end + 1e-6:   # the end: the cut into its own continuation
        warn(f"{what}: {sync.get('to')} ({to:.2f} s) is outside the shot "
             f"({shot.start:.2f}-{shot.end:.2f} s), so the beat will not be on screen")
    local = to - shot.start - shot.held(to)
    if sync.get("fit", "shift") == "stretch":
        speed = (beat - shot.src_in) / max(0.05, local)
        shot.speed = min(4.0, max(0.25, speed))
        if shot.speed != speed:
            warn(f"{what}: stretching would need speed {speed:.2f}; clamped to {shot.speed:.2f}")
    else:
        shot.src_in = beat - shot.speed * local
        if shot.src_in < -0.05:
            warn(f"{what}: to hit the beat, the clip would have to start {-shot.src_in:.1f} s "
                 "before its first frame, which is held instead (start the shot later, or "
                 "use \"fit\": \"stretch\")")


def _label(shot):
    """The honesty label: from the clip's sidecar unless the shot says otherwise."""
    side = _sidecar(shot.path, ".clip.json") if shot.path else None
    has = side and side.get("label")
    burned = bool(side and side.get("burned"))
    if "label" in shot.spec:
        shot.label = shot.spec["label"]
        if has and not shot.label and not burned:
            warn(f"shots[{shot.n}] ({shot.name}): its label {side['label']!r} is switched off")
    elif has and not burned:
        shot.label = side["label"]
    if burned:
        warn(f"shots[{shot.n}] ({shot.name}) has its label burned into the video, where a zoom "
             "can crop it: re-render it with scenes.py --no-label")


def shot_at(shots, t):
    for s in reversed(shots):
        if t >= s.start:
            return s
    return shots[0]


def load_captions(lines, style):
    out = []
    starts = [ln.start for ln in lines[1:]] + [math.inf]
    for ln, next_line in zip(lines, starts):
        if not ln.caption:
            continue
        chunk_starts = [ln.times[c[0][0]][0] - style["caption_lead"] for c in ln.chunks]
        for i, c in enumerate(ln.chunks):
            end = (chunk_starts[i + 1] if i + 1 < len(ln.chunks)
                   else min(ln.end + style["caption_linger"], next_line - style["caption_lead"]))
            words = [(_display(w, style), hl) for _, w, hl in c]
            words = [(w, hl) for w, hl in words if w]
            if words and end > chunk_starts[i]:
                out.append(Caption(chunk_starts[i], end, words))
    return out


def _display(word, style):
    return word.rstrip(".,!?;:…—") if style["strip_punctuation"] else word


# --- the picture --------------------------------------------------------------------

class Frames:
    def __init__(self, size, fps, shots, effects, captions, typeset, style, ffmpeg, draft):
        self.W, self.H = size
        self.fps, self.shots, self.effects, self.captions = fps, shots, effects, captions
        self.ts, self.st, self.ffmpeg = typeset, style, ffmpeg
        self.k = self.W / 1080.0
        self.resample = Image.BILINEAR if draft else Image.BICUBIC

    def reader(self, shot, t):
        if shot.info is None:
            card = self.ts.card((self.W, self.H), ["clip not rendered yet:", shot.name])
            return media.StillClip(card)
        return media.ClipReader(self.ffmpeg, shot.path, shot.info, (self.W, self.H),
                                start=shot.src_time(t))

    def camera(self, shot, t):
        u = (t - shot.start) / max(1e-6, shot.end - shot.start)
        e = ease_in_out(u)
        scale = shot.zoom[0] + (shot.zoom[1] - shot.zoom[0]) * e
        (ax, ay), (bx, by) = shot.focus
        fx, fy = ax + (bx - ax) * e, ay + (by - ay) * e
        dx = dy = 0.0
        for ef in self.effects:
            dt = t - ef.at
            if ef.kind == "punch":
                k = _punch(dt, ef)
                if k > 0:
                    scale *= 1 + (float(ef.p.get("scale", self.st["punch_scale"])) - 1) * k
                    if "focus" in ef.p:
                        fx += (ef.p["focus"][0] - fx) * k
                        fy += (ef.p["focus"][1] - fy) * k
            elif ef.kind == "shake" and 0 <= dt < ef.dur:
                amp = float(ef.p["strength"]) * self.k * (1 - dt / ef.dur) ** 2
                rnd = random.Random(round(t * 1000))
                dx += amp * (2 * rnd.random() - 1)
                dy += amp * (2 * rnd.random() - 1)
        return scale, fx, fy, dx, dy

    def frame(self, shot, reader, t):
        base = reader.frame(shot.src_time(t))
        box = crop_box(self.W, self.H, *self.camera(shot, t), shot.overscan)
        img = base.copy() if box is None else _crop(base, box, (self.W, self.H), self.resample)
        for ef in self.effects:
            if ef.kind == "flash" and 0 <= t - ef.at < ef.dur:
                a = 0.85 * (1 - (t - ef.at) / ef.dur)
                img = Image.blend(img, Image.new("RGB", img.size, rgba(ef.p["color"])[:3]), a)
        if shot.label:
            pill = self.ts.label(shot.label)
            img.paste(pill, (round(0.04 * self.W), round(self.st["label_y"] * self.H)), pill)
        for ef in self.effects:
            if ef.kind == "mark":
                self._mark(img, ef, t, box)
        for ef in self.effects:
            self._overlay(img, ef, t)
        cap = next((c for c in reversed(self.captions) if c.start <= t < c.end), None)
        if cap:
            put(img, scaled(self.ts.caption(cap.words), self._pop(t - cap.start)),
                self.W / 2, float(shot.spec.get("caption_y", self.st["caption_y"])) * self.H)
        return img

    def _mark(self, img, ef, t, box):
        dt = t - ef.at
        if not 0 <= dt < ef.dur:
            return
        x, y, s = to_screen(box, self.W, self.H, *ef.p["pos"])
        if ef.p["shape"] == "arrow":
            tail = to_screen(box, self.W, self.H, *ef.p["from"])[:2]
            arrow(img, (x, y), tail, float(ef.p["size"]) * self.k, ef.p["color"], dt, ef.dur, self.k)
        else:   # a ring hugs its target, so it zooms with the camera
            ring(img, (x, y), float(ef.p["size"]) * self.k * s, ef.p["color"], dt, ef.dur, self.k)

    def _pop(self, dt):
        u = min(1.0, dt / self.st["pop_seconds"])
        a = self.st["pop_from"]
        return a + (1 - a) * back_out(u)

    def _overlay(self, img, ef, t):
        dt = t - ef.at
        if ef.kind == "callout":
            total, text = ef.dur, ef.p["text"]
        elif ef.kind == "counter":
            total = ef.dur + float(ef.p["hold"])
            u = 1.0 if ef.dur <= 0 else min(1.0, dt / ef.dur)
            v = ef.p["from"] + (ef.p["to"] - ef.p["from"]) * (1 - (1 - u) ** 3)
            text = ef.p.get("prefix", "") + ef.p["format"].format(v) + ef.p.get("suffix", "")
        else:
            return
        if not 0 <= dt < total:
            return
        colour = ef.p.get("color", self.st["callout_color"])
        pic = self.ts.callout(text, colour, float(ef.p.get("size", self.st["callout_size"])))
        out = max(0.0, 1 - (total - dt) / 0.12)      # the last 0.12 s shrink away
        pic = with_alpha(scaled(pic, self._pop(dt) * (1 - 0.3 * out)), 1 - out)
        put(img, pic, self.W / 2, float(ef.p.get("y", self.st["callout_y"])) * self.H)


def _punch(dt, ef):
    """0..1: how far into a punch-in its zoom is at dt seconds."""
    attack = 0.06
    if dt < 0:
        return 0.0
    if dt < attack:
        return ease_in_out(dt / attack)
    if ef.p.get("hold"):
        if dt < ef.dur:
            return 1.0
        return max(0.0, 1 - ease_in_out((dt - ef.dur) / 0.2))
    if dt < ef.dur:
        return 1 - ease_in_out((dt - attack) / max(1e-6, ef.dur - attack))
    return 0.0


def crop_box(W, H, scale, fx, fy, dx, dy, overscan=0.0):
    """The part of the frame the camera shows: zoom `scale` toward the focus
    point (0..1 of the frame), shaken by (dx, dy) output pixels. It stays
    inside the frame, or up to `overscan` (a fraction of it) past an edge,
    which slides the picture along and shows black beyond. None = the whole
    frame, untouched."""
    s = max(1.0, scale, 1 + 2 * abs(dx) / W + 1e-3 if dx else 1.0, 1 + 2 * abs(dy) / H + 1e-3 if dy else 1.0)
    cw, ch = W / s, H / s
    ox, oy = overscan * W, overscan * H

    def clamp(c, half, size, o):
        return min(max(c, half - o), size - half + o)

    cx = clamp(clamp(fx * W, cw / 2, W, ox) + dx / s, cw / 2, W, ox)
    cy = clamp(clamp(fy * H, ch / 2, H, oy) + dy / s, ch / 2, H, oy)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    if max(abs(box[0]), abs(box[1]), abs(box[2] - W), abs(box[3] - H)) < 1e-3:
        return None
    return box


def _crop(frame, box, size, resample):
    """Resample `box` of the frame to `size`. The box may leave the frame
    (overscan): crop() fills outside with black, and the resize keeps the
    sub-pixel position, so a slow push does not shimmer."""
    x0, y0, x1, y1 = box
    ix0, iy0 = math.floor(x0), math.floor(y0)
    region = frame.crop((ix0, iy0, math.ceil(x1), math.ceil(y1)))
    return region.resize(size, resample, box=(x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0))


def to_screen(box, W, H, x, y):
    """A point of the clip (0..1 of its frame) -> (screen x, screen y, zoom)."""
    if box is None:
        return x * W, y * H, 1.0
    x0, y0, x1, _ = box
    s = W / (x1 - x0)
    return (x * W - x0) * s, (y * H - y0) * s, s


# --- output ---------------------------------------------------------------------------

def render(frames, f0, f1, path, crf):
    total = f1 - f0
    t0 = last = time.time()
    reader, current = None, None
    with VideoWriter(path, (frames.W, frames.H), frames.fps, frames.ffmpeg, crf) as vid:
        for i in range(f0, f1):
            t = i / frames.fps
            shot = shot_at(frames.shots, t)
            if shot is not current:
                if reader:
                    reader.close()
                reader, current = frames.reader(shot, t), shot
            vid.write(frames.frame(shot, reader, t))
            if time.time() - last > 5 or i == f1 - 1:
                last = time.time()
                done = i - f0 + 1
                eta = (last - t0) / done * (total - done)
                print(f"  {done}/{total} frames ({100 * done // total}%), {eta:.0f} s left", flush=True)
    if reader:
        reader.close()


def audio_events(shots, effects, captions, style):
    ev = [(s.start + style["cut_sfx_offset"], s.sfx, style["sfx_db"]) for s in shots if s.sfx]
    for e in effects:
        if e.sfx:
            ev.append((e.at + (e.dur if e.kind == "counter" else 0.0), e.sfx, e.sfx_db))
    if style["sfx"].get("caption"):
        ev += [(c.start, style["sfx"]["caption"], style["sfx_db"]) for c in captions]
    return ev


def print_timings(report, voice, duration, lines, shots, effects, captions, swells=()):
    print(f"\nvoiceover  {voice.name if voice else 'none'}: {report}")
    print(f"length     {duration:.2f} s\n\nLINES (word times; ~ = not heard, estimated)")
    for ln in lines:
        note = f"   (after a {ln.pause:g} s pause)" if ln.pause else ""
        print(f"  {ln.start:6.2f}-{ln.end:6.2f}  {ln.id}{note}")
        print("        " + "  ".join(f"{w}{'' if h else '~'}@{t[0]:.2f}"
                                    for w, t, h in zip(ln.words, ln.times, ln.heard)))
    print("\nSHOTS")
    for s in shots:
        src = f"clip {s.src_time(s.start):.2f}-{s.src_time(s.end):.2f}"
        if s.info:
            src += f" of {s.info.duration:.2f} s"
        extra = f"  x{s.speed:.2f}" if abs(s.speed - 1) > 1e-3 else ""
        print(f"  {s.start:6.2f}-{s.end:6.2f}  #{s.n} {s.name:<40} {src}{extra}"
              f"{'  [' + s.label + ']' if s.label else ''}")
    print("\nEFFECTS")
    for e in effects:
        what = e.p.get("text") or (f"{e.p['from']} -> {e.p['to']}" if e.kind == "counter" else "")
        print(f"  {e.at:6.2f}  {e.kind:<8} {what:<24} at {e.spec}{'  sfx ' + e.sfx if e.sfx else ''}")
    if swells:
        print("\nPAUSES (music swells)\n  " + "  ".join(f"{a:.2f}-{b:.2f}" for a, b in swells))
    print("\nCAPTIONS")
    for c in captions:
        print(f"  {c.start:6.2f}-{c.end:6.2f}  " + " ".join(f"*{w}*" if h else w for w, h in c.words))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "short.json"))
    ap.add_argument("--draft", action="store_true", help="half size, fast encode")
    ap.add_argument("--from", dest="t0", type=float, default=None, help="render from this second")
    ap.add_argument("--to", dest="t1", type=float, default=None, help="render up to this second")
    ap.add_argument("--still", type=float, default=None, help="write one frame at this second as a PNG")
    ap.add_argument("--timings", action="store_true", help="print the timeline and stop")
    ap.add_argument("--voiceover", default=None, help="use this recording instead of the config's")
    ap.add_argument("--music", default=None, help="use this music instead of the config's")
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    style = _merged(DEFAULT_STYLE, cfg.get("style"))
    out_cfg = {"file": "../out/short.mp4", "size": [1080, 1920], "fps": 30, "crf": 18,
               "tail": 1.5, **cfg.get("output", {})}
    ffmpeg = find_ffmpeg(args.ffmpeg)

    voice = Path(args.voiceover) if args.voiceover else _find_audio(base / cfg.get("voiceover", "assets/voiceover.mp3"))
    if voice is not None and not voice.exists():
        raise SystemExit(f"no voiceover at {voice}")
    if voice is None:
        warn("no voiceover yet (assets/voiceover.mp3): timed at a normal speaking pace, silent")
    music_cfg = _merged(DEFAULT_MUSIC, cfg.get("music") if isinstance(cfg.get("music"), dict) else {})
    music_file = args.music or (cfg.get("music") or {}).get("file")
    music = Path(music_file) if args.music else (_find_audio(base / music_file) if music_file else None)
    if music_file and music is None:
        warn(f"no music at {base / music_file}; the edit has none")

    lines = load_lines(cfg, style)
    report = time_lines(lines, cfg, voice, print)
    cuts, swells = apply_pauses(lines, voice is not None)
    if cuts:
        report += f"; {sum(c[1] for c in cuts):.1f} s of silence added for the pauses"
    # The video ends `tail` seconds after the last word, whatever silence the
    # recording trails, unless the config gives a length.
    spoken_end = max(ln.end for ln in lines)
    duration = float(out_cfg.get("duration") or spoken_end + float(out_cfg["tail"]))
    clock = Clock(lines, duration)
    effects = load_effects(cfg, clock, style)
    shots = load_shots(cfg, clock, duration, (base / cfg.get("clips_dir", "../out")).resolve(),
                       ffmpeg, style, effects)
    captions = load_captions(lines, style)

    if args.timings:
        print_timings(report, voice, duration, lines, shots, effects, captions, swells)
        return

    W, H = (int(v) for v in out_cfg["size"])
    if args.draft:
        W, H = W // 4 * 2, H // 4 * 2
    fps = int(out_cfg["fps"])
    typeset = Typesetter(base / style["font"], W, style, warn)
    frames = Frames((W, H), fps, shots, effects, captions, typeset, style, ffmpeg, args.draft)

    out = (base / out_cfg["file"]).resolve()
    stem = out.stem + ("_draft" if args.draft else "")
    if args.still is not None:
        shot = shot_at(shots, args.still)
        reader = frames.reader(shot, args.still)
        path = out.with_name(f"{stem}_still_{args.still:g}s.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        frames.frame(shot, reader, args.still).save(path)
        reader.close()
        print(f"wrote {path}")
        return

    t0 = max(0.0, args.t0 or 0.0)
    t1 = min(duration, args.t1 if args.t1 is not None else duration)
    if t1 <= t0:
        raise SystemExit(f"--from {t0:g} --to {t1:g}: nothing to render (the edit is {duration:.1f} s)")
    if args.t0 is not None or args.t1 is not None:
        stem += f"_{t0:g}-{t1:g}s"
    f0, f1 = round(t0 * fps), round(t1 * fps)
    print(f"\n{report}\nrendering {(f1 - f0) / fps:.1f} s of {duration:.1f} s at {W}x{H}, "
          f"{len(shots)} shots, {len(captions)} captions")
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    silent = CACHE / "video.mp4"
    render(frames, f0, f1, silent, int(out_cfg["crf"]) + (5 if args.draft else 0))

    sounds = mix.Sounds(ffmpeg, base / "assets" / "sfx")
    fx = mix.effects_track(audio_events(shots, effects, captions, style), duration, sounds)
    if cuts:
        voice = mix.insert_silence(ffmpeg, voice, cuts, CACHE / "voiceover_paced.wav")
    audio = mix.mix(ffmpeg, CACHE, duration, voice, music, music_cfg, fx, swells)
    final = out.with_name(stem + ".mp4")
    final.parent.mkdir(parents=True, exist_ok=True)
    mix.mux(ffmpeg, silent, audio, f0 / fps, (f1 - f0) / fps, final)
    print(f"\nwrote {final} ({final.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f} s")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
