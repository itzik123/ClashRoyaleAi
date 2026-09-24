"""Sound: effects, the ducked music bed, loudness, and the final mix.

Levels come from MEASURED loudness, not guessed gains. The voiceover is
brought to -16 LUFS and the music to `below_voice_db` under that, then the
music is ducked (sidechain-compressed by the voice) wherever you speak. The
finished mix is brought to -14 LUFS, YouTube's playback level, by measured
gain and a peak limiter.

Sound effects are built in (whoosh, pop, impact, ding) and synthesised here;
drop a file with the same name into assets/sfx/ (whoosh.wav, ding.mp3, ...) to
use your own instead, or any new name ("scratch") to add one.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 48000
VOICE_LUFS = -16.0
TARGET_LUFS = -14.0
PEAK = 10 ** (-2.0 / 20)        # the limiter's ceiling, -2 dBFS: under -1 dBTP after encoding
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".m4a", ".flac")
BUILTIN = ("whoosh", "pop", "impact", "ding", "wrong", "right")


def _run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n  {' '.join(map(str, cmd))}\n{r.stderr.strip()[-800:]}")
    return r.stderr


def decode(ffmpeg, path):
    """Any audio file -> float32 stereo at 48 kHz, shape (samples, 2)."""
    r = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2",
                        "-ar", str(SR), "-"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot decode {path}: {r.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2)


def loudness(ffmpeg, path):
    """Integrated loudness in LUFS (EBU R128); -70 for silence."""
    err = _run([ffmpeg, "-hide_banner", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"])
    hits = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", err)
    return -70.0 if not hits or hits[-1] == "-inf" else max(-70.0, float(hits[-1]))


# --- sound effects -------------------------------------------------------------

def synth(name):
    """A built-in effect as float32 stereo, or None for an unknown name."""
    rng = np.random.default_rng(7)
    if name == "whoosh":
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        u = t / t[-1]
        # Noise through a lowpass whose cutoff sweeps up and back down.
        cut = 250.0 * 20.0 ** np.sin(np.pi * u)
        a = np.exp(-2 * np.pi * cut / SR)
        x, y, acc = rng.standard_normal(n), np.empty(n), 0.0
        for i in range(n):
            acc = a[i] * acc + (1 - a[i]) * x[i]
            y[i] = acc
        y *= u ** 1.3 * (1 - u) ** 0.8
        stereo = np.stack([y * (1.2 - u), y * (0.2 + u)], axis=1)
    elif name == "pop":
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        f = 260 + 900 * np.exp(-t * 35)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)
        stereo = np.stack([y, y], axis=1)
    elif name == "impact":
        n = int(0.7 * SR)
        t = np.arange(n) / SR
        f = 42 + 120 * np.exp(-t * 14)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5)
        hit = np.convolve(rng.standard_normal(n), np.ones(24) / 24, "same") * np.exp(-t * 40)
        y = body + 0.6 * hit
        stereo = np.stack([y, y], axis=1)
    elif name == "ding":
        n = int(1.2 * SR)
        t = np.arange(n) / SR
        y = sum(g * np.sin(2 * np.pi * f * t) for f, g in ((1568.0, 1.0), (3136.0, 0.3), (2349.3, 0.2)))
        y *= np.exp(-t * 4.5) * np.minimum(1.0, t / 0.003)
        stereo = np.stack([y, y], axis=1)
    elif name == "wrong":
        # A game-show buzzer: two detuned, softened square waves.
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        y = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (147.0, 155.0))
        y = np.convolve(y, np.ones(12) / 12, "same") * np.minimum(1.0, t / 0.01) * np.exp(-t * 2.5)
        stereo = np.stack([y, y], axis=1)
    elif name == "right":
        # Two rising chimes.
        n = int(0.55 * SR)
        t = np.arange(n) / SR
        y = np.zeros(n)
        for start, f in ((0.0, 1046.5), (0.11, 1568.0)):
            u = np.clip(t - start, 0, None)
            y += (t >= start) * np.sin(2 * np.pi * f * u) * np.exp(-u * 7) * np.minimum(1.0, u / 0.004)
        stereo = np.stack([y, y], axis=1)
    else:
        return None
    return (stereo * (0.7 / np.abs(stereo).max())).astype(np.float32)


class Sounds:
    """Effects by name: assets/sfx/<name>.* if you put one there, else built in."""

    def __init__(self, ffmpeg, folder):
        self.ffmpeg, self.folder, self._cache = ffmpeg, Path(folder), {}

    def __call__(self, name):
        if name not in self._cache:
            own = [self.folder / f"{name}{ext}" for ext in AUDIO_EXTS]
            own = next((p for p in own if p.exists()), None)
            sound = decode(self.ffmpeg, own) if own else synth(name)
            if sound is None:
                raise SystemExit(f"no sound effect {name!r}: put {name}.wav (or .mp3) in "
                                 f"{self.folder}, or use a built-in one: {', '.join(BUILTIN)}")
            self._cache[name] = sound
        return self._cache[name]


def effects_track(events, duration, sounds):
    """events: [(seconds, name, gain dB)] -> one float32 stereo track."""
    out = np.zeros((int(round(duration * SR)) + 1, 2), dtype=np.float32)
    for t, name, db in events:
        s = sounds(name) * (10 ** (db / 20))
        i = int(round(t * SR))
        a, b = max(0, i), min(len(out), i + len(s))
        if a < b:
            out[a:b] += s[a - i:b - i]
    return out


# --- the mix -------------------------------------------------------------------

def insert_silence(ffmpeg, voice, cuts, out):
    """The voiceover with `seconds` of silence inserted at each (time, seconds)
    cut, written to `out` (a WAV)."""
    audio = decode(ffmpeg, voice)
    parts, last = [], 0
    for at, seconds in sorted(cuts):
        i = int(round(at * SR))
        parts += [audio[last:i], np.zeros((int(round(seconds * SR)), 2), np.float32)]
        last = i
    parts.append(audio[last:])
    raw = Path(out).with_suffix(".f32")
    np.concatenate(parts).astype(np.float32).tofile(raw)
    _run([ffmpeg, "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", raw,
          "-c:a", "pcm_f32le", out])
    return out


def mix(ffmpeg, workdir, duration, voice, music, music_cfg, effects, swells=(), log=print):
    """voice and music are paths or None; effects a float32 stereo array;
    swells [(start, end)]: where the music rises (the pauses you left for it).
    Writes and returns workdir/mix.wav, `duration` seconds long."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    fx = work / "effects.f32"
    effects.astype(np.float32).tofile(fx)

    args, chains, outs, n = [], [], [], 0
    stereo = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"
    if voice:
        gain = VOICE_LUFS - loudness(ffmpeg, voice)
        args += ["-i", voice]
        tail = "asplit=2[vo][sc]" if music else "anull[vo]"
        chains.append(f"[{n}:a]{stereo},volume={gain:.2f}dB,{tail}")
        outs.append("[vo]")
        n += 1
    if music:
        gain = VOICE_LUFS - music_cfg["below_voice_db"] - loudness(ffmpeg, music)
        fade = music_cfg["fade_out"]
        args += ["-stream_loop", "-1", "-i", music]
        chain = (f"[{n}:a]{stereo},volume={gain:.2f}dB,atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d=0.2,afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade}")
        if swells and music_cfg.get("swell_db"):
            # Up by swell_db inside each pause, ramping over 0.4 s at each end.
            lift = 10 ** (music_cfg["swell_db"] / 20) - 1
            ramps = "+".join(f"clip(min((t-{a:.3f})/0.4,({b:.3f}-t)/0.4),0,1)" for a, b in swells)
            chain += f",volume='1+{lift:.4f}*({ramps})':eval=frame"
        if voice:
            d = music_cfg["duck"]
            chain += (f"[bed];[bed][sc]sidechaincompress=threshold={d['threshold']}:"
                      f"ratio={d['ratio']}:attack={d['attack_ms']}:release={d['release_ms']}[mu]")
        else:
            chain += "[mu]"
        chains.append(chain)
        outs.append("[mu]")
        n += 1
    args += ["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", fx]
    chains.append(f"[{n}:a]anull[fx]")
    outs.append("[fx]")
    chains.append(f"{''.join(outs)}amix=inputs={len(outs)}:normalize=0:duration=longest,"
                  f"atrim=0:{duration:.3f},apad=whole_dur={duration:.3f}[mix]")
    premix = work / "premix.wav"
    _run([ffmpeg, "-y", "-v", "error", *args, "-filter_complex", ";".join(chains),
          "-map", "[mix]", "-c:a", "pcm_f32le", premix])

    # Gain to the target from the measured loudness, with a limiter to catch
    # the peaks (a voice's plosives), then one correction for what the
    # limiter took. loudnorm cannot do this: when a linear gain would clip it
    # switches to its dynamic mode, which measured 1.6 LU short.
    out = work / "mix.wav"
    level = loudness(ffmpeg, premix)
    if level <= -60:                        # silence: nothing to normalise
        shutil.copyfile(premix, out)
        return out
    gain = TARGET_LUFS - level
    for _ in range(4):
        _run([ffmpeg, "-y", "-v", "error", "-i", premix, "-af",
              f"volume={gain:.2f}dB,alimiter=limit={PEAK:.4f}:attack=5:release=50:level=false,"
              f"aresample={SR}", "-c:a", "pcm_s16le", out])
        miss = TARGET_LUFS - loudness(ffmpeg, out)
        if abs(miss) < 0.2:
            break
        gain += miss
    log(f"mix: voice {'yes' if voice else 'none'}, music {'ducked under the voice' if voice and music else 'yes' if music else 'none'}, "
        f"normalised to {TARGET_LUFS:g} LUFS")
    return out


def mux(ffmpeg, video, audio, t0, seconds, out):
    """Video + the matching stretch of the mix -> the finished MP4."""
    _run([ffmpeg, "-y", "-v", "error", "-i", video, "-ss", f"{t0:.3f}", "-t", f"{seconds:.3f}",
          "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-shortest", "-movflags", "+faststart", out])
