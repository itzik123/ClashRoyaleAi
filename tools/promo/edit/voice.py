"""Make the voiceover with a text-to-speech voice, and fix how it ends sentences.

    tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/voice.py
    tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/voice.py --voice en-US-AndrewMultilingualNeural
    tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/voice.py --endings-only

Reads the lines from short.json, says them with a Microsoft neural voice
(edge-tts), and writes assets/voiceover.mp3 for edit_short.py.

Some voices barely mark the end of a sentence. Measured on Christopher, the
last word is not drawn out at all, and "Can it beat the game?" ends falling,
like a statement. So the last word of every sentence is reshaped with Praat
(PSOLA resynthesis, which changes pitch and timing, not the voice):

    "."  and "!"   glides down to end --fall semitones below its sentence's
                   median pitch, and its end is drawn out (--stretch)
    "?"            glides up to end --rise semitones above it
    "..."          is held: drawn out (--hold), its pitch left alone

Which words end a sentence comes from the script's punctuation; when they are
said comes from Whisper, as for everything else in the edit. The take as the
voice gave it is kept in cache/voiceover_tts.mp3, and --endings-only reshapes
it again with new settings without asking the voice for a new take.
--no-endings writes the voice as it comes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common import find_ffmpeg  # noqa: E402

import align  # noqa: E402
from captions import parse_line  # noqa: E402

PITCH_FLOOR, PITCH_CEILING = 75, 300     # Hz: covers a male or female speaking voice


def script(cfg):
    """[[word, ...] per line] and the text to speak, one line per line."""
    words = [parse_line(line["text"], 99)[0] for line in cfg["lines"]]
    return words, "\n".join(" ".join(ws) for ws in words)


def speak(text, voice, rate, out):
    import edge_tts
    asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(str(out)))


def endings(words):
    """(line, first word of the sentence, its last word, kind) for every
    sentence end. A "..." pause ends a stretch too, so the words after it
    start a new one."""
    out = []
    for li, ws in enumerate(words):
        first = 0
        for wi, w in enumerate(ws):
            kind = ("hold" if w.endswith("...") else "rise" if w.endswith("?")
                    else "fall" if w.endswith((".", "!")) else None)
            if kind:
                out.append((li, first, wi, kind))
                first = wi + 1
    return out


def reshape(wav_in, wav_out, spans, fall, rise, stretch, hold):
    """Praat: land the last word of each sentence on a target pitch, and draw
    out its end. spans are (sentence start, last word start, last word end,
    kind); the target is set against the sentence's own median pitch, so a
    word the voice pushed up still ends low (a statement) or high (a
    question). The bend starts from the word's own pitch, so it joins
    smoothly."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav_in))
    manip = call(snd, "To Manipulation", 0.01, PITCH_FLOOR, PITCH_CEILING)
    pitch = call(manip, "Extract pitch tier")
    duration = call(manip, "Extract duration tier")
    n = call(pitch, "Get number of points")
    points = [(call(pitch, "Get time from index", i), call(pitch, "Get value at index", i))
              for i in range(1, n + 1)]

    def median(a, b):
        v = sorted(f for t, f in points if a <= t < b)
        return v[len(v) // 2] if v else None

    for bs, s, e, kind in spans:
        if e - s < 0.05:
            continue
        if kind != "hold":
            word = [(t, f) for t, f in points if s <= t <= e]
            ref = median(bs, s) or median(s, s + 0.4 * (e - s))
            if word and ref:
                target = ref * 2 ** ((fall if kind == "fall" else rise) / 12)
                # One way only: a statement may end lower than the voice made
                # it, never higher, and a question higher, never lower. A fall
                # the voice already made steeper is left alone.
                keep = min if kind == "fall" else max
                call(pitch, "Remove points between", s, e)
                for t, f in word:
                    w = ((t - s) / (e - s)) ** 1.2
                    call(pitch, "Add point", t, keep(f, f ** (1 - w) * target ** w))
        # Draw out the back of the word, where the final syllable is.
        k = hold if kind == "hold" else stretch
        m = s + 0.35 * (e - s)
        for t, v in ((m - 0.005, 1.0), (m, k), (e, k), (e + 0.005, 1.0)):
            call(duration, "Add point", t, v)
    call([manip, pitch], "Replace pitch tier")
    call([manip, duration], "Replace duration tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    # Resynthesis can overshoot full scale, which a 16-bit WAV would clip.
    # The editor levels the voice itself, so only the peak matters here.
    call(out, "Scale peak", 0.95)
    out.save(str(wav_out), "WAV")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "short.json"))
    ap.add_argument("--voice", default="en-US-ChristopherNeural")
    ap.add_argument("--rate", default="+8%", help="speaking rate, e.g. +8%% or -5%%")
    ap.add_argument("--fall", type=float, default=-4.0,
                    help="where a statement ends: semitones against its sentence's median pitch")
    ap.add_argument("--rise", type=float, default=5.0,
                    help="where a question ends: semitones against its sentence's median pitch")
    ap.add_argument("--stretch", type=float, default=1.35,
                    help="how much the end of a sentence's last word is drawn out")
    ap.add_argument("--hold", type=float, default=1.35, help="the same, for a '...' word")
    ap.add_argument("--endings-only", action="store_true",
                    help="reshape the last take again (with new settings) instead of making a new one")
    ap.add_argument("--no-endings", action="store_true", help="write the voice as it comes")
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = (Path(args.config).parent / cfg["voiceover"]).resolve()
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    words, text = script(cfg)

    # The take as the voice gave it. The reshaping always starts from it, so
    # re-running with new settings never reshapes an already-reshaped file.
    raw = cache / "voiceover_tts.mp3"
    if args.endings_only:
        if not raw.exists():
            raise SystemExit(f"no take at {raw} yet: run once without --endings-only")
        print(f"reshaping the last take, {raw.name}")
    else:
        print(f"speaking {len(text)} characters as {args.voice} at {args.rate}...")
        speak(text, args.voice, args.rate, raw)

    if args.no_endings:
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(raw), "-ar", "44100",
                        "-b:a", "192k", str(out)], check=True)
        print(f"wrote {out}")
        return

    wav = cache / "voiceover_tts.wav"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", "24000",
                    str(wav)], check=True)
    wcfg = cfg.get("whisper", {})
    heard = align.transcribe(wav, wcfg.get("model", "base.en"), wcfg.get("language", "en"),
                             wcfg.get("prompt"), cache)
    times, found = align.match(words, heard)
    spans = []
    for li, first, wi, kind in endings(words):
        s, e = times[li][wi]
        if found[li][wi]:
            spans.append((times[li][first][0], s, e, kind))
        else:
            print(f"  ! not heard, left as is: {words[li][wi]!r}")
    counts = {k: sum(1 for *_, kk in spans if kk == k) for k in ("fall", "rise", "hold")}
    print(f"reshaping {len(spans)} sentence ends: {counts['fall']} fall, {counts['rise']} rise, "
          f"{counts['hold']} held")
    shaped = cache / "voiceover_shaped.wav"
    reshape(wav, shaped, spans, args.fall, args.rise, args.stretch, args.hold)
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(shaped), "-ar", "44100",
                    "-b:a", "192k", str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
