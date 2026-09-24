"""When each script word is spoken: faster-whisper, matched to the script.

Whisper transcribes the voiceover with a start and end time for every word it
hears. The script is then lined up against that transcript (difflib, on
normalised tokens: lower case, no punctuation, numbers spelled out, so "64",
"sixty-four" and "Sixty four" all match). A script word Whisper heard
differently, or not at all, is timed between its matched neighbours.

The transcript is cached per recording, so editing the script re-runs only the
matching, which is instant. A new take (a different file) is transcribed again.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _int_words(n):
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [ONES[n // 100], "hundred"] + (_int_words(n % 100) if n % 100 else [])
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            if n // size >= 1000:
                return [str(n)]
            return _int_words(n // size) + [name] + (_int_words(n % size) if n % size else [])
    return [str(n)]


def _number_words(tok):
    whole, _, frac = tok.replace(",", "").partition(".")
    out = _int_words(int(whole)) if whole else []
    return out + (["point"] + [ONES[int(d)] for d in frac] if frac else [])


def normalize(word):
    """A word as comparable tokens: "20,000x" -> [twenty, thousand, x]."""
    w = word.lower().replace("×", " times ").replace("%", " percent ").replace("&", " and ")
    out = []
    for tok in re.findall(r"[a-z]+|\d[\d,]*(?:\.\d+)?", w.replace("-", " ")):
        out += _number_words(tok) if tok[0].isdigit() else [tok]
    return out


# --- the transcript -------------------------------------------------------------

def transcribe(audio, model="base.en", language="en", prompt=None, cache_dir=None, log=print):
    """[{"w": word, "s": start, "e": end}] for every word Whisper hears."""
    audio = Path(audio)
    digest = hashlib.sha1(audio.read_bytes()).hexdigest()
    key = hashlib.sha1(f"{digest}|{model}|{language}|{prompt}".encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"transcript_{key}.json" if cache_dir else None
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["words"]

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")     # no "unauthenticated" nag
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n\nTiming captions to the voiceover needs faster-whisper, which lives "
            "in the editor's own venv. Set it up once (tools/promo/README.md, "
            "'edit_short.py'), then run the editor with that venv's Python:\n\n"
            "  tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/edit_short.py\n")
    log(f"transcribing {audio.name} with Whisper '{model}' "
        "(the first run downloads the model, about 150 MB)...")
    whisper = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(str(audio), language=language, initial_prompt=prompt,
                                     word_timestamps=True, beam_size=5)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
             for seg in segments for w in (seg.words or [])]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"audio": audio.name, "model": model, "words": words},
                                    indent=1), encoding="utf-8")
    return words


# --- script <-> transcript --------------------------------------------------------

def match(lines, heard):
    """Time every script word from the transcript.

    lines: [[word, ...], ...], the spoken words of each script line.
    Returns ([[(start, end), ...] per line], [[heard?, ...] per line]).
    """
    flat = [w for ws in lines for w in ws]
    S = [(tok, k) for k, w in enumerate(flat) for tok in normalize(w)]
    T = [(tok, j) for j, h in enumerate(heard) for tok in normalize(h["w"])]
    sm = difflib.SequenceMatcher(None, [a for a, _ in S], [b for b, _ in T], autojunk=False)
    times = [None] * len(flat)
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            k, j = S[a + i][1], T[b + i][1]
            s, e = heard[j]["s"], heard[j]["e"]
            times[k] = [s, e] if times[k] is None else [min(times[k][0], s), max(times[k][1], e)]
    found = [t is not None for t in times]
    _fill(times, flat)
    return _per_line(times, lines), _per_line(found, lines)


def estimate(lines, pauses=None, start=0.4, gap=0.3):
    """Times at a brisk speaking pace (~170 words a minute), for a preview
    before the voiceover exists. pauses[i]: the least silence before line i."""
    out, t = [], start - gap
    for i, ws in enumerate(lines):
        t += max(gap, pauses[i] if pauses else 0.0)
        row = []
        for w in ws:
            d = _say(w)
            row.append((t, t + d))
            t += d + (0.25 if w[-1] in ".!?:…" else 0.12 if w[-1] in ",;" else 0.0)
        out.append(row)
    return out


def _say(word):
    """Rough seconds to say a word."""
    return 0.1 + 0.046 * sum(len(tok) for tok in normalize(word) or [word])


def _fill(times, words):
    """Time the unheard words between their heard neighbours, in proportion
    to how long each takes to say."""
    n, k = len(times), 0
    while k < n:
        if times[k] is not None:
            k += 1
            continue
        j = k
        while j < n and times[j] is None:
            j += 1
        need = [_say(w) for w in words[k:j]]
        lo = times[k - 1][1] if k > 0 else None
        hi = times[j][0] if j < n else None
        if lo is None and hi is None:
            lo = 0.4
        if lo is None:
            lo = max(0.0, hi - sum(need))
        if hi is None:
            hi = lo + sum(need)
        span, t = max(0.0, hi - lo), lo
        for i, d in enumerate(need):
            dt = span * d / sum(need)
            times[k + i] = [t, t + dt]
            t += dt
        k = j


def _per_line(flat, lines):
    out, k = [], 0
    for ws in lines:
        out.append([tuple(v) if isinstance(v, list) else v for v in flat[k:k + len(ws)]])
        k += len(ws)
    return out
