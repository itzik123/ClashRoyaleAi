"""On-screen type: pop-in captions, call-outs, counters and the label pill.

Every size in short.json is in pixels on a 1080-wide frame; a --draft render
scales them with the frame. Images are rendered once per distinct text and
cached, so animating them costs one small resize per frame.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Used when the font in short.json is missing, so a draft still renders.
FALLBACK_FONTS = (r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arialbd.ttf")
TRAILING = ".,!?;:…"


def parse_line(text, max_words):
    """One script line -> (spoken words, caption chunks).

    `*word*` (or `*several words*`) is highlighted, and ` / ` forces a caption
    break. Without any ` / `, chunks hold up to `max_words` words and break
    after punctuation. A chunk is a list of (word index, word, highlighted).
    """
    words, chunks, cur, hl = [], [], [], False
    explicit = "/" in text.split()
    for tok in text.split():
        if tok == "/":
            if cur:
                chunks.append(cur)
            cur = []
            continue
        opens = tok.startswith("*")
        word = tok[1:] if opens else tok
        closes = "*" in word
        word = word.replace("*", "")
        if not word:
            continue
        cur.append((len(words), word, hl or opens))
        words.append(word)
        hl = (hl or opens) and not closes
    if cur:
        chunks.append(cur)
    if not explicit:
        chunks = [piece for c in chunks for piece in _split(c, max_words)]
    return words, chunks


def _split(chunk, max_words):
    out, cur = [], []
    for item in chunk:
        cur.append(item)
        if len(cur) >= max_words or item[1][-1] in TRAILING:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def rgba(spec, alpha=255):
    s = spec.lstrip("#")
    if len(s) not in (6, 8):
        raise SystemExit(f"colour {spec!r}: use #RRGGBB")
    vals = [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
    return tuple(vals) + ((alpha,) if len(vals) == 3 else ())


@lru_cache(maxsize=128)
def _load(path, px):
    return ImageFont.truetype(path, px)


def back_out(u):
    """0 -> 1 with a small overshoot: the "pop"."""
    c1 = 1.70158
    v = u - 1
    return 1 + (c1 + 1) * v ** 3 + c1 * v ** 2


def with_alpha(img, a):
    if a >= 1:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lambda v: round(v * max(0.0, a))))
    return out


def scaled(img, s):
    if abs(s - 1) < 1e-3:
        return img
    size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
    return img.resize(size, Image.BICUBIC)


def put(frame, img, cx, cy):
    """Paste an RGBA image onto the frame, centred on (cx, cy)."""
    frame.paste(img, (round(cx - img.width / 2), round(cy - img.height / 2)), img)


class Typesetter:
    def __init__(self, font_path, frame_width, style, warn):
        path = Path(font_path)
        if not path.exists():
            fallback = next((f for f in FALLBACK_FONTS if Path(f).exists()), None)
            if fallback is None:
                raise SystemExit(f"font not found: {path}")
            warn(f"font {path.name} not found in {path.parent}; using {Path(fallback).name} "
                 "until it is there")
            path = Path(fallback)
        self.font_path = str(path)
        self.k = frame_width / 1080.0
        self.st = style
        self.max_w = frame_width * style["caption_max_width"]
        self._cache = {}

    def _font(self, px):
        return _load(self.font_path, max(8, round(px)))

    @staticmethod
    def _width(rows, f):
        sp = f.getlength(" ")
        return max(sum(f.getlength(w) for w, _ in r) + sp * (len(r) - 1) for r in rows)

    def _fit(self, words, px, wrap):
        """Rows and a font size that fit the caption width: one row if it fits,
        else the most even two-row split, else shrink."""
        size = px
        while True:
            f = self._font(size)
            if self._width([words], f) <= self.max_w:
                return [words], size
            splits = [[words[:i], words[i:]] for i in range(1, len(words))] if wrap else []
            fits = [r for r in splits if self._width(r, f) <= self.max_w]
            if fits:
                return min(fits, key=lambda r: self._width(r, f)), size
            if size <= px * 0.5:
                return (min(splits, key=lambda r: self._width(r, f)) if splits else [words]), size
            size *= 0.92

    def _draw(self, rows, size):
        f = self._font(size)
        st = self.st
        stroke = round(st["stroke"] * size / st["caption_size"])  # scales with the text
        sp = f.getlength(" ")
        asc, desc = f.getmetrics()
        lh = round((asc + desc) * 1.0)
        widths = [sum(f.getlength(w) for w, _ in r) + sp * (len(r) - 1) for r in rows]
        drop = round(size * 0.05)
        blur = max(1, round(size * 0.05))
        pad = stroke + 2 * blur + drop
        W = math.ceil(max(widths)) + 2 * pad
        H = lh * len(rows) + 2 * pad
        edge = rgba(st["stroke_color"])

        def words(draw, dy, colour=None, stroke_fill=edge):
            for i, row in enumerate(rows):
                x = (W - widths[i]) / 2
                y = pad + i * lh + dy
                for w, c in row:
                    draw.text((x, y), w, font=f, fill=colour or c, anchor="la",
                              stroke_width=stroke, stroke_fill=stroke_fill)
                    x += f.getlength(w) + sp

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if st["shadow"] > 0:
            dark = (0, 0, 0, round(255 * st["shadow"]))
            shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            words(ImageDraw.Draw(shadow), drop, dark, dark)
            img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(blur)))
        words(ImageDraw.Draw(img), 0)
        return img

    def caption(self, chunk):
        """chunk: [(word, highlighted)] -> RGBA image."""
        key = ("cap", tuple(chunk))
        if key not in self._cache:
            st = self.st
            base, hl = rgba(st["caption_color"]), rgba(st["highlight_color"])
            words = [(w.upper() if st["uppercase"] else w, hl if h else base) for w, h in chunk]
            rows, size = self._fit(words, st["caption_size"] * self.k, wrap=True)
            self._cache[key] = self._draw(rows, size)
        return self._cache[key]

    def callout(self, text, colour, size):
        """One big line (a number, a URL), shrunk until it fits the width."""
        key = ("call", text, colour, size)
        if key not in self._cache:
            words = [(w.upper() if self.st["uppercase"] and not _is_url(text) else w, rgba(colour))
                     for w in text.split()]
            rows, px = self._fit(words, size * self.k, wrap=False)
            self._cache[key] = self._draw(rows, px)
        return self._cache[key]

    def label(self, text):
        """The small "Recreated in the engine" pill."""
        key = ("label", text)
        if key not in self._cache:
            k = self.k
            f = self._font(self.st["label_size"] * k)
            asc, desc = f.getmetrics()
            px, py = round(18 * k), round(9 * k)
            W, H = math.ceil(f.getlength(text)) + 2 * px, asc + desc + 2 * py
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 2, fill=(0, 0, 0, 150))
            d.text((px, py), text, font=f, fill=(245, 245, 245, 255), anchor="la")
            self._cache[key] = img
        return self._cache[key]

    def credit(self, lines, size):
        """Small centred lines on a dark box, like the label: the music credit
        and the fan-content notice. The type shrinks until the box fits 90% of
        the frame."""
        key = ("credit", tuple(lines), size)
        if key not in self._cache:
            k, px = self.k, size * self.k
            while True:
                f = self._font(px)
                pad_x, pad_y, gap = round(30 * k), round(18 * k), round(8 * k)
                W = math.ceil(max(f.getlength(ln) for ln in lines)) + 2 * pad_x
                if W <= 0.9 * 1080 * k or px <= 8:
                    break
                px *= 0.95
            asc, desc = f.getmetrics()
            H = len(lines) * (asc + desc) + (len(lines) - 1) * gap + 2 * pad_y
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, W - 1, H - 1), radius=round(26 * k), fill=(0, 0, 0, 170))
            for i, ln in enumerate(lines):
                d.text((W / 2, pad_y + i * (asc + desc + gap)), ln, font=f,
                       fill=(245, 245, 245, 255), anchor="ma")
            self._cache[key] = img
        return self._cache[key]

    def card(self, size, lines):
        """A full-frame placeholder, for a clip that has not been rendered yet."""
        W, H = size
        img = Image.new("RGB", size, (24, 26, 32))
        f = self._font(34 * self.k)
        y = H * 0.42
        for text in lines:
            ImageDraw.Draw(img).text((W / 2, y), text, font=f, fill=(230, 230, 230), anchor="mm")
            y += 50 * self.k
        return img


def _is_url(text):
    return "/" in text or ".com" in text
