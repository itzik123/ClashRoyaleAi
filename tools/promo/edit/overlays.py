"""Graphics over the footage: a target ring and a pointer arrow.

Both are placed in the CLIP's coordinates, so they stay on their target
through zooms and shakes. They are drawn at 4x and scaled down for smooth
edges, with a thin white edge and a soft shadow so they read on any board.

  ring   locks onto its target: it shrinks from 1.4x its size as it fades in
  arrow  slides in along its own direction, then bobs gently toward the target
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter

from captions import rgba, with_alpha

SS = 4           # supersampling
IN = 0.28        # seconds to animate in
FADE = 0.2       # seconds to fade out
SHAPES = ("ring", "arrow")


def _ease_out(u):
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 3


def _finish(img, layer, x0, y0, alpha, k):
    """Downsample the 4x layer, put a soft shadow under it, paste at (x0, y0)."""
    layer = layer.resize((layer.width // SS, layer.height // SS), Image.LANCZOS)
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow.putalpha(layer.getchannel("A").point(lambda v: v * 0.55).filter(
        ImageFilter.GaussianBlur(max(1.0, 5 * k))))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(shadow, (0, max(1, round(3 * k))))
    out.alpha_composite(layer)
    out = with_alpha(out, alpha)
    img.paste(out, (round(x0), round(y0)), out)


def ring(img, at, size, colour, dt, dur, k):
    """A ring of diameter `size` around `at` (screen pixels)."""
    e = _ease_out(dt / IN)
    alpha = min(e, (dur - dt) / FADE)
    if alpha <= 0:
        return
    col = rgba(colour)
    r = size / 2 * (1.4 - 0.4 * e)
    w, edge = max(3.0, 8 * k), max(1.5, 2.5 * k)
    half = r + w / 2 + edge + 12 * k
    n = math.ceil(2 * half)
    layer = Image.new("RGBA", (n * SS, n * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = n * SS / 2

    def circle(radius, **kw):
        d.ellipse((c - radius * SS, c - radius * SS, c + radius * SS, c + radius * SS), **kw)

    circle(r, fill=col[:3] + (24,))                                          # faint tint
    circle(r + w / 2 + edge, outline=(255, 255, 255, 240), width=round((w + 2 * edge) * SS))
    circle(r + w / 2, outline=col, width=round(w * SS))
    _finish(img, layer, at[0] - n / 2, at[1] - n / 2, alpha, k)


def arrow(img, tip, tail, size, colour, dt, dur, k):
    """A pointer from `tail` to `tip` (screen pixels); `size` sets its weight."""
    e = _ease_out(dt / IN)
    alpha = min(e, (dur - dt) / FADE)
    if alpha <= 0:
        return
    (tx, ty), (px, py) = tail, tip
    length = math.hypot(px - tx, py - ty)
    if length < 1:
        return
    ux, uy = (px - tx) / length, (py - ty) / length
    # Slide in from behind, then bob toward the target.
    back = (1 - e) * 0.35 * length + (0 if e < 1 else 5 * k * (0.5 + 0.5 * math.sin(2 * math.pi * 1.4 * dt)))
    px, py, tx, ty = px - ux * back, py - uy * back, tx - ux * back, ty - uy * back
    nx, ny = -uy, ux
    edge = max(1.5, 3 * k)

    def shape(grow):
        shaft, head_w, head_l = 0.16 * size + 2 * grow, 0.5 * size + 2.6 * grow, 0.5 * size + 1.6 * grow
        hx, hy = px - ux * (head_l - grow), py - uy * (head_l - grow)     # the head's base
        tipx, tipy = px + ux * grow, py + uy * grow
        body = [(tx + nx * shaft / 2, ty + ny * shaft / 2), (hx + nx * shaft / 2, hy + ny * shaft / 2),
                (hx + nx * head_w / 2, hy + ny * head_w / 2), (tipx, tipy),
                (hx - nx * head_w / 2, hy - ny * head_w / 2), (hx - nx * shaft / 2, hy - ny * shaft / 2),
                (tx - nx * shaft / 2, ty - ny * shaft / 2)]
        return body, shaft / 2

    pad = 0.6 * size + 14 * k
    x0, y0 = min(px, tx) - pad, min(py, ty) - pad
    W, H = math.ceil(abs(px - tx) + 2 * pad), math.ceil(abs(py - ty) + 2 * pad)
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def P(pt):
        return ((pt[0] - x0) * SS, (pt[1] - y0) * SS)

    for grow, fill in ((edge, (255, 255, 255, 240)), (0.0, rgba(colour))):
        body, rad = shape(grow)
        d.polygon([P(p) for p in body], fill=fill)
        cx, cy = P((tx, ty))                                            # round tail
        d.ellipse((cx - rad * SS, cy - rad * SS, cx + rad * SS, cy + rad * SS), fill=fill)
    _finish(img, layer, x0, y0, alpha, k)
