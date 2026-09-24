"""A small, dependency-free SVG chart kit for the RESEARCH.md figures.

Standard library only, so the figures regenerate on any Python without
installing a plotting stack. Every figure is rendered twice, light and dark,
from the same code; RESEARCH.md embeds both with <picture> so GitHub serves the
one matching the reader's theme.

Colours are the validated reference palette (categorical slots 1-3 pass the
CVD and normal-vision gates in both modes; slot 3 needs visible labels on the
light surface, which every figure provides). Mark specs: bars <= 24 px with a
4 px rounded data end, 2 px lines, markers r >= 4 with a 2 px surface ring,
hairline solid gridlines, text in ink tokens and never in a series colour.
"""
from html import escape

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "seq": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "div": ("#2a78d6", "#f0efec", "#e34948"),
        "board_line": "#e1e0d9", "river": "#d6e6f7", "tower": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "seq": ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
        "div": ("#3987e5", "#383835", "#e66767"),
        "board_line": "#2c2c2a", "river": "#1d3a5c", "tower": "#898781",
    },
}


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a, b, t):
    ra, rb = _hex(a), _hex(b)
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def ramp(stops, t):
    """Piecewise-linear colour along `stops` for t in [0, 1]."""
    t = min(1.0, max(0.0, t))
    n = len(stops) - 1
    i = min(int(t * n), n - 1)
    return _mix(stops[i], stops[i + 1], t * n - i)


def diverging(theme, t):
    """t in [-1, 1]: blue (negative) - neutral - red (positive)."""
    lo, mid, hi = THEMES[theme]["div"]
    return _mix(mid, hi, t) if t >= 0 else _mix(mid, lo, -t)


def fmt(v, digits=2):
    if isinstance(v, int) or float(v).is_integer() and abs(v) >= 10:
        return f"{int(round(v)):,}"
    return f"{v:,.{digits}f}"


class Canvas:
    def __init__(self, width, height, theme, title, desc=""):
        self.w, self.h, self.theme = width, height, theme
        self.t = THEMES[theme]
        self.items = []
        self.title, self.desc = title, desc

    # -- primitives -----------------------------------------------------
    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill, rx=0, tip=None, opacity=None):
        op = f' fill-opacity="{opacity}"' if opacity is not None else ""
        body = f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" rx="{rx}" fill="{fill}"{op}'
        self.add(body + (f"><title>{escape(tip)}</title></rect>" if tip else "/>"))

    def line(self, x1, y1, x2, y2, stroke, width=1, cap="butt"):
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="{cap}"/>')

    def polyline(self, pts, stroke, width=2):
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polyline points="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')

    def path(self, d, fill, stroke="none", width=0):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def dot(self, x, y, fill, r=4.5, tip=None):
        ring = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r + 2:.2f}" fill="{self.t["surface"]}"/>'
        c = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"'
        self.add(ring + c + (f"><title>{escape(tip)}</title></circle>" if tip else "/>"))

    def text(self, x, y, s, size=12, color=None, anchor="start", weight="normal",
             baseline="alphabetic", rotate=None, italic=False):
        color = color or self.t["ink"]
        rot = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" dominant-baseline="{baseline}"{rot}{st}>'
                 f"{escape(str(s))}</text>")

    def bar_v(self, x, y_base, width, y_top, fill, tip=None):
        """Vertical bar grown from y_base to y_top, 4 px rounded data end."""
        h = y_base - y_top
        if h <= 0:
            return
        r = min(4.0, width / 2, h)
        d = (f"M{x:.2f},{y_base:.2f} L{x:.2f},{y_top + r:.2f} Q{x:.2f},{y_top:.2f} {x + r:.2f},{y_top:.2f} "
             f"L{x + width - r:.2f},{y_top:.2f} Q{x + width:.2f},{y_top:.2f} {x + width:.2f},{y_top + r:.2f} "
             f"L{x + width:.2f},{y_base:.2f} Z")
        self.add(f'<path d="{d}" fill="{fill}">' + (f"<title>{escape(tip)}</title>" if tip else "") + "</path>")

    def bar_h(self, x_base, y, height, x_end, fill, tip=None):
        """Horizontal bar grown from x_base to x_end (either direction)."""
        if abs(x_end - x_base) < 0.01:
            return
        sgn = 1 if x_end > x_base else -1
        w = abs(x_end - x_base)
        r = min(4.0, height / 2, w)
        xe = x_end
        xr = xe - sgn * r
        d = (f"M{x_base:.2f},{y:.2f} L{xr:.2f},{y:.2f} Q{xe:.2f},{y:.2f} {xe:.2f},{y + r:.2f} "
             f"L{xe:.2f},{y + height - r:.2f} Q{xe:.2f},{y + height:.2f} {xr:.2f},{y + height:.2f} "
             f"L{x_base:.2f},{y + height:.2f} Z")
        self.add(f'<path d="{d}" fill="{fill}">' + (f"<title>{escape(tip)}</title>" if tip else "") + "</path>")

    # -- chrome --------------------------------------------------------------
    def heading(self, s, x=24, y=30, sub=None):
        self.text(x, y, s, size=15, weight="600")
        if sub:
            self.text(x, y + 19, sub, size=12, color=self.t["ink2"])

    def legend(self, x, y, entries, gap=20):
        """entries: [(label, colour, kind)] with kind 'bar' | 'line' | 'dot'."""
        cx = x
        for label, colour, kind in entries:
            if kind == "line":
                self.line(cx, y - 4, cx + 16, y - 4, colour, 2, cap="round")
                cx += 22
            elif kind == "dot":
                self.dot(cx + 5, y - 4, colour, r=4)
                cx += 16
            else:
                self.rect(cx, y - 10, 12, 12, colour, rx=3)
                cx += 18
            self.text(cx, y, label, size=12, color=self.t["ink2"])
            cx += 7.2 * len(label) + gap

    def svg(self):
        t = self.t
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" font-family=\'{FONT}\'>'
                f"<title>{escape(self.title)}</title><desc>{escape(self.desc)}</desc>"
                f'<rect width="{self.w}" height="{self.h}" rx="8" fill="{t["surface"]}"/>')
        return head + "".join(self.items) + "</svg>\n"


class Axes:
    """A cartesian panel: data -> pixel transforms, gridlines, ticks, labels."""

    def __init__(self, cv, x, y, w, h, xlim, ylim):
        self.cv, self.x, self.y, self.w, self.h = cv, x, y, w, h
        self.xlim, self.ylim = xlim, ylim

    def sx(self, v):
        a, b = self.xlim
        return self.x + (v - a) / (b - a) * self.w

    def sy(self, v):
        a, b = self.ylim
        return self.y + self.h - (v - a) / (b - a) * self.h

    def grid_y(self, ticks, fmt_fn=None, label=None):
        t = self.cv.t
        for v in ticks:
            yy = self.sy(v)
            self.cv.line(self.x, yy, self.x + self.w, yy, t["grid"], 1)
            self.cv.text(self.x - 8, yy + 4, (fmt_fn or fmt)(v), size=11, color=t["muted"], anchor="end")
        if label:
            self.cv.text(self.x - 44, self.y + self.h / 2, label, size=12, color=t["ink2"],
                         anchor="middle", rotate=-90)

    def grid_x(self, ticks, fmt_fn=None, label=None):
        t = self.cv.t
        for v in ticks:
            xx = self.sx(v)
            self.cv.line(xx, self.y, xx, self.y + self.h, t["grid"], 1)
            self.cv.text(xx, self.y + self.h + 16, (fmt_fn or fmt)(v), size=11, color=t["muted"],
                         anchor="middle")
        if label:
            self.cv.text(self.x + self.w / 2, self.y + self.h + 36, label, size=12, color=t["ink2"],
                         anchor="middle")

    def baseline(self, v=0.0):
        self.cv.line(self.x, self.sy(v), self.x + self.w, self.sy(v), self.cv.t["axis"], 1)

    def vbaseline(self, v=0.0):
        self.cv.line(self.sx(v), self.y, self.sx(v), self.y + self.h, self.cv.t["axis"], 1)

    def xcats(self, labels, label=None, size=11):
        n = len(labels)
        step = self.w / n
        for i, s in enumerate(labels):
            self.cv.text(self.x + step * (i + 0.5), self.y + self.h + 17, s, size=size,
                         color=self.cv.t["ink2"], anchor="middle")
        if label:
            self.cv.text(self.x + self.w / 2, self.y + self.h + 38, label, size=12,
                         color=self.cv.t["ink2"], anchor="middle")
        return step

    def title(self, s):
        self.cv.text(self.x, self.y - 12, s, size=12.5, weight="600")
