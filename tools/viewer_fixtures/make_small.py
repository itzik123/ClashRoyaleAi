"""Trim the full fixtures down to browser-testable size.

The full teacher.json is ~3 MB, which is fine for a real browser but times out
in a throttled/non-painting automation pane. These keep the first N ticks and
the decision windows that cover them, so every code path is still represented
at a fraction of the parse cost.

Run after make_fixtures.py:
    python_ai/venv/Scripts/python.exe tools/viewer_fixtures/make_small.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "web", "fixtures")

TICKS = 600


def trim(src, dst, ticks=TICKS):
    with open(os.path.join(OUT, src)) as f:
        data = json.load(f)
    data["ticks"] = data["ticks"][:ticks]
    data["totalTicks"] = len(data["ticks"])
    td = data.get("teacherDebug")
    if td and isinstance(td.get("decisions"), list):
        skip = td.get("skipFrames") or 10
        keep = (ticks + skip - 1) // skip
        td["decisions"] = td["decisions"][:keep]
    with open(os.path.join(OUT, dst), "w") as f:
        json.dump(data, f)
    return os.path.getsize(os.path.join(OUT, dst))


def main():
    for src, dst in (("modern.json", "modern_small.json"),
                     ("legacy.json", "legacy_small.json"),
                     ("teacher.json", "teacher_small.json"),
                     ("mixed.json", "mixed_small.json")):
        n = trim(src, dst)
        print(f"{dst}: {n / 1024:.0f} KB")


if __name__ == "__main__":
    main()
