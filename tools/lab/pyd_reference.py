"""Reference rollouts through the Python engine (clash_royale_env .pyd), for
lab_cli's probe to reproduce exactly.

Same semantics as web/lab/engine/lab_engine.h -- attacker for team 1 and
defender for team 0 both injected with the real deploy time, one tick per
step_self_play_fast, destroyed towers counted as 0 hp -- but written
independently and with NO early termination, so a match also checks that
LabEngine's "stop once no enemy unit remains" never cuts damage short.

Run:  python_ai/venv/Scripts/python.exe tools/lab/pyd_reference.py
Prints one JSON line per case: [attacker, sx, sy, defender, cx, cy, dropTick, damage]
(cx = -1 for no defence).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import python_ai  # noqa: E402,F401  -- puts the .pyd on sys.path
import clash_royale_env as E  # noqa: E402

DECK = [15, 6, 25, 40, 24, 72, 33, 7]   # lab_engine.h baseDeck(); any legal deck does
WINDOW = 300

CASES = [
    # Hog Rider vs Cannon
    (15, 14, 20, 25, -1, -1, 0),
    (15, 14, 20, 25, 11, 9, 0),
    (15, 14, 20, 25, 11, 9, 20),
    (15, 14, 20, 25, 11, 9, 33),
    (15, 14, 20, 25, 14, 9, 0),
    (15, 14, 20, 25, 14, 9, 10),
    (15, 3, 21, 25, 6, 10, 40),
    # other pairings, so the probe is not one card deep
    (2, 3, 19, 25, 5, 10, 15),       # Giant vs Cannon
    (45, 9, 22, 26, 8, 8, 5),        # Balloon vs Tesla
    (12, 14, 18, 10, 13, 11, 25),    # Skeleton Army vs Valkyrie
    (81, 4, 21, 96, 6, 9, 0),        # Battle Ram vs Tombstone (Barbarians on death)
    (14, 12, 19, 0, 12, 12, 30),     # Prince vs Knight
]


def tower_hp(e):
    return sum(max(0, e.get_tower_hp(0, s)) for s in range(3))


def main():
    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.seed(1)
    for att, sx, sy, dfn, cx, cy, drop in CASES:
        e = env.snapshot()
        e.inject(att, float(sx), float(sy), 1)
        start = tower_hp(e)
        for t in range(WINDOW):
            if cx >= 0 and t == drop:
                e.inject(dfn, float(cx), float(cy), 0)
            e.step_self_play_fast(4, 0, 0, 4, 0, 0, 1)
        print(json.dumps([att, sx, sy, dfn, cx, cy, drop, start - tower_hp(e)]))


if __name__ == "__main__":
    main()
