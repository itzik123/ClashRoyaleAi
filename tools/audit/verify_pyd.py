"""Confirm the compiled .pyd carries the engine fixes Python will train against.

WHY THIS EXISTS. The C++ suite proves the SOURCE is correct. It says nothing
about `python_ai/clash_royale_env.pyd`, which is a separate build artifact and
has silently gone stale more than once in this project -- most recently on
2026-08-19, when it predated commit 26de409 and quietly blocked two
measurements. A stale .pyd fails by behaving like an older engine, which is
invisible in every Python metric.

Run it after any engine rebuild, before starting a training run:

    python_ai/venv/Scripts/python.exe tools/audit/verify_pyd.py

BOTH PROBES USE step_self_play, NOT step. `ClashEnv::step` also runs the C++
HeuristicOpponent, which defends -- so a Giant that never reaches a tower proves
nothing about navigation, and the first version of this script "failed" at 2/10
for exactly that reason while the .pyd was perfectly current. `step_self_play`
deliberately never calls opponentTurn(), which is the isolation these checks
need. Same attribution trap CLAUDE.md records for get_troop_damage_dealt.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import python_ai  # noqa: F401  -- appends PACKAGE_DIR so the .pyd is importable
import clash_royale_env as cre

DECK = [15, 6, 25, 40, 24, 72, 33, 7]
MUSKETEER, GIANT = 6, 2
NOOP = 4

failures = []

size = cre.ClashRoyaleEnv(DECK, DECK).observation_size()
print(f"observation_size = {size}")
if size != 13606:
    failures.append(f"observation_size is {size}, expected 13606")

# ---- sight/attack geometry: a Musketeer 8 tiles from a Princess Tower must
# ---- NOT be able to siege it for free.
env = cre.ClashRoyaleEnv(DECK, DECK)
env.reset()
env.inject(MUSKETEER, 4.0, 19.0, 0)      # 8.0 tiles south of the (4, 27) tower
for _ in range(400):
    env.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
dealt = env.get_tower_damage_dealt(0)
print(f"tower damage by a lone Musketeer from 8 tiles: {dealt}")
# Pre-fix she dealt ~5355 to a 3204 hp tower and took ZERO damage in return,
# because sightRange was compared centre-to-centre while attacks used
# attackRange + both radii. Post-fix the tower sees her and kills her.
if not (0 < dealt < 3000):
    failures.append(f"free-siege regime: Musketeer dealt {dealt} (pre-fix ~5355)")

# ---- bridge navigation: a lone slow tank must complete a crossing.
crossed = trials = 0
for x in (3.0, 3.5, 4.0, 4.5, 5.0, 13.0, 13.5, 14.0, 14.5, 15.0):
    e = cre.ClashRoyaleEnv(DECK, DECK)
    e.reset()
    e.inject(GIANT, x, 10.0, 0)
    trials += 1
    for _ in range(900):
        e.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
    # A Giant that never crosses can never reach a tower, so tower damage is the
    # end-to-end proof that it got there -- no entity introspection needed.
    if e.get_tower_damage_dealt(0) > 0:
        crossed += 1
print(f"lone Giants reaching an enemy tower: {crossed}/{trials}")
# Pre-fix ~20% of lone ground crossings deadlocked a few thousandths of a tile
# short of the far bank, worst for slow units -- the Giant was 27/34.
if crossed != trials:
    failures.append(f"bridge exit trap: only {crossed}/{trials} Giants crossed")

if failures:
    print("\nFAILED -- the .pyd does not match the current engine source:")
    for f in failures:
        print(f"  - {f}")
    print("\nRebuild it (see CLAUDE.md) and check no Python process holds it open.")
    sys.exit(1)

print("\nOK: the deployed .pyd carries both 2026-08-20 engine fixes.")
