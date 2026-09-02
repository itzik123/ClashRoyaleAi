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
MUSKETEER, GIANT, HOG = 6, 2, 15
NOOP = 4

failures = []

size = cre.ClashRoyaleEnv(DECK, DECK).observation_size()
print(f"observation_size = {size}")
# 13977 since 2026-09-02: the elixir-phase multiplier scalar (item 26) took
# NUM_EXTRA_SCALARS 9 -> 10. Before that it was 13976 from the cycle blocks
# (item 24). Updated deliberately -- this literal is a tripwire for an
# ACCIDENTAL resize, since any change here kills every existing checkpoint, so
# editing it is the acknowledgement that the change was intended.
if size != 13977:
    failures.append(f"observation_size is {size}, expected 13977")

# The elixir phase must actually be in the vector, at the index the layout
# says, and must be SYMMETRIC across teams -- the tower block two slots earlier
# is mirrored, and a copy-paste of that mirroring here would be silent.
_pe = cre.ClashRoyaleEnv(DECK, DECK)
_pe.reset()
_phase_idx = cre.ClashRoyaleEnv.EXTRA_SCALARS_START + 9
for _tick, _want in ((0, 1.0), (1200, 2.0), (1800, 3.0)):
    _pe.set_current_tick(_tick)
    if abs(_pe.get_elixir_multiplier() - _want) > 1e-6:
        failures.append(f"elixir multiplier at tick {_tick} is "
                        f"{_pe.get_elixir_multiplier()}, expected {_want}")
    for _team in (0, 1):
        _got = _pe.get_observation_for_team(_team)[_phase_idx] * cre.MAX_ELIXIR_MULTIPLIER
        if abs(_got - _want) > 1e-4:
            failures.append(f"obs phase scalar (team {_team}) at tick {_tick} "
                            f"is {_got}, expected {_want}")

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

# ---------------------------------------------------------------------------
# The 2026-08-21 arena correction, and the bindings that ended three stale
# copies of it.
#
# This is the cheapest possible staleness check: a .pyd built before that date
# reports the old columns (King 9.0, left Princess 4.0, bridges 4.0/14.0) or
# lacks the ARENA_* attributes entirely. The C++ suite cannot tell you any of
# this -- it never loads the .pyd -- and the copy has silently failed twice.
# step_self_play_fast (UPSTREAM_REQUESTS item 21, 2026-08-23) belongs in the
# same check for the same reason: `opponents/teacher.py` refuses to import
# without it, so a .pyd that predates it turns every phase-1 run into an
# ImportError at startup. Catching it HERE, in the post-build gate, names the
# cause before a training run does.
#
# `cre` is the MODULE; step_self_play_fast is a method on the ClashRoyaleEnv
# CLASS. Checking the module was a false negative that could never pass, so
# from the day this check landed the gate reported FAILED on a perfectly good
# .pyd -- the worst failure mode available to a gate, since it teaches the
# reader to ignore it. Corrected 2026-08-24; the arena check below already
# reads module-level constants and is right to use `cre`.
if not hasattr(cre.ClashRoyaleEnv, "step_self_play_fast"):
    failures.append("stale .pyd: missing step_self_play_fast -- rebuild, and "
                    "check the post-build copy into python_ai/ actually landed "
                    "(MSB3073 if any Python process has the .pyd loaded)")

missing = [n for n in ("ARENA_CENTER_X", "ARENA_LEFT_LANE_X", "ARENA_RIGHT_LANE_X",
                       "ARENA_LEFT_BRIDGE_X", "ARENA_RIGHT_BRIDGE_X", "ARENA_BRIDGE_Y")
           if not hasattr(cre, n)]
if missing:
    failures.append("stale .pyd: missing arena bindings " + ", ".join(missing))
else:
    expected = {
        "ARENA_CENTER_X": 8.5,
        "ARENA_LEFT_LANE_X": 3.0,
        "ARENA_RIGHT_LANE_X": 14.0,
        "ARENA_LEFT_BRIDGE_X": 2.5,
        "ARENA_RIGHT_BRIDGE_X": 14.5,
    }
    got = {n: getattr(cre, n) for n in expected}
    print("arena geometry: " + "  ".join(f"{k.replace('ARENA_', '')}={v}" for k, v in got.items()))
    for name, want in expected.items():
        if abs(got[name] - want) > 1e-6:
            failures.append(f"arena drift: {name} is {got[name]}, expected {want}")
    # Symmetry is the property that actually matters -- one lane playing
    # differently from the other is the failure this geometry exists to prevent.
    mirror = (cre.ARENA_WIDTH - 1) - cre.ARENA_LEFT_LANE_X
    if abs(mirror - cre.ARENA_RIGHT_LANE_X) > 1e-6:
        failures.append(f"arena asymmetry: lanes {cre.ARENA_LEFT_LANE_X}/{cre.ARENA_RIGHT_LANE_X}")
    mirror_b = (cre.ARENA_WIDTH - 1) - cre.ARENA_LEFT_BRIDGE_X
    if abs(mirror_b - cre.ARENA_RIGHT_BRIDGE_X) > 1e-6:
        failures.append(f"arena asymmetry: bridges {cre.ARENA_LEFT_BRIDGE_X}/{cre.ARENA_RIGHT_BRIDGE_X}")

# The King Tower starts DORMANT (2026-08-21). A .pyd predating that has a King
# firing from tick 0, which is worth ~950 tower damage against a lone Hog -- a
# large gameplay difference that nothing else here would notice.
_env = cre.ClashRoyaleEnv(DECK, DECK)
_env.reset()
_env.inject(HOG, 2.5, 17.5, 1)
for _ in range(600):
    _env.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
_hog_dmg = _env.get_tower_damage_dealt(1)
print(f"lone Hog tower damage against dormant Kings: {_hog_dmg}")
# Measured 2219 with the King asleep, 1268 with it awake. The midpoint is a
# generous bar that still separates the two engines unambiguously.
if _hog_dmg < 1700:
    failures.append(f"stale .pyd: lone Hog dealt {_hog_dmg}, expected ~2219 with a "
                    f"dormant King (~1268 means the King is still firing from tick 0)")

# The observation's river mask must agree with the PHYSICS.
#
# Channel 8 is the only thing telling the network where it can cross, and it was
# painted from a hardcoded `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)` while
# movement used Board's bridges. When the arena was corrected only the physics
# followed, so the net was told columns 2 and 15 (real bridge) were water and
# columns 4 and 13 (real water) were bridge -- half the crossing map wrong in
# both directions, on every tick of every episode, with the whole C++ suite
# green. Checked here as well as in C++ because this is the artifact TRAINING
# loads, and a stale .pyd is exactly how a fixed engine still trains wrong.
_W = cre.ClashRoyaleEnv.BOARD_WIDTH
_plane = _W * cre.ClashRoyaleEnv.BOARD_HEIGHT
_menv = cre.ClashRoyaleEnv(DECK, DECK)
_menv.reset()
for _team in (0, 1):
    _obs = _menv.get_observation_for_team(_team)
    _row = "".join("B" if _obs[8 * _plane + 17 * _W + x] > 0 else "W" for x in range(_W))
    print(f"team {_team} observation river row: {_row}")
    if _row != "WWBBWWWWWWWWWWBBWW":
        failures.append(f"team {_team} river mask is {_row}, expected WWBBWWWWWWWWWWBBWW "
                        f"(the old encoder painted WWWBBWWWWWWWWBBWWW)")

if failures:
    print("\nFAILED -- the .pyd does not match the current engine source:")
    for f in failures:
        print(f"  - {f}")
    print("\nRebuild it (see CLAUDE.md) and check no Python process holds it open.")
    sys.exit(1)

print("\nOK: the deployed .pyd carries the 2026-08-20 and 2026-08-21 engine fixes.")
