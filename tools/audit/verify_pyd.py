"""Confirm the compiled .pyd carries the engine Python will train against.

The C++ suite tests the source, not python_ai/clash_royale_env.pyd, which is a
separate artifact. A stale .pyd behaves like an older engine and is invisible
in every Python metric. Run it after any rebuild, before a training run:

    python_ai/venv/Scripts/python.exe tools/audit/verify_pyd.py

The probes use step_self_play, not step: step also runs the C++
HeuristicOpponent, which defends, so a Giant that never reaches a tower would
prove nothing about navigation.
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
# A tripwire for an accidental resize, which kills every checkpoint; editing
# this literal acknowledges the change was intended.
if size != 13977:
    failures.append(f"observation_size is {size}, expected 13977")

# The elixir phase must sit at the index the layout says and be symmetric
# across teams: the tower block two slots earlier is mirrored, and copying that
# here would be silent.
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

# --- sight/attack geometry: a Musketeer ~8 tiles from a Princess Tower must
# not siege it for free ---
env = cre.ClashRoyaleEnv(DECK, DECK)
env.reset()
env.inject(MUSKETEER, 4.0, 19.0, 0)      # ~8 tiles from the left Princess Tower at (3, 27)
for _ in range(400):
    env.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
dealt = env.get_tower_damage_dealt(0)
print(f"tower damage by a lone Musketeer from 8 tiles: {dealt}")
# If sight and attack range disagree she outranges the tower and takes no
# damage in return (~5355 dealt); with them consistent the tower sees her and
# kills her.
if not (0 < dealt < 3000):
    failures.append(f"free-siege regime: Musketeer dealt {dealt} (pre-fix ~5355)")

# --- bridge navigation: a lone slow tank must complete a crossing ---
crossed = trials = 0
for x in (3.0, 3.5, 4.0, 4.5, 5.0, 13.0, 13.5, 14.0, 14.5, 15.0):
    e = cre.ClashRoyaleEnv(DECK, DECK)
    e.reset()
    e.inject(GIANT, x, 10.0, 0)
    trials += 1
    for _ in range(900):
        e.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
    # A Giant that never crosses can never reach a tower, so tower damage
    # proves it got there.
    if e.get_tower_damage_dealt(0) > 0:
        crossed += 1
print(f"lone Giants reaching an enemy tower: {crossed}/{trials}")
# The bridge-exit absorbing state deadlocked ~20% of lone ground crossings,
# worst for slow units.
if crossed != trials:
    failures.append(f"bridge exit trap: only {crossed}/{trials} Giants crossed")

# --- stale-artifact checks ---
# A .pyd built against an older engine reports the old arena columns or lacks
# the ARENA_* bindings. opponents/teacher.py refuses to import without
# step_self_play_fast, so catching it here names the cause before a training
# run does. step_self_play_fast is a method on the class, not the module.
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
    # Symmetry is the property that matters: one lane must not play differently
    # from the other.
    mirror = (cre.ARENA_WIDTH - 1) - cre.ARENA_LEFT_LANE_X
    if abs(mirror - cre.ARENA_RIGHT_LANE_X) > 1e-6:
        failures.append(f"arena asymmetry: lanes {cre.ARENA_LEFT_LANE_X}/{cre.ARENA_RIGHT_LANE_X}")
    mirror_b = (cre.ARENA_WIDTH - 1) - cre.ARENA_LEFT_BRIDGE_X
    if abs(mirror_b - cre.ARENA_RIGHT_BRIDGE_X) > 1e-6:
        failures.append(f"arena asymmetry: bridges {cre.ARENA_LEFT_BRIDGE_X}/{cre.ARENA_RIGHT_BRIDGE_X}")

# The King Tower starts dormant. A .pyd with a King firing from tick 0 differs
# by ~950 tower damage against a lone Hog.
_env = cre.ClashRoyaleEnv(DECK, DECK)
_env.reset()
_env.inject(HOG, 2.5, 17.5, 1)
for _ in range(600):
    _env.step_self_play(NOOP, 0, 0, NOOP, 0, 0, 1)
_hog_dmg = _env.get_tower_damage_dealt(1)
print(f"lone Hog tower damage against dormant Kings: {_hog_dmg}")
# ~2219 with the King asleep, ~1268 awake; the bar sits between them.
if _hog_dmg < 1700:
    failures.append(f"stale .pyd: lone Hog dealt {_hog_dmg}, expected ~2219 with a "
                    f"dormant King (~1268 means the King is still firing from tick 0)")

# The observation's river mask (channel 8, the net's only map of where it can
# cross) must agree with the physics. Checked here as well as in C++ because
# this is the artifact training loads.
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
