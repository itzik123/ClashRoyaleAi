# ClashRoyaleEnv

A Clash Royale battle simulator in C++ (MSVC / CMake / pybind11) with a PPO
agent in Python, plus `perception/`, which reads real matches off the screen and
drives the simulator from them as a state estimator.

## Where to look

| | |
|---|---|
| **`CLAUDE.md`** | The operational reference. Rules, engine facts, the training mechanism, measured baselines, open problems, layout. Read this before changing anything. |
| **`TODO.md`** | The single list of pending work, verified against the source tree. Item 0 is what to do next. |
| **`DECISIONS.md`** | The narrative half of the knowledge base: how the learning mechanism got here, every measured result and every reversal. Read it before re-proposing anything. |
| `perception/README.md` | Per-stage status of the live sensor, with measured numbers. |
| `perception/UPSTREAM_REQUESTS.md` | Engine changes requested from `perception/`, with evidence and blast radius. |
| `perception/BOT_REQUESTS.md` | Training-side suggestions from `perception/`. |
| `docs/superpowers/specs/` | Design rationale for the utility teacher and the live sensor. |

## Layout

```
include/, src/       C++ engine. READ-ONLY by default.
tests/               C++ Catch2 tests (ClashRoyaleTests).
python_ai/           PPO agent, training pipelines, measurement harnesses.
                     READ-ONLY by default -- training runs here.
                     A package since 2026-08-20; one subpackage per
                     responsibility (models, envs, rl, trainers, eval, ...).
                     See CLAUDE.md's Layout section.
perception/          Screen -> placement events -> simulator as estimator.
                     Self-contained: own venv, own requirements.txt.
web/viewer.html      Replay viewer.
```

## Running things

The compiled extension `clash_royale_env.pyd` is built for **Python 3.11 only**.
The default `python` on the development machine is 3.14 and fails with
`ImportError: DLL load failed`, which reads like a corrupt build and is only a
version mismatch. Use one of:

```bash
python_ai/venv/Scripts/python.exe
```
```bash
perception/.venv/Scripts/python.exe
```

Tests:

```bash
python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
```
```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

Rebuild the engine (`cmake`, `cl` and `msbuild` are not on PATH). **Run this
from PowerShell, never Bash** — MSYS path translation rewrites `/p:` and `/m`
into an MSB1008 that reads like a bad project argument:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" build_python\clash_royale_env.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

**The toolchain differs between the machines this repo is worked on — probe,
don't inherit.** See CLAUDE.md's "Environment" section for the two commands
that settle it. `\18\` appeared here until 2026-08-24 and matches no machine
on record.

The post-build copy into `python_ai/` fails with MSB3073 if any Python process
holds the `.pyd` open — Windows will not overwrite a mapped DLL. That looks
exactly like a broken compile and almost never is.

## The rules that matter

- **`include/`, `src/` and `python_ai/` are READ-ONLY unless explicitly asked.**
  A training run is usually live against them.
- **Never change C++ without confirming the exact diagnosis and the exact edit
  first**, even when clearly justified. Propose it in
  `perception/UPSTREAM_REQUESTS.md`.
- **Any gameplay-affecting engine change invalidates the checkpoints' win-rate
  history.** Say so when proposing one.
- **Don't put a second copy of an engine constant in Python.** Derive it from
  the bindings.

`perception/` is the one place to edit freely.
