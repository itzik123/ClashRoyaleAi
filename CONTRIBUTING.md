# Contributing to ClashRoyaleEnv

Thanks for your interest! This is a research project built mostly by one person
so far, and outside help makes a real difference, whether it's a bug report, a
fix, a feature, or a sharp question.

## Where to start

- **Pick a [good first issue](https://github.com/itzik123/ClashRoyaleAi/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).**
  Each one explains why it matters, which files to look at, and what "done"
  means. Comment on the issue before you start, so two people don't do the same
  work.
- **Ask in [Discussions](https://github.com/itzik123/ClashRoyaleAi/discussions).**
  Use *Q&A* for setup problems and questions, *Ideas* for proposals, and
  *Show and tell* for things you built with the engine.
- **Planning something big?** Open an issue or a discussion first. Some parts
  of the engine are calibrated against measurements that aren't obvious from
  the code, and a short conversation can save you a rewrite.

## You don't need C++ to help

- **The replay viewer** (`web/viewer.html`) is one HTML file with no
  dependencies. Open it in a browser and drop a replay JSON onto it.
- **Card accuracy.** If you know Clash Royale well, compare how a card behaves
  in the engine with the real game. A mismatch reported with a source (for
  example, the card's published stats) is a valuable bug report. The engine's
  HP and damage values match level-11 cards.
- **Documentation.** If something was confusing when you set up or read the
  code, a fix to the docs helps the next person.

## Setting up

Today the supported setup is **Windows** with Visual Studio 2022 (with the
*Desktop development with C++* workload), Python 3.11 and Git. One script does
the whole setup:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_dev_env.ps1
```

Linux support is being worked on in
[#2](https://github.com/itzik123/ClashRoyaleAi/issues/2), and a Docker image in
[#3](https://github.com/itzik123/ClashRoyaleAi/issues/3).

Two errors that look worse than they are:

- **`ImportError: DLL load failed`** when importing `clash_royale_env` means you
  are running a Python other than 3.11. The build isn't broken. Use
  `python_ai\venv\Scripts\python.exe` or `py -3.11`.
- **A rebuild fails while copying the `.pyd` (MSB3073).** A Python process
  still has the old module loaded, and Windows won't overwrite it. The compile
  itself succeeded. Close that process and build again.

## Running the tests

```powershell
.\build_python\Release\ClashRoyaleTests.exe                          # C++
python_ai\venv\Scripts\python.exe -m pytest python_ai\tests -q       # training stack
perception\.venv\Scripts\python.exe -m pytest perception\tests -q    # perception
```

- **In a healthy C++ run exactly one case fails "as expected".** It pins a
  known open collision bug, and the runner still exits 0. A non-zero exit or a
  second failure is a real problem.
- **CI runs the C++ suite** on pull requests that touch `include/`, `src/`,
  `tests/` or `CMakeLists.txt`. The Python suites aren't in CI yet, so run them
  locally and say in your pull request that you did.
- **New C++ test files go in `tests/core/` or `tests/entities/`.** The CMake
  glob only looks there. After adding a file, run the build **twice**: the
  first build regenerates the project and can report success without your new
  file. Then check that your tests really run:
  `ClashRoyaleTests.exe "[your-tag]"` should report more than zero cases.

## Changing the engine

The engine is the foundation everything else is measured against, so changes
there follow a few rules.

1. **Say whether your change affects gameplay.** Anything that changes how a
   match plays out, such as card stats, timing, targeting, pathing or placement
   rules, makes trained models and past win rates out of date. That's often
   worth it. Just say so clearly in the pull request.
2. **Show the behaviour, not just the code.** Include a test that fails before
   your change and passes after it. For a stat or balance change, link the
   source for the new value.
3. **Don't copy engine constants.** Arena geometry lives in
   `include/core/ArenaLayout.h` and is exposed to Python through
   `python_ai/engine_constants.py`. Read values from there. Copies have gone
   stale several times in this project's history. Where a copy truly can't be
   avoided (the HTML viewer can't read the engine), add a comment naming the
   header that owns the value.
4. **Keep it deterministic.** The same inputs must always give the same match.
   The engine's only randomness, the opening-hand shuffle and the built-in
   heuristic opponent, is controlled by `seed()`.

Units worth knowing before you read the code:

- **Time:** 10 ticks = 1 second.
- **Speed:** tiles per tick.
- **Board:** 18 columns by 34 rows. Column `i` covers `x` from `i - 0.5` to
  `i + 0.5`.

For the full background, read the engine facts in
[`.claude/CLAUDE.md`](.claude/CLAUDE.md). It's long, so search it for the part
you're changing. The reasoning behind most decisions, including the ones that
were reversed, is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Changing the Python side

- **Don't commit generated files:** checkpoints (`*.pth`), TensorBoard logs
  (`runs/`), replays, or other large binaries.
- **Changing the observation or action layout breaks existing checkpoints.**
  Say so in the pull request.

## Pull requests

- Fork the repo, create a branch from `main`, and keep each pull request to
  one focused change.
- In the description, say what changed and why, how you tested it (the
  commands and their results), and whether it affects gameplay.
- Match the style of the code around your change. Comments should explain why
  the code does something, not restate what it does.
- Make sure CI passes.

## Match recordings

The perception pipeline needs recordings of real matches, but a recording
shows other players' names. If you'd like to contribute recordings, start a
discussion first so we can agree on the format and how to handle that.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE), like the rest of the project.
