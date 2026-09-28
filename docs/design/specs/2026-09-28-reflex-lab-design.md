# Reflex Lab: a live, in-browser learning demo for Show HN

Status: approved 2026-09-28 (design sessions with the maintainer). Built
autonomously from this spec overnight 2026-09-28/29.

## Goal

Give a Show HN visitor something they can *touch* that shows the project's
inner workings honestly: the real C++ simulator running in their browser tab,
and a small neural network visibly learning a defensive reflex against it in
well under a minute. Zero install, static hosting.

It is a **miniature** of the real problem -- one attacker, one defender, no
elixir -- and the page says so. The real agent plays full matches and trains
for days on a laptop CPU; it cannot visibly improve inside a browser session
(one PPO update takes ~47 s and a fresh net won 0 of its first 100 games), so
the lab does not pretend it can.

## What the visitor experiences

Look: the replay viewer's colour tokens exactly (`web/viewer.html` `:root`),
with **Unbounded** (Google Fonts, chosen by the maintainer from three
candidates) for all typography. Premium, easy on the eyes, satisfying to use.

1. **Intro.** A short screen that says what they are about to see and what is
   expected of them, then the lab.
2. **Pick a matchup** from a *curated* roster: ~10 attackers and ~10
   defenders, each of which passed the curation suite below. Not the whole
   registry: a card that misbehaves would make the project look amateurish.
   Default: Hog Rider vs Cannon. The no-defence damage is always shown, so a
   pointless pairing reads "best found: saves 0%" rather than looking broken.
3. **The challenge (skippable).** Five fixed attacks play in real time. The
   visitor drags the defender card from a tray and releases it on the board
   to place it -- where AND when in one gesture. Pointer events: identical on
   mouse and touch; on touch the ghost floats above the finger so it is not
   hidden. Snaps to the nearest legal cell (legal = the engine's own
   predicate), shows red over an illegal one, and a release off the board
   cancels. Each round scores the % of no-defence tower damage prevented; the
   average becomes a reference line on the learning curve.
4. **Train.** Every try drops the attacker at a random spot on the enemy side
   of the river; the learner commits to a cell and a drop delay; the engine
   plays it out. Live:
   - a heatmap of where the learner wants to place, for a preview attacker
     the visitor can drag around -- the heatmap follows it ("it reads the
     board");
   - the learning curve (greedy damage prevented) against the random-placement
     baseline, the exhaustive-search best, and the visitor's score;
   - counters: tries, tries/second, elapsed.
5. **Show me** animates the learner's current best answer to the preview
   attack. **After training** the learner plays the visitor's five challenge
   attacks, side by side with the visitor's own tries: "You saved 62%. The
   learner saved 81% after 40 s of training."
6. **Footer:** the miniature caption, repo link, credits (Ambash, who built
   most of the card roster; AI tools were used), and "unofficial, not
   affiliated with Supercell".

## The learning problem

- **State**: the cell the attacker was dropped on -- any legal cell for the
  attacker (team 1, the engine's predicate) in the first rows past the river
  on the enemy side. Rows are derived from the engine's river bounds, not
  restated. Spawns whose no-defence damage is 0 are dropped (the attacker
  cannot reach a tower in the window), so no try is scored against zero.
- **Action**, autoregressive like the real agent: a legal cell for the
  defender (engine predicate, team 0), then a drop delay 0-5 s in 0.5 s steps
  conditioned on that cell.
- **Reward**: `(D0 - D) / D0`, the fraction of no-defence damage to our three
  crown towers prevented over a 30 s window, clipped to [-1, 1]. The rollout
  ends early once no enemy unit remains (so a Battle Ram's Barbarians still
  count). `D0` is measured per spawn when a matchup loads.
- **Engine semantics**: both bodies are placed with `ClashEnv::inject` and
  keep the real 1 s deploy time; ticks advance with `stepSelfPlayFast` and
  no-op cards, so the C++ heuristic opponent never acts.
- **Learner** (`learner.js`, plain JS, no ML library, readable): spawn
  one-hots -> hidden layer -> (a) placement logits as a coarse radial-basis
  map evaluated at every legal cell, so a try at one cell teaches its
  neighbours (the real agent's placement head is convolutional for the same
  reason); (b) delay logits from the hidden layer plus the chosen cell's
  basis features. REINFORCE with a per-spawn running baseline, Adam, a small
  entropy bonus. Greedy evaluation on a fixed spawn set drives the curve.

Measured before designing (native engine, Hog at (14,20)): no defence 2,535
tower HP (the whole Princess Tower); Cannon at (11,9) dropped at once: 0;
the same cell 2 s late: 951. Timing matters as much as the cell. A 30 s
rollout costs 1.67 ms driven tick by tick from Python.

## Architecture

| Piece | Role |
|---|---|
| `web/lab/engine/lab_engine.h` | `LabEngine`: matchup setup, spawn band, legal cells, `D0`, batched rollouts, a live stepper for the challenge, frames for animation, and the arena geometry -- all derived from the engine headers, which are included **unchanged**. Plain C++, no bindings. |
| `web/lab/engine/lab_wasm.cpp` | Thin embind layer over `LabEngine` -> `engine.js` + `engine.wasm` (Emscripten). |
| `tools/lab/lab_cli.cpp` | The same `LabEngine` compiled natively (MSVC): the curation suite, exhaustive outcome tables, and a line-JSON `serve` mode for local development. |
| `web/lab/learner.js` | Network, sampling, gradient step. No DOM, no engine; runs in the worker and in Node. |
| `web/lab/worker.js` | Owns the engine and the learner; runs the training loop and posts heatmaps/metrics a few times a second so the UI never stutters. |
| `web/lab/index.html`, `lab.css`, `lab.js`, `board.js` | The UI and its renderer. |
| `web/lab/roster.json` | Generated by the curation suite: the cards that passed and the measured facts per matchup. The page reads it; nobody hand-edits it. |
| `.github/workflows/lab-pages.yml` | Builds the WASM with emsdk in CI and deploys `web/` to GitHub Pages. |

**Deviation from the approved section 3 (recorded here on purpose):** it said
the viewer's board drawing would be extracted into a shared
`web/board_renderer.js`. The viewer's renderer is bound to its replay-file
globals (`CARD_META`, `SYMBOL_INDEX`, `ROLL_ORIGINS`, `gameData`) and to the
promo export tooling that drives it headless. Extracting it the night before
a launch would put both at risk for no visible gain. The lab gets its own
renderer (`board.js`) in the same visual style instead; it takes **every**
piece of geometry from the engine (river rows, bridge cells via
`Board::isOnBridge`, towers from the board's own entities), so it holds no
second copy of any engine constant, which is the rule that matters.

`include/`, `src/` and `python_ai/` are untouched. The training run is
unaffected.

## Testing (each must pass before the page ships)

- **Engine parity**: `LabEngine` (native) reproduces the `.pyd`'s damage for
  the probe cases exactly; the WASM build, run in Node, reproduces the native
  build on 200 fixed tries exactly (the engine is deterministic -- equality,
  not tolerance).
- **Curation suite** decides the roster. A card passes if, in every matchup
  it is part of: the attacker reaches a tower from the spawn band; no unit
  stalls (stationary with nothing in reach -- the engine's own soak
  criterion); and the learner reaches >= 90% of the exhaustive-search best on
  the evaluation spawns within its try budget, on 3 seeds. Failures are
  dropped. Starting candidates:
  attackers Hog Rider, Giant, Royal Giant, Balloon, Battle Ram, Ram Rider,
  Prince, Mini P.E.K.K.A, Valkyrie, Skeleton Army; defenders Cannon, Tesla,
  Inferno Tower, Tombstone, Bomb Tower, Mini P.E.K.K.A, Knight, Valkyrie,
  Musketeer, Skeleton Army.
- **Learner**: finite-difference gradient check; convergence on a synthetic
  problem.
- **Drag and drop**: scripted mouse and touch drags in headless Chrome
  (`tools/promo/cdp.py`) at desktop and phone sizes: drop, cancel off-board,
  illegal cell. Plus a hands-on pass in the browser pane.
- **Layout**: fit at phone and desktop widths, in the manner of
  `tools/viewer_fixtures/check_fit.py`.

## Out of scope (v1)

Spells as defenders (their "where" is on the attacker's path and their value
hinges on sub-second timing -- needs its own design); multiple defenders or
elixir; the trained agent itself.

## Open at hand-off

Emscripten is not installed on the build machine, and downloading it (~1 GB)
needs the maintainer's explicit yes, which was asked and not yet given. So
the WASM compile is the one step left for the maintainer: everything else is
built and tested against the native engine, and the build script is a single
command.

## Revision after the first hands-on review (2026-09-28)

The maintainer's review of the first build, and what changed:

- **General-purpose copy.** "Show HN" is gone from the intro; the page is a
  demo in its own right.
- **Royal Giant pairings failed to load.** The committed roster was a
  provisional one with a single Royal Giant pairing, while the page offered
  nine. The full curation run replaces it, the page no longer offers a pairing
  the roster lacks, and `check_lab_ui.py` now loads every offered pairing.
- **Attackers: five or six**, with Battle Ram and Goblin Barrel. The Goblin
  Barrel is a spell, so the engine surface gained spell attackers that drop
  troops (measured, not listed: cast it on an empty board and look for
  bodies), aimed at the cells within 2 of either of our Princess Towers. A
  rollout also keeps running while the barrel is in the air. The Battle Ram
  deviated from the real card (it never broke on a building); fixed in the
  engine on the maintainer's go-ahead as UPSTREAM item 34, base card only
  (the Evolution, a separate card, really does keep swinging).
- **Defenders: the Knight is cut** (a weaker Valkyrie against every
  attacker; against the barrel 80% vs 99% best), and the Tombstone stays out
  (item 33).
- **Onboarding.** A ten-second demo after the intro, always Hog Rider vs
  Cannon: countdown, the Hog drops, a hand drags the Cannon from the tray and
  lets go on the engine's drop tick, the Hog turns onto the Cannon. Recorded
  from the engine at load; "How to play" replays it.
- **Buttons on the board.** A card in the middle of the board carries the
  next action (start round, next round with the last score, watch the AI,
  Train), and a bar under the board carries Train/Pause, Show me and See
  results. The side panel keeps its buttons.
- **Chart labels clipped** ("100%", "random 41%"): the margins are now
  measured from the label text, and the labels are kept inside the plot.
- **Results show the process.** Each attack's heatmap is snapshotted during
  training and replayed converging, paced by how much it changes, before the
  answer, the replays (at 0.67x) and the verdict appear.
- **Background.** A looping field of small squares in the two team colours,
  in waves; one still frame under reduced motion.
- **Learner.** The entropy bonus now anneals 0.1 -> 0.005 over 10,000 tries
  (it was a constant 0.01). On Giant vs Cannon, 3 seeds at batches of 16 and
  64, that cut the runs missing 90% of the best from 5 of 6 to 1 of 6 (none
  at 16). The curation gate trains at the page's batch of 16.
