# Reflex Lab Implementation Plan

> Spec: `docs/design/specs/2026-09-28-reflex-lab-design.md`. Executed inline
> by the same agent that wrote it (the maintainer is asleep and asked for
> autonomous execution), so this plan pins interfaces, commands and gates
> rather than duplicating every line of code.

**Goal:** a static page where the real engine (WASM) scores a tiny JS policy
learning where and when to drop one defender against one attacker, plus a
drag-and-drop challenge the learner then tries to beat.

**Architecture:** one C++ `LabEngine` over the unchanged engine headers,
compiled three ways (native CLI for curation and dev, WASM for the page);
a readable JS learner shared by the worker and Node tests; a UI in the
viewer's visual language that talks only to a worker.

**Tech:** C++20 (MSVC now, Emscripten later), vanilla JS (classic scripts
and a classic worker; no bundler, no npm dependencies), Node 24 for tests,
headless Chrome via `tools/promo/cdp.py` for UI tests.

## Global constraints

- `include/`, `src/`, `python_ai/` are READ-ONLY. Nothing here edits them.
- No second copy of an engine constant in JS: geometry, legality, card facts
  come from `LabEngine`.
- Colours: the viewer's `:root` tokens verbatim. Font: Unbounded.
- Copy: the miniature caption, credits (Ambash; AI tools used), unofficial /
  not affiliated with Supercell.
- Destroyed towers read -1 from `getTowerHp`; always clamp to 0.
- Commit on `feat/reflex-lab`; never `git add -A`.

## File map

```
web/lab/engine/lab_engine.h    LabEngine (header-only C++, no bindings)
web/lab/engine/lab_wasm.cpp    embind layer
web/lab/engine/build.ps1       emcc -> web/lab/engine.js + engine.wasm
tools/lab/lab_cli.cpp          native driver: probe | table | check | serve
tools/lab/build_cli.ps1        cl.exe -> tools/lab/out/lab_cli.exe
tools/lab/dev_server.mjs       static files + POST /engine -> lab_cli serve
tools/lab/curate.mjs           tables -> learner convergence -> web/lab/roster.json
tools/lab/test_learner.mjs     gradient check, synthetic convergence
tools/lab/check_lab_ui.py      headless drag/drop + layout checks
tools/lab/parity.mjs           WASM vs native, 200 tries (needs the WASM build)
web/lab/learner.js             policy network + REINFORCE
web/lab/engine_client.js       one async API over WASM or the dev backend
web/lab/worker.js              training loop, challenge stepping
web/lab/board.js               renderer (viewer style, engine geometry)
web/lab/index.html lab.css lab.js   the page
web/lab/roster.json            generated
.github/workflows/lab-pages.yml
```

## LabEngine interface (every later task depends on these names)

```cpp
namespace lab {
constexpr int WINDOW_TICKS = 300, DELAY_STEPS = 11, DELAY_TICK_STEP = 5,
              SPAWN_ROWS = 6, NO_DEFENCE = -1;
class LabEngine {
 public:
  explicit LabEngine(unsigned seed = 1);
  std::string cardsJson(const std::vector<int>& ids) const;   // name, cost, flags
  std::string arenaJson() const;           // width,height,river rows,bridge cols
  std::string setMatchup(int attackerId, int defenderId);    // -> matchup JSON
  float rollout(int spawn, int cell, int delay) const;       // tower HP lost
  void rolloutBatch(const int* sca, int n, float* out) const;
  std::string framesJson(int spawn, int cell, int delay) const;
  void liveStart(int spawn);
  std::string liveStepJson(int ticks);     // frames for the ticks advanced
  bool livePlace(int cell);                // defender now; false if illegal/used
  bool liveDone() const;  int liveTick() const;  float liveDamage() const;
};
}
```

Matchup JSON: `{attacker, defender, spawns:[[x,y]...], d0:[...],
cells:[[x,y]...], delaySteps, delayTickStep, windowTicks}` (spawns with
`d0 == 0` already removed). Frame JSON: `{t, towers:[[team,slot,hp,max]...],
e:[[id,cardId,team,x,y,hp,maxHp,flags,symbol,name]...]}`; flags bit0 flying,
bit1 building, bit2 tower, bit3 deploying.

## Tasks

### 1. LabEngine + native CLI, pinned against the .pyd
- `lab_engine.h`, `lab_cli.cpp probe`, `build_cli.ps1`.
- Gate: `lab_cli probe` prints, for Hog (15) at (14,20) vs Cannon (25):
  no defence 2534; Cannon (11,9) delay 0 -> 0; delay idx 4 (20 ticks) -> 951.
  These are the `.pyd` numbers from the design probe, corrected for the -1
  destroyed-tower sentinel. Also: two identical rollouts agree; a rollout
  after 1,000 others agrees with a fresh one (no hidden history).
- Commit.

### 2. Outcome tables and the per-card engine checks
- `lab_cli table <att> <def> <out.bin>`: reward fraction for every
  spawn x cell x delay (float32), plus a JSON sidecar (spawns, cells, d0,
  best per spawn, random mean per spawn).
- `lab_cli check <att> <def>`: stall detector (a unit stationary >= 30 ticks
  with nothing in its reach), attacker-reaches-tower count.
- Gate: Hog vs Cannon table built; best-per-spawn mean and random mean
  printed and sane (best > random, best <= 1).
- Commit.

### 3. The learner
- `learner.js` (UMD-ish: `self.Learner` in the worker, `module.exports` in
  Node). Seeded PRNG, MLP, RBF placement head, autoregressive delay head,
  REINFORCE + per-spawn baseline, Adam, entropy bonus, greedy policy,
  `policyMap(spawn)` for the heatmap.
- `test_learner.mjs`: finite-difference gradient check (rel err < 1e-4);
  synthetic table where the optimum moves with the spawn -> greedy picks it.
- Gate: on the Hog vs Cannon table the greedy policy reaches >= 90% of
  best-per-spawn on the eval spawns within the try budget, 3 seeds.
- Commit.

### 4. Curation suite -> roster.json
- `curate.mjs`: for every candidate attacker x defender: build table and
  checks via `lab_cli` (parallel processes), run the learner gate on 3 seeds,
  record facts. A card passes if all its matchups pass the engine checks and
  the learner gate. Writes `web/lab/roster.json` with the passing cards,
  per-matchup `d0` summary, `best`, `random`, and the 5 challenge spawns.
- Gate: >= 8 attackers and >= 8 defenders pass, Hog Rider and Cannon among
  them; failures listed with reasons in the commit message.
- Commit.

### 5. Dev backend
- `lab_cli serve`: line protocol on stdin/stdout.
- `dev_server.mjs`: static files from `web/`, POST `/engine` forwards one
  line and returns one line. `engine_client.js` wraps both backends behind
  `await client.call(method, ...args)`.
- Gate: `curl` round-trip of `matchup 15 25` and a rollout batch matches the
  CLI directly.
- Commit.

### 6. The page
- `board.js`, `worker.js`, `index.html`, `lab.css`, `lab.js`.
- Gate: in the browser pane via the dev backend: intro -> matchup ->
  challenge (5 rounds, drag and drop) -> train (heatmap follows the preview
  attacker, curve crosses the baseline) -> Show me -> results side by side.
  No console errors.
- Commit.

### 7. UI tests
- `check_lab_ui.py`: headless Chrome; mouse drag and touch drag place on a
  legal cell; drop off-board cancels; illegal cell refused; layout fits at
  390x844 and 1440x900 with no horizontal scroll.
- Gate: all pass. Commit.

### 8. WASM packaging
- `lab_wasm.cpp`, `build.ps1`, `lab-pages.yml`, `parity.mjs`.
- Gate (when emsdk exists): `parity.mjs` 200/200 exact. Until then: the
  embind file compiles natively with a stub guard, and the workflow is
  syntactically valid.
- Commit.

### 9. Docs and hand-off
- README section, `.claude/CLAUDE.md` short section, `docs/TODO.md` item.
- Re-run every gate. Fast-forward `main` only if all pass (never push).
