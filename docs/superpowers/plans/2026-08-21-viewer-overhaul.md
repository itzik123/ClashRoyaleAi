# Replay Viewer Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `web/viewer.html` with a real-Clash-Royale map palette and a
minimalist greyscale dark theme, fix the entity-inspector `?` bug, and add a
Teacher Debug "Simulation View" that visualizes the teacher's candidate
rollouts.

**Architecture:** `viewer.html` stays a **single self-contained file** — it is
opened from `file://` with a replay dropped onto it, and splitting it into
modules would break that workflow. Internally the board renderer is refactored
to take an explicit render-context object so the main arena and the Simulation
View's mini-boards share one drawing path. Teacher rollout data is captured by
a subclass wrapper living **outside** `python_ai/` (a training run is live) and
merged into the replay JSON as a purely additive `teacherDebug` block.

**Tech Stack:** Vanilla HTML/CSS/Canvas 2D (no build step, no dependencies),
Python 3.11 via `python_ai/venv` for fixture generation, `clash_royale_env.pyd`
for the engine.

## Global Constraints

- **No file under `python_ai/` may be created or modified.** A training run is
  live; `train.py` `subprocess.Popen`s `train_selfplay.py`, so a spawned
  process would import edited code. Importing is fine; writing is not.
- **No C++ change.** `include/` and `src/` are read-only. Engine-side fixes go
  to `perception/UPSTREAM_REQUESTS.md` as proposals.
- `viewer.html` remains one self-contained file with **no external requests**
  except the existing Google Fonts `@import`.
- Every replay-derived value must degrade gracefully: **no console error and no
  empty framed region** for any missing field.
- Engine geometry facts, verbatim from `include/core/ArenaLayout.h` and
  `include/core/Board.h`:
  - `WIDTH = 18`, `HEIGHT = 34`
  - river band `riverY_start = 15.5`, `riverY_end = 17.5` → integer rows **16 and 17**
  - `LEFT_BRIDGE_X = 2.5`, `RIGHT_BRIDGE_X = 14.5`, `BRIDGE_HALF_WIDTH = 1.0`
    → bridge cells **2, 3** and **14, 15**
  - `KING_Y_TEAM0 = 2.5`, `PRINCESS_Y_TEAM0 = 6.0`, `LEFT_LANE_X = 3.0`, `RIGHT_LANE_X = 14.0`
- Use the training venv for all Python:
  `python_ai/venv/Scripts/python.exe`
- Scratch dir for temp files:
  `C:\Users\itzik\AppData\Local\Temp\claude\C--Users-itzik-source-repos-ClashRoyaleEnv\30b97aa9-bccf-4262-8d58-3e9e0302100f\scratchpad`

---

### Task 1: Fixture replays and the in-page test harness

Everything after this is verified against these files, so they come first.

**Files:**
- Create: `tools/viewer_fixtures/make_fixtures.py`
- Create: `web/fixtures/modern.json` (generated)
- Create: `web/fixtures/legacy.json` (generated)
- Modify: `web/viewer.html` (add `window.__viewerTests` export block)

**Interfaces:**
- Consumes: nothing.
- Produces: `web/fixtures/modern.json` and `web/fixtures/legacy.json`;
  `window.__viewerTests` — an object exposing the viewer's pure functions for
  in-page assertion, populated at the end of the `<script>` block.

- [ ] **Step 1: Write the fixture generator**

Create `tools/viewer_fixtures/make_fixtures.py`:

```python
"""Generate replay fixtures for web/viewer.html development.

`modern.json`  -- a current-schema replay: cardMeta, cardId, result.
`legacy.json`  -- the same match with cardMeta, cardId and result STRIPPED,
                  standing in for a replay saved before those fields existed.
                  This is the graceful-degradation fixture; the viewer must
                  render it with no console error.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: F401  -- puts the unpackaged .pyd on sys.path
import clash_royale_env as E

DECK = [15, 6, 25, 40, 24, 72, 33, 7]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web", "fixtures")


def play_match(path, seed=17, decisions=220):
    """A teacher-vs-teacher match, saved through the engine's own logger."""
    import numpy as np
    from python_ai.opponents.teacher import UtilityTeacher

    env = E.ClashRoyaleEnv(DECK, DECK, 3600)
    env.seed(seed)
    env.reset()
    t0 = UtilityTeacher(DECK, team=0); t0.set_stage(5); t0.reset()
    t1 = UtilityTeacher(DECK, team=1); t1.set_stage(5); t1.reset()
    for _ in range(decisions):
        if env.is_game_over():
            break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0)
        a1 = t1.act(env, o1)
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    env.save_log(path)
    return path


def strip_to_legacy(src, dst):
    """Remove every field added after the original replay schema."""
    with open(src) as f:
        data = json.load(f)
    data.pop("cardMeta", None)
    data.pop("result", None)
    for tick in data["ticks"]:
        for ent in tick["entities"]:
            ent.pop("cardId", None)
            ent.pop("isFlying", None)
    with open(dst, "w") as f:
        json.dump(data, f)


def main():
    os.makedirs(OUT, exist_ok=True)
    modern = os.path.join(OUT, "modern.json")
    play_match(modern)
    strip_to_legacy(modern, os.path.join(OUT, "legacy.json"))
    for name in ("modern.json", "legacy.json"):
        p = os.path.join(OUT, name)
        print(f"{name}: {os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify both fixtures exist and differ correctly**

```bash
python_ai/venv/Scripts/python.exe tools/viewer_fixtures/make_fixtures.py
```

Expected: two sizes printed. Then verify the strip actually worked:

```bash
python_ai/venv/Scripts/python.exe -c "import json; m=json.load(open('web/fixtures/modern.json')); l=json.load(open('web/fixtures/legacy.json')); print('modern has cardMeta:', 'cardMeta' in m); print('legacy has cardMeta:', 'cardMeta' in l); print('modern ent has cardId:', 'cardId' in m['ticks'][0]['entities'][0]); print('legacy ent has cardId:', 'cardId' in l['ticks'][0]['entities'][0]); print('ticks:', m['totalTicks'])"
```

Expected: `True / False / True / False`, and a nonzero tick count.

- [ ] **Step 3: Add the in-page test export to `viewer.html`**

`viewer.html` has no test framework and needs none. Expose the pure functions
so assertions can be run against a loaded page via the browser console. Add
immediately before the closing `</script>` tag:

```javascript
// Test surface. Pure functions only -- nothing here mutates viewer state.
// Used by the browser-driven checks in docs/superpowers/plans/; harmless in
// normal use (it only assigns references to already-defined functions).
window.__viewerTests = {
  getEntityMeta,
  buildSymbolIndex,
  formatMaxHp,
  teacherDebugForTick,
  bridgeCells: () => BRIDGE_CELLS,
  riverRows: () => RIVER_ROWS,
};
```

Note: `buildSymbolIndex`, `formatMaxHp`, `teacherDebugForTick`, `BRIDGE_CELLS`
and `RIVER_ROWS` are introduced in Tasks 3 and 6. Until then this block will
throw a `ReferenceError` on load, so **add only the entries whose symbols
already exist** (`getEntityMeta`) and extend the object as each task lands.

- [ ] **Step 4: Verify the page still loads clean**

Open `web/viewer.html`, load `web/fixtures/modern.json`, and check the console
is empty and `window.__viewerTests.getEntityMeta` is a function.

- [ ] **Step 5: Commit**

```bash
git add tools/viewer_fixtures/make_fixtures.py web/fixtures/ web/viewer.html
git commit -m "test: add replay fixtures and in-page test surface for the viewer"
```

---

### Task 2: Refactor the board renderer to take a render context

`renderTick`, `drawEntity`, `drawTypeBadge` and `gameToCanvas` all close over
the module globals `ctx`, `canvas`, `CELL_SIZE`, `PADDING` and `gameData`. The
Simulation View needs to draw the same board into small canvases at a different
scale, so these must become parameterized. **This task changes no visuals** —
it is a pure refactor, and the check is that the rendered output is unchanged.

**Files:**
- Modify: `web/viewer.html:903-916` (`gameToCanvas` / `canvasToGame`)
- Modify: `web/viewer.html:970-1036` (`renderTick`)
- Modify: `web/viewer.html:1038-1170` (`drawTypeBadge`, `drawEntity`)

**Interfaces:**
- Consumes: `web/fixtures/modern.json` from Task 1.
- Produces:
  - `makeRenderCtx(ctx2d, boardWidth, boardHeight, cellSize, padding) -> rc`
    where `rc = {ctx, boardWidth, boardHeight, cellSize, padding, width, height}`
  - `gameToCanvas(rc, gx, gy) -> {x, y}`
  - `drawBoard(rc, tickData, opts)` where
    `opts = {selectedEntityId: number|null, showLabels: boolean, ghost: boolean}`
  - `mainRc()` — the render context for the primary arena canvas.

- [ ] **Step 1: Add the render-context constructor**

Insert after the `CELL_SIZE` declaration (`viewer.html:723`):

```javascript
// A render context bundles everything a board drawing needs, so the main arena
// and the Simulation View's mini-boards can share one drawing path instead of
// growing a second renderer that drifts from the first.
function makeRenderCtx(ctx2d, boardWidth, boardHeight, cellSize, padding) {
  return {
    ctx: ctx2d,
    boardWidth, boardHeight, cellSize, padding,
    width: boardWidth * cellSize + padding * 2,
    height: boardHeight * cellSize + padding * 2,
  };
}

function mainRc() {
  return makeRenderCtx(ctx, gameData.boardWidth, gameData.boardHeight,
                       CELL_SIZE, PADDING);
}
```

- [ ] **Step 2: Re-sign the coordinate helpers**

Replace `gameToCanvas` / `canvasToGame` (`viewer.html:903-916`) with:

```javascript
function gameToCanvas(rc, gx, gy) {
  return {
    x: rc.padding + gx * rc.cellSize + rc.cellSize / 2,
    y: rc.padding + (rc.boardHeight - 1 - gy) * rc.cellSize + rc.cellSize / 2
  };
}

function canvasToGame(rc, cx, cy) {
  return {
    gx: (cx - rc.padding) / rc.cellSize,
    gy: rc.boardHeight - 1 - (cy - rc.padding) / rc.cellSize
  };
}
```

- [ ] **Step 3: Re-sign the drawing functions**

Change the signatures to take `rc` first and replace every use of the globals:

- `function drawTypeBadge(ent, pos, radius)` → `function drawTypeBadge(rc, ent, pos, radius)`; inside, `ctx` → `rc.ctx`, `CELL_SIZE` → `rc.cellSize`.
- `function drawEntity(ent, isSelected)` → `function drawEntity(rc, ent, isSelected)`; same substitutions, and its internal `gameToCanvas(...)` calls become `gameToCanvas(rc, ...)`.
- `function renderTick()` → split into `renderTick()` (thin) and `drawBoard(rc, tickData, opts)` (everything else):

```javascript
function renderTick() {
  if (!gameData) return;
  const tickData = gameData.ticks[currentTick];
  if (!tickData) return;
  drawBoard(mainRc(), tickData, {
    selectedEntityId: selectedEntityId,
    showLabels: true,
    ghost: false,
  });
}
```

`drawBoard` contains the former body of `renderTick`, with `ctx` → `rc.ctx`,
`w`/`h` → `rc.width`/`rc.height`, `PADDING` → `rc.padding`,
`CELL_SIZE` → `rc.cellSize`, `gameData.boardWidth` → `rc.boardWidth`,
`gameData.boardHeight` → `rc.boardHeight`, and the two entity loops passing
`rc` through to `drawEntity`. The territory-label block is wrapped in
`if (opts.showLabels)`.

- [ ] **Step 4: Fix the two remaining call sites**

The hover and click handlers (`viewer.html:1474-1516`) call `gameToCanvas(ent.x, ent.y)`
and use `CELL_SIZE` directly. Add `const rc = mainRc();` at the top of each
handler and change those calls to `gameToCanvas(rc, ent.x, ent.y)` and
`rc.cellSize`.

- [ ] **Step 5: Verify the render is byte-identical**

Load `web/fixtures/modern.json`, scrub to a tick with several entities, and
confirm visually that the board is unchanged from before the refactor. Check
the console is empty and that hover tooltips and click-to-select still work.

- [ ] **Step 6: Commit**

```bash
git add web/viewer.html
git commit -m "refactor: parameterize the board renderer on an explicit render context"
```

---

### Task 3: The map — real Clash Royale palette, and the bridge-column fix

**Files:**
- Modify: `web/viewer.html:19-34` (`:root` map color variables)
- Modify: `web/viewer.html` `drawBoard` (from Task 2)

**Interfaces:**
- Consumes: `drawBoard(rc, tickData, opts)`, `makeRenderCtx` from Task 2.
- Produces: `BRIDGE_CELLS` (array of `[startCol, endCol]`), `RIVER_ROWS`
  (`[startRow, endRow]`), both module constants.

- [ ] **Step 1: Add the geometry constants, derived from the engine's headers**

The current code hardcodes `[[3, 4], [13, 14]]`, which is the **pre-2026-08-21
arena** and is wrong in both directions — columns 4 and 13 are painted as
bridge but are water, and columns 2 and 15 are bridge painted as water. This is
the same stale-second-copy failure that hit the observation encoder. The replay
JSON does not carry arena geometry, so per CLAUDE.md's rule these are hardcoded
**with a comment naming the header**, the same pattern `perception/geometry.py`
uses. Insert near `CELL_SIZE`:

```javascript
// Arena geometry. NOT derivable from the replay JSON, so hardcoded here with
// the header that owns each value named -- the pattern CLAUDE.md requires and
// perception/geometry.py already follows.
//
// include/core/ArenaLayout.h: LEFT_BRIDGE_X = 2.5, RIGHT_BRIDGE_X = 14.5.
// include/core/Board.h:       BRIDGE_HALF_WIDTH = 1.0.
// A bridge centre sits on the SEAM between two tiles, so 2.5 +/- 1.0 spans
// exactly cells 2 and 3, reproducing the real river row WWBBWWWWWWWWWWBBWW.
// This file previously said [[3,4],[13,14]] -- the arena that was corrected on
// 2026-08-21 -- and painted four of eighteen columns wrong.
const BRIDGE_CELLS = [[2, 3], [14, 15]];

// include/core/Board.h: riverY_start = 15.5, riverY_end = 17.5. Cell i covers
// [i-0.5, i+0.5], so the band is exactly integer rows 16 and 17.
const RIVER_ROWS = [16, 17];
```

- [ ] **Step 2: Replace the map color variables**

In `:root` (`viewer.html:19-34`), delete `--river-blue`, `--bridge-brown`,
`--grass-dark`, `--grass-light` and add:

```css
    /* Arena palette, matched to the real game: one continuous lawn with a
       two-tone mown checker, a cyan river, wood bridges and tan lane paths. */
    --grass-light: #93C756;
    --grass-dark: #84BC46;
    --river: #37BEDC;
    --river-edge: #2AA5C2;
    --bridge-deck: #C89B5C;
    --bridge-seam: #A87C46;
    --lane-path: #CFC08F;
    --arena-surround: #2E4A1C;
```

- [ ] **Step 3: Rewrite the board background in `drawBoard`**

Replace the gradient fill, grid lines, river, bridge and territory blocks with:

```javascript
  const css = getComputedStyle(document.documentElement);
  const C = n => css.getPropertyValue(n).trim();
  const cs = rc.cellSize, pad = rc.padding;

  rc.ctx.clearRect(0, 0, rc.width, rc.height);

  // Surround
  rc.ctx.fillStyle = C('--arena-surround');
  rc.ctx.fillRect(0, 0, rc.width, rc.height);

  // Lawn, as a mown checker. No team territory tint: the real arena is one
  // green field and team identity comes from tower and unit colour.
  for (let x = 0; x < rc.boardWidth; x++) {
    for (let y = 0; y < rc.boardHeight; y++) {
      rc.ctx.fillStyle = ((x + y) % 2 === 0) ? C('--grass-light') : C('--grass-dark');
      rc.ctx.fillRect(pad + x * cs, pad + y * cs, cs, cs);
    }
  }

  // Lane paths: bridge column down to each tower row, both halves.
  // ArenaLayout.h LEFT_LANE_X = 3.0 / RIGHT_LANE_X = 14.0.
  rc.ctx.fillStyle = C('--lane-path');
  for (const [c0, c1] of BRIDGE_CELLS) {
    rc.ctx.fillRect(pad + c0 * cs, pad, (c1 - c0 + 1) * cs, rc.height - pad * 2);
  }

  // River. RIVER_ROWS are game rows; canvas row = boardHeight - 1 - gameRow.
  const rTop = pad + (rc.boardHeight - 1 - RIVER_ROWS[1]) * cs;
  const rBot = pad + (rc.boardHeight - RIVER_ROWS[0]) * cs;
  rc.ctx.fillStyle = C('--river');
  rc.ctx.fillRect(pad, rTop, rc.width - pad * 2, rBot - rTop);
  rc.ctx.fillStyle = C('--river-edge');
  rc.ctx.fillRect(pad, rTop, rc.width - pad * 2, Math.max(1, cs * 0.12));

  // Bridges over the river band.
  for (const [c0, c1] of BRIDGE_CELLS) {
    const bl = pad + c0 * cs, br = pad + (c1 + 1) * cs;
    rc.ctx.fillStyle = C('--bridge-deck');
    rc.ctx.fillRect(bl, rTop, br - bl, rBot - rTop);
    rc.ctx.strokeStyle = C('--bridge-seam');
    rc.ctx.lineWidth = Math.max(1, cs * 0.06);
    for (let p = 1; p < 4; p++) {
      const py = rTop + (rBot - rTop) * (p / 4);
      rc.ctx.beginPath(); rc.ctx.moveTo(bl, py); rc.ctx.lineTo(br, py); rc.ctx.stroke();
    }
  }
```

The lane-path fill must be drawn **before** the river so the river covers it,
and the bridges after so they sit on top.

- [ ] **Step 4: Verify against the reference**

Load `web/fixtures/modern.json`, screenshot the arena, and compare against the
reference frame. Then assert the geometry constants in the console:

```javascript
JSON.stringify(window.__viewerTests.bridgeCells()) === '[[2,3],[14,15]]'
```

Expected: `true`. (Add `bridgeCells` / `riverRows` to `window.__viewerTests`
now that they exist.)

- [ ] **Step 5: Verify a unit standing on a bridge is drawn on the bridge**

Scrub to a tick where an entity is crossing (`y` between 15.5 and 17.5) and
confirm it renders over bridge planks, not over open water. Before this fix a
unit at `x = 2.5` rendered over water.

- [ ] **Step 6: Commit**

```bash
git add web/viewer.html
git commit -m "feat: real-game arena palette, and fix bridges drawn on the pre-2026-08-21 columns"
```

---

### Task 4: Greyscale dark theme for all chrome

**Files:**
- Modify: `web/viewer.html:11-34` (`:root`), `:60-84` (header), and every rule
  referencing `--elixir-purple`.

**Interfaces:**
- Consumes: nothing.
- Produces: the finalized `:root` token set.

- [ ] **Step 1: Replace the chrome tokens**

```css
    --bg-primary: #0B0B0C;
    --bg-secondary: #101012;
    --bg-card: #1C1C1F;
    --bg-hover: #26262A;
    --border: #26262A;
    --text-primary: #F5F5F5;
    --text-secondary: #A1A1A6;
    --text-muted: #6E6E73;
    --accent: #F5F5F5;
```

Keep `--blue-team`, `--blue-team-dim`, `--red-team`, `--red-team-dim` — team
identity is functional, not decoration.

- [ ] **Step 2: Delete `--elixir-purple` and every reference**

Find them all first:

```bash
grep -n "elixir-purple\|c471ed" web/viewer.html
```

Replace each: the elixir bar fill becomes `--text-primary`, the spell badge
`--badge-spell` becomes `#B9B9C0`, and the header `h1` loses its gradient:

```css
  .header h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }
```

Delete the three `-webkit-background-clip` / `-webkit-text-fill-color` /
`background: linear-gradient(...)` lines from that rule.

- [ ] **Step 3: Neutralize the remaining accent colors**

```bash
grep -n "accent-green\|accent-yellow\|badge-ground\|badge-air\|badge-building" web/viewer.html
```

Keep `--accent-green` and `--accent-yellow` **only** where they carry meaning
(the Agent Brain gauge's positive/negative arc, the draw badge). Change the
entity type badges to greyscale: `--badge-ground: #C9C9CE`,
`--badge-air: #8E8E93`, `--badge-building: #6E6E73`.

- [ ] **Step 4: Verify no purple remains**

```bash
grep -ni "purple\|c471ed" web/viewer.html
```

Expected: no output.

- [ ] **Step 5: Load and screenshot**

Load `web/fixtures/modern.json`. Confirm the chrome is greyscale, the arena is
green, towers and units still read blue/red, and the console is empty.

- [ ] **Step 6: Commit**

```bash
git add web/viewer.html
git commit -m "feat: greyscale dark theme, removing all purple accents"
```

---

### Task 5: Right panel rebuild

**Files:**
- Modify: `web/viewer.html:540-585` (sidebar markup)
- Modify: `web/viewer.html` sidebar CSS block
- Modify: `web/viewer.html:1386-1410` (`updateHandPanel`)

**Interfaces:**
- Consumes: `gameData.cardNames`, `tickData.aiHand` / `oppHand`,
  `tickData.elixirAI` / `elixirOpp`.
- Produces: `renderHandTile(container, handIds, elixir)` — renders card tiles
  and dims those costing more than `elixir`.

- [ ] **Step 1: Rewrite the elixir block markup**

Replace the `<!-- Elixir -->` panel with a compact two-row block:

```html
    <div class="panel">
      <div class="panel-title">Match State</div>
      <div class="stat-row">
        <span class="stat-key" style="color:var(--blue-team)">BLUE</span>
        <div class="meter"><div class="meter-fill" id="elixirBarAI"></div></div>
        <span class="stat-num" id="elixirValAI">5.0</span>
      </div>
      <div class="stat-row">
        <span class="stat-key" style="color:var(--red-team)">RED</span>
        <div class="meter"><div class="meter-fill" id="elixirBarOpp"></div></div>
        <span class="stat-num" id="elixirValOpp">5.0</span>
      </div>
    </div>
```

- [ ] **Step 2: Add the shared 8px-scale panel CSS**

```css
  .panel { background: var(--bg-panel, var(--bg-secondary)); border: 1px solid var(--border);
           border-radius: 10px; padding: 12px; }
  .panel-title { font-size: 10px; font-weight: 600; text-transform: uppercase;
                 letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px; }
  .stat-row { display: grid; grid-template-columns: 40px 1fr 44px;
              align-items: center; gap: 8px; height: 24px; }
  .stat-key { font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
  .stat-num { font-size: 13px; font-weight: 600; text-align: right;
              font-variant-numeric: tabular-nums; }
  .meter { height: 6px; background: var(--bg-card); border-radius: 3px; overflow: hidden; }
  .meter-fill { height: 100%; background: var(--text-primary); border-radius: 3px;
                transition: width 0.12s linear; }
  .card-tile { display: flex; justify-content: space-between; align-items: center;
               gap: 6px; background: var(--bg-card); border: 1px solid var(--border);
               border-radius: 6px; padding: 5px 8px; font-size: 11px; }
  .card-tile.unaffordable { opacity: 0.38; }
  .card-tile .cost { font-weight: 700; font-variant-numeric: tabular-nums;
                     color: var(--text-secondary); }
  .hand-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
```

- [ ] **Step 3: Rewrite `updateHandPanel` to render real tiles**

`cardNames` values are formatted `"Hog Rider (4)"`, so the cost is parseable
from the name string and no new data is needed:

```javascript
function updateHandPanel(tickData) {
  const cardNames = gameData.cardNames || {};

  function renderHandTile(container, handIds, elixir) {
    container.innerHTML = '';
    if (!handIds || handIds.length === 0) {
      container.innerHTML = '<span class="empty-note">No data</span>';
      return;
    }
    for (const id of handIds) {
      const raw = cardNames[id] || `Card #${id}`;
      const m = /^(.*)\s\((\d+)\)$/.exec(raw);
      const name = m ? m[1] : raw;
      const cost = m ? parseInt(m[2], 10) : null;
      const tile = document.createElement('div');
      tile.className = 'card-tile'
        + (cost !== null && elixir !== null && cost > elixir ? ' unaffordable' : '');
      tile.title = raw;
      tile.innerHTML = `<span class="nm">${name}</span>`
        + (cost !== null ? `<span class="cost">${cost}</span>` : '');
      container.appendChild(tile);
    }
  }

  renderHandTile(document.getElementById('handCardsAI'),
                 tickData.aiHand, tickData.elixirAI ?? null);
  renderHandTile(document.getElementById('handCardsOpp'),
                 tickData.oppHand, tickData.elixirOpp ?? null);
}
```

Add `class="hand-grid"` to both `handCardsAI` and `handCardsOpp` containers.

- [ ] **Step 4: Convert the tower grid to slim meters**

In `updateTowerGrid`'s cell template, replace the existing bar markup with the
`.meter` / `.meter-fill` classes, tinting the fill with the team color:

```javascript
  bar.className = 'meter';
  fill.className = 'meter-fill';
  fill.style.background = def.team === 0 ? 'var(--blue-team)' : 'var(--red-team)';
  fill.style.width = (100 * Math.max(0, hp) / def.maxHp) + '%';
```

- [ ] **Step 5: Verify**

Load `web/fixtures/modern.json`. Confirm: hand tiles show full names and costs
(no `Musket...` truncation), unaffordable cards dim as elixir drops while
scrubbing, tower bars track HP, console empty.

- [ ] **Step 6: Verify against the legacy fixture**

Load `web/fixtures/legacy.json`. Hands and towers must still render (that file
keeps `cardNames`, `aiHand`, `oppHand`). Console must stay empty.

- [ ] **Step 7: Commit**

```bash
git add web/viewer.html
git commit -m "feat: rebuild the right panel on a consistent scale with real card tiles"
```

---

### Task 6: Entity inspector — kill every bare `?`

**Files:**
- Modify: `web/viewer.html:664-690` (`CARD_META`, `getEntityMeta`)
- Modify: `web/viewer.html:1342-1368` (`updateInspector`)
- Modify: `web/viewer.html:1519-1524` (hover tooltip)
- Modify: `web/viewer.html` `initViewer` (build the symbol index)

**Interfaces:**
- Consumes: `gameData.cardMeta` (may be absent — legacy fixture).
- Produces:
  - `buildSymbolIndex(cardMeta) -> {bySymbol: {sym: meta}, ambiguous: Set<sym>}`
  - `formatMaxHp(meta) -> string`
  - `getEntityMeta(ent) -> {name, maxHp, isBuilding, isSpell, isFlying, inferred}`

- [ ] **Step 1: Write the failing assertions**

These run in the browser console against a loaded page. Write them down now as
the acceptance check for this task:

```javascript
const T = window.__viewerTests;
// 1. A card whose registry symbol is literally '?' must not be named '?'.
console.assert(!/^\?+$/.test(T.getEntityMeta({id: -1, symbol: '?', hp: 100}).name),
  'FAIL: bare ? rendered as a name');
// 2. maxHp 0 (every spell) must not render '?'.
console.assert(T.formatMaxHp({maxHp: 0}) === '—', 'FAIL: maxHp 0 rendered as ?');
// 3. A cardId:-1 entity must still resolve by symbol from the replay's cardMeta.
console.assert(T.getEntityMeta({id: -1, symbol: 'H', hp: 100, cardId: -1}).name === 'Hog Rider',
  'FAIL: symbol index did not resolve a cardId:-1 entity');
```

- [ ] **Step 2: Run them and watch them fail**

Load `web/fixtures/modern.json`, paste the block. Expected: assertion 2 and 3
fail (`formatMaxHp` is not defined yet; the symbol index does not exist).

- [ ] **Step 3: Add the symbol index**

`cardMeta` already carries a `symbol` per card, generated live from
`CardRegistry`, so a symbol lookup can be built from the replay itself instead
of the hardcoded table that covers only ~45 of 173 cards. Add next to
`CARD_META`:

```javascript
// Symbol -> meta, derived from the replay's OWN cardMeta block. This is the
// fallback for entities carrying cardId -1 (Entity.h's default -- death-spawns,
// spawner buildings and tower troops never get one assigned), which the
// id-keyed lookup cannot resolve.
//
// Symbols genuinely collide: 'M' is both Mini PEKKA (1390 hp) and Monk (2214).
// For a cardId -1 entity the collision is not resolvable, so the ambiguous set
// is recorded and such a resolution is flagged `inferred` rather than silently
// picking whichever card happened to load last.
let SYMBOL_INDEX = { bySymbol: {}, ambiguous: new Set() };

function buildSymbolIndex(cardMeta) {
  const bySymbol = {}, ambiguous = new Set();
  for (const id of Object.keys(cardMeta || {})) {
    const meta = cardMeta[id];
    if (!meta || !meta.symbol) continue;
    if (bySymbol[meta.symbol] && bySymbol[meta.symbol].name !== meta.name) {
      ambiguous.add(meta.symbol);
    } else if (!bySymbol[meta.symbol]) {
      bySymbol[meta.symbol] = meta;
    }
  }
  return { bySymbol, ambiguous };
}
```

In `initViewer`, immediately after `CARD_META = gameData.cardMeta || {};` add:

```javascript
  SYMBOL_INDEX = buildSymbolIndex(CARD_META);
```

- [ ] **Step 4: Insert the index into `getEntityMeta`'s chain and kill the bare symbol**

```javascript
function getEntityMeta(ent) {
  const towerDef = TOWER_DEFS.find(td => td.id === ent.id);
  if (towerDef) return { name: towerDef.name, maxHp: towerDef.maxHp,
                         isBuilding: true, isSpell: false, isFlying: false,
                         inferred: false };
  if (ent.cardId !== undefined && CARD_META[ent.cardId]) {
    return Object.assign({ inferred: false }, CARD_META[ent.cardId]);
  }
  // cardId missing or -1: resolve by symbol against the replay's own registry
  // data before falling back to the legacy hardcoded tables.
  const fromSym = SYMBOL_INDEX.bySymbol[ent.symbol];
  if (fromSym) {
    return Object.assign({}, fromSym,
      { inferred: true, ambiguous: SYMBOL_INDEX.ambiguous.has(ent.symbol) });
  }
  const legacyName = SYMBOL_NAMES[ent.symbol];
  return {
    // NEVER the bare symbol: CardDefinition's default symbol is '?' and two
    // registry cards ship it explicitly (Cannon Cart 69, Guards 76), so the
    // old `|| ent.symbol` fallback rendered a literal '?' as a card name.
    name: legacyName || `Unknown (symbol '${ent.symbol}')`,
    maxHp: MAX_HP[ent.symbol] || 0,
    isBuilding: BUILDING_SYMBOLS.has(ent.symbol),
    isSpell: SPELL_SYMBOLS.has(ent.symbol),
    isFlying: FLYING_SYMBOLS_FALLBACK.has(ent.symbol),
    inferred: true,
  };
}

// maxHp 0 is every spell (no persistent HP to bar-render) and any card the
// legacy table never knew. Render an em dash, never '?'.
function formatMaxHp(meta) {
  return (meta && meta.maxHp > 0) ? String(meta.maxHp) : '—';
}
```

Note the legacy branch now returns `maxHp: 0` rather than `|| ent.hp`. The old
`|| ent.hp` made `hpRatio` permanently 1.0, so an unknown card's HP bar never
moved; `formatMaxHp` plus a guarded ratio is the correct handling.

- [ ] **Step 5: Guard the HP-bar ratio against maxHp 0**

Find where `hpRatio` is computed in `drawEntity` and make it:

```javascript
  const hpRatio = meta.maxHp > 0 ? Math.max(0, Math.min(1, ent.hp / meta.maxHp)) : 1;
```

- [ ] **Step 6: Use `formatMaxHp` in the inspector and add a provenance note**

In `updateInspector`, replace `const maxHp = meta.maxHp || '?';` with
`const maxHp = formatMaxHp(meta);` and append after the Symbol row:

```javascript
    + (meta.inferred ? `<div class="inspector-note">identified by symbol${
        meta.ambiguous ? ' — this symbol is shared by more than one card' : ''
      }</div>` : '')
```

Add the style:

```css
  .inspector-note { margin-top: 6px; font-size: 10px; color: var(--text-muted);
                    font-style: italic; }
```

- [ ] **Step 7: Extend the test surface and re-run the assertions**

Add `buildSymbolIndex` and `formatMaxHp` to `window.__viewerTests`. Reload
`web/fixtures/modern.json` and paste the Step 1 block.
Expected: all three assertions pass (no console output).

- [ ] **Step 8: Verify on the legacy fixture**

Load `web/fixtures/legacy.json` (no `cardMeta`, no `cardId`). Click several
entities. Expected: every one shows a real name or `Unknown (symbol 'x')`,
never a bare `?`; HP shows `n/n` or `n/—`; console empty.

- [ ] **Step 9: Commit**

```bash
git add web/viewer.html
git commit -m "fix: resolve entity names from the replay's own registry data, never a bare ?"
```

---

### Task 7: Teacher rollout capture (outside `python_ai/`) and its fixture

**Files:**
- Create: `tools/viewer_fixtures/teacher_debug_capture.py`
- Modify: `tools/viewer_fixtures/make_fixtures.py`
- Create: `web/fixtures/teacher.json` (generated)

**Interfaces:**
- Consumes: `python_ai.opponents.teacher.UtilityTeacher` (import only).
- Produces:
  - `CapturingTeacher(UtilityTeacher)` with `.debug_records` — a list of
    per-decision dicts matching the `teacherDebug` schema.
  - `attach_teacher_debug(replay_path, records, skip_frames)` — merges records
    into the replay JSON.

- [ ] **Step 1: Write the capturing subclass**

This reimplements `act()`'s rollout branch with capture. It lives outside
`python_ai/` because a training run is live; it is the code that later moves
into `python_ai/rl/teacher_debug.py`.

```python
"""Teacher rollout capture for the viewer's Simulation View.

Lives OUTSIDE python_ai/ on purpose: a training run is live and
python_ai/opponents/teacher.py must not be edited. Importing it is harmless.
When the run finishes this becomes python_ai/rl/teacher_debug.py.

COST: capturing a candidate's predicted board costs ~32 ms (a save_log file
round-trip; the cost is disk sync, not parsing) against a 0.31 ms bare rollout.
So only the chosen candidate and the next `top_k - 1` are captured, and the
whole thing is off unless explicitly constructed.
"""
import json
import os
import tempfile

from python_ai.opponents.teacher import UtilityTeacher, NOOP, HAND_SIZE
from python_ai.advisors import tactics


def _final_entities(env_snapshot, path):
    """Roll-out board as the engine's own logger sees it.

    ClashEnv::snapshot() copies everything EXCEPT the replay logger, which
    starts empty and stays live -- so save_log on a snapshot writes ONLY the
    rollout's own ticks, in the schema the viewer already renders.
    """
    env_snapshot.save_log(path)
    with open(path) as f:
        data = json.load(f)
    if not data.get("ticks"):
        return []
    return data["ticks"][-1]["entities"]


class CapturingTeacher(UtilityTeacher):
    """UtilityTeacher that records what it considered. Action-identical."""

    def __init__(self, *a, top_k=4, **kw):
        super().__init__(*a, **kw)
        self.top_k = int(top_k)
        self.debug_records = []
        self._tmp = os.path.join(tempfile.gettempdir(), f"td_{id(self)}.json")

    def _capture_board(self, env, cand):
        s = env.snapshot()
        self.execute_steps(s, cand, self.rollout_ticks())
        return _final_entities(s, self._tmp)

    def act(self, env, obs_own):
        # Delegate to the parent for the ACTION, so the chosen play can never
        # diverge. Capture is reconstructed afterwards from state the parent
        # does not consume.
        pre_pending = self.pending
        pre_ticks = self.pending_ticks
        action = super().act(env, obs_own)

        rec = {"team": self.team, "kind": "rollout",
               "horizonTicks": int(self.rollout_ticks()),
               "chosenIndex": -1, "baseline": None, "candidates": []}

        if self.horizon_ticks <= 0:
            rec["kind"] = "rules_only"
            self.debug_records.append(rec)
            return action

        # Re-derive the candidate set against the SAME pending state the parent
        # saw, so the recorded list matches what it actually ranked.
        saved = (self.pending, self.pending_ticks)
        self.pending, self.pending_ticks = pre_pending, pre_ticks
        try:
            cands = [c for c in self.candidates(env, obs_own) if c.steps]
            if not cands:
                rec["kind"] = "no_candidates"
                self.debug_records.append(rec)
                return action
            baseline = self.rollout_stats(env, NOOP)
            elixir_now = tactics.own_elixir(obs_own)
            scored = []
            for c in cands:
                sc = self.score(env, c, baseline, obs_own)
                scored.append((c, float(sc), float(self.margin_for(c, elixir_now))))
            scored.sort(key=lambda t: -t[1])

            rec["baseline"] = {"entities": self._capture_board(env, NOOP)}
            for i, (c, sc, mg) in enumerate(scored):
                row = {
                    "steps": [{"cardId": int(s.card_id), "slot": int(s.slot),
                               "x": float(s.x), "y": float(s.y),
                               "delayTicks": int(s.delay_ticks)} for s in c.steps],
                    "kind": c.kind,
                    "score": sc,
                    "margin": mg,
                    "rejected": bool(sc <= mg),
                }
                if i < self.top_k:
                    row["final"] = {"entities": self._capture_board(env, c)}
                rec["candidates"].append(row)

            chosen = [i for i, (c, sc, mg) in enumerate(scored)
                      if sc > mg and c.slot == action[0]
                      and abs(c.x - action[1]) < 1e-6 and abs(c.y - action[2]) < 1e-6]
            rec["chosenIndex"] = chosen[0] if chosen else -1

            # A decision that played nothing is a legitimate outcome of the
            # rollout branch (every candidate fell below its margin), and it is
            # NOT the same as the exploration branch firing. Distinguishing
            # them matters: the candidate list is meaningful in the first case
            # and misleading in the second, where the parent ignored the
            # ranking entirely and picked uniformly.
            rec["held"] = bool(action[0] == HAND_SIZE)
            if rec["chosenIndex"] == -1 and not rec["held"] and self.epsilon > 0.0:
                rec["kind"] = "epsilon"
        finally:
            self.pending, self.pending_ticks = saved

        self.debug_records.append(rec)
        return action


def attach_teacher_debug(replay_path, records, skip_frames=10):
    """Merge per-decision records into an already-saved replay, one
    skip_frames-wide window each -- the same shape
    annotate_replay_with_agent_info uses for the agent's own internals."""
    with open(replay_path) as f:
        data = json.load(f)
    for i, tick in enumerate(data["ticks"]):
        j = i // skip_frames
        if j < len(records):
            tick["teacherDebug"] = records[j]
    with open(replay_path, "w") as f:
        json.dump(data, f)
```

- [ ] **Step 2: Verify action identity against a pristine teacher**

This is the invariant the whole approach rests on. Write
`tools/viewer_fixtures/check_action_identity.py`:

```python
"""CapturingTeacher must choose exactly what UtilityTeacher would choose."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import python_ai  # noqa: F401
import numpy as np
import clash_royale_env as E
from python_ai.opponents.teacher import UtilityTeacher
from teacher_debug_capture import CapturingTeacher

DECK = [15, 6, 25, 40, 24, 72, 33, 7]

def run(make_t1, seed=23, n=120):
    env = E.ClashRoyaleEnv(DECK, DECK, 3600); env.seed(seed); env.reset()
    t0 = UtilityTeacher(DECK, team=0); t0.set_stage(5); t0.reset()
    t1 = make_t1(); t1.set_stage(5); t1.reset()
    acts = []
    for _ in range(n):
        if env.is_game_over(): break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0); a1 = t1.act(env, o1)
        acts.append(a1)
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    return acts

plain = run(lambda: UtilityTeacher(DECK, team=1))
capt  = run(lambda: CapturingTeacher(DECK, team=1, top_k=4))
print(f"decisions compared: {len(plain)}")
mismatch = [i for i, (a, b) in enumerate(zip(plain, capt)) if a != b]
print(f"mismatches: {len(mismatch)}")
assert len(plain) == len(capt), "different decision counts"
assert not mismatch, f"ACTION DIVERGENCE at {mismatch[:10]}"
print("PASS: capture is action-identical")
```

Run it:

```bash
cd tools/viewer_fixtures && ../../python_ai/venv/Scripts/python.exe check_action_identity.py
```

Expected: `PASS: capture is action-identical`. **If this fails, stop** — a
capture hook that changes the teacher's play is not usable.

- [ ] **Step 3: Generate the teacher fixture**

Add to `make_fixtures.py`:

```python
def play_match_with_debug(path, seed=17, decisions=220, top_k=4):
    import numpy as np
    from python_ai.opponents.teacher import UtilityTeacher
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher_debug_capture import CapturingTeacher, attach_teacher_debug

    env = E.ClashRoyaleEnv(DECK, DECK, 3600); env.seed(seed); env.reset()
    t0 = UtilityTeacher(DECK, team=0); t0.set_stage(5); t0.reset()
    t1 = CapturingTeacher(DECK, team=1, top_k=top_k); t1.set_stage(5); t1.reset()
    for _ in range(decisions):
        if env.is_game_over(): break
        o0 = np.asarray(env.get_observation_for_team(0), np.float32)
        o1 = np.asarray(env.get_observation_for_team(1), np.float32)
        a0 = t0.act(env, o0); a1 = t1.act(env, o1)
        env.step_self_play(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], 10)
    env.save_log(path)
    attach_teacher_debug(path, t1.debug_records, skip_frames=10)
    return path
```

and call it from `main()` for `web/fixtures/teacher.json`.

- [ ] **Step 4: Run and inspect the shape**

```bash
python_ai/venv/Scripts/python.exe tools/viewer_fixtures/make_fixtures.py
python_ai/venv/Scripts/python.exe -c "
import json, collections
d=json.load(open('web/fixtures/teacher.json'))
k=collections.Counter(t['teacherDebug']['kind'] for t in d['ticks'] if 'teacherDebug' in t)
print('kinds:', dict(k))
r=[t['teacherDebug'] for t in d['ticks'] if t.get('teacherDebug',{}).get('kind')=='rollout']
print('rollout decisions:', len(r))
if r:
    c=r[0]['candidates']
    print('cands in first:', len(c), 'with boards:', sum(1 for x in c if 'final' in x))
    print('has baseline:', r[0]['baseline'] is not None)
print('size MB: %.1f' % (len(json.dumps(d))/1e6))
"
```

Expected: `kinds` includes `rollout` and `no_candidates`; at most 4 candidates
carry `final`; `baseline` is present; size is a few MB.

- [ ] **Step 5: Commit**

```bash
git add tools/viewer_fixtures/ web/fixtures/teacher.json
git commit -m "feat: teacher rollout capture (staged outside python_ai) and its fixture"
```

---

### Task 8: Arena candidate rings and the toggle chip

**Files:**
- Modify: `web/viewer.html` `drawBoard` (add the overlay), `initViewer`,
  controls markup.

**Interfaces:**
- Consumes: `tickData.teacherDebug` (may be absent).
- Produces:
  - `teacherDebugForTick(tickData) -> record|null`
  - `hasTeacherDebug(gameData) -> boolean`
  - `drawCandidateRings(rc, record)`

- [ ] **Step 1: Add the accessors, written to tolerate absence**

```javascript
// Returns the teacher's decision record for a tick, or null. EVERY consumer
// must handle null: replays generated without --teacher-debug have none, and
// even in a captured replay a tick can legitimately carry no decision.
function teacherDebugForTick(tickData) {
  const td = tickData && tickData.teacherDebug;
  if (!td || typeof td !== 'object') return null;
  if (!Array.isArray(td.candidates)) td.candidates = [];
  return td;
}

function hasTeacherDebug(data) {
  return !!(data && Array.isArray(data.ticks)
    && data.ticks.some(t => t && t.teacherDebug));
}
```

- [ ] **Step 2: Draw the rings**

```javascript
// Deliberately minimal: a hollow numbered ring per candidate cell, the chosen
// one filled. At most `top_k` of them. Anything more on the arena itself is
// the clutter the Simulation View exists to avoid.
function drawCandidateRings(rc, record) {
  if (!record || !record.candidates.length) return;
  const shown = record.candidates.slice(0, 4);
  shown.forEach((cand, i) => {
    const step = cand.steps && cand.steps[0];
    if (!step) return;
    const p = gameToCanvas(rc, step.x, step.y);
    const chosen = (i === record.chosenIndex);
    const r = rc.cellSize * 0.62;
    rc.ctx.beginPath();
    rc.ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    rc.ctx.lineWidth = 2;
    rc.ctx.strokeStyle = chosen ? '#FFFFFF' : 'rgba(255,255,255,0.55)';
    if (chosen) { rc.ctx.fillStyle = 'rgba(255,255,255,0.22)'; rc.ctx.fill(); }
    rc.ctx.stroke();
    rc.ctx.fillStyle = chosen ? '#FFFFFF' : 'rgba(255,255,255,0.7)';
    rc.ctx.font = `600 ${Math.round(rc.cellSize * 0.5)}px Inter, sans-serif`;
    rc.ctx.textAlign = 'center'; rc.ctx.textBaseline = 'middle';
    rc.ctx.fillText(String(i + 1), p.x, p.y);
  });
  rc.ctx.textBaseline = 'alphabetic';
}
```

Call it at the end of `drawBoard`, guarded so mini-boards never draw rings:

```javascript
  if (!opts.ghost) {
    drawCandidateRings(rc, teacherDebugForTick(tickData));
  }
```

- [ ] **Step 3: Add the chip to the controls bar**

```html
  <button id="btnSimView" class="sim-chip" style="display:none" title="Simulation View (T)">
    🧠 <span id="simChipCount">0</span> candidates
  </button>
```

```css
  .sim-chip { font-size: 11px; }
  .sim-chip[disabled] { opacity: 0.4; cursor: default; }
```

- [ ] **Step 4: Show the chip only when the file has data**

In `initViewer`, after `CARD_META` is set:

```javascript
  // The button does not exist for a replay with no teacher data -- an inert
  // control is worse than an absent one.
  document.getElementById('btnSimView').style.display =
    hasTeacherDebug(gameData) ? 'flex' : 'none';
```

And in the per-tick update path:

```javascript
  const td = teacherDebugForTick(gameData.ticks[currentTick]);
  const chip = document.getElementById('btnSimView');
  if (chip.style.display !== 'none') {
    document.getElementById('simChipCount').textContent =
      td ? td.candidates.length : 0;
    chip.disabled = !td;
  }
```

- [ ] **Step 5: Verify all three fixtures**

- `web/fixtures/teacher.json` → chip visible, count changes while scrubbing,
  rings appear on the arena. Console empty.
- `web/fixtures/modern.json` → **chip absent entirely**, no rings, console empty.
- `web/fixtures/legacy.json` → same as modern. Console empty.

- [ ] **Step 6: Commit**

```bash
git add web/viewer.html
git commit -m "feat: candidate rings on the arena and a teacher-debug chip that hides itself"
```

---

### Task 9: The slide-over Simulation View

**Files:**
- Modify: `web/viewer.html` — new panel markup, CSS, and render function.

**Interfaces:**
- Consumes: `teacherDebugForTick`, `drawBoard`, `makeRenderCtx`,
  `getEntityMeta`, `formatMaxHp`.
- Produces: `renderSimView(record)`, `toggleSimView(force)`.

- [ ] **Step 1: Add the panel markup**

Insert as the last child of `.main-layout`:

```html
  <aside class="sim-panel" id="simPanel" aria-hidden="true">
    <div class="sim-head">
      <span class="panel-title" style="margin:0">Simulation View</span>
      <button class="sim-close" id="simClose" title="Close (T)">✕</button>
    </div>
    <div class="sim-body" id="simBody"></div>
  </aside>
```

- [ ] **Step 2: Add the CSS — the arena shrinks, it is not covered**

```css
  .sim-panel { width: 0; flex-shrink: 0; overflow: hidden; background: var(--bg-secondary);
               border-left: 1px solid var(--border); display: flex; flex-direction: column;
               transition: width 0.18s ease; }
  .sim-panel.open { width: 380px; }
  .sim-head { display: flex; justify-content: space-between; align-items: center;
              padding: 10px 12px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .sim-close { background: none; border: none; color: var(--text-muted); cursor: pointer;
               font-size: 13px; }
  .sim-body { overflow-y: auto; padding: 10px 12px; display: flex;
              flex-direction: column; gap: 10px; }
  .sim-baseline { font-size: 11px; color: var(--text-secondary);
                  border: 1px dashed var(--border); border-radius: 8px; padding: 8px; }
  .sim-row { border: 1px solid var(--border); border-radius: 8px; padding: 8px;
             background: var(--bg-card); border-left: 3px solid transparent; }
  .sim-row.chosen { border-left-color: var(--text-primary); background: var(--bg-hover); }
  .sim-row.rejected { opacity: 0.6; }
  .sim-row-head { display: flex; justify-content: space-between; align-items: baseline;
                  font-size: 11px; margin-bottom: 6px; gap: 8px; }
  .sim-score { font-weight: 700; font-variant-numeric: tabular-nums; }
  .sim-boards { display: flex; align-items: center; gap: 6px; }
  .sim-boards canvas { border-radius: 4px; flex: 1; min-width: 0; }
  .sim-arrow { color: var(--text-muted); font-size: 14px; flex-shrink: 0; }
  .sim-strip { display: flex; justify-content: space-between; font-size: 11px;
               color: var(--text-secondary); }
  .sim-note { font-size: 11px; color: var(--text-muted); font-style: italic;
              text-align: center; padding: 16px 8px; }
```

- [ ] **Step 3: Write `renderSimView`, covering every degradation case**

```javascript
const SIM_MINI_CELL = 5;   // px per tile in a mini-board

function cardLabel(step) {
  const raw = (gameData.cardNames || {})[step.cardId];
  const nm = raw ? raw.replace(/\s\(\d+\)$/, '') : `Card #${step.cardId}`;
  return `${nm} → (${step.x}, ${step.y})`;
}

function miniBoard(entities) {
  const cv = document.createElement('canvas');
  const rc = makeRenderCtx(cv.getContext('2d'), gameData.boardWidth,
                           gameData.boardHeight, SIM_MINI_CELL, 2);
  cv.width = rc.width; cv.height = rc.height;
  drawBoard(rc, { entities: entities || [] },
            { selectedEntityId: null, showLabels: false, ghost: true });
  return cv;
}

function renderSimView(record) {
  const body = document.getElementById('simBody');
  body.innerHTML = '';

  if (!record) {
    body.innerHTML = '<div class="sim-note">No teacher decision on this tick.</div>';
    return;
  }
  if (record.kind === 'no_candidates') {
    body.innerHTML = '<div class="sim-note">Nothing affordable this decision.</div>';
    return;
  }
  if (record.kind === 'rules_only' || record.kind === 'epsilon') {
    body.innerHTML = `<div class="sim-note">Decided by the ${
      record.kind === 'epsilon' ? 'exploration (epsilon) branch'
                                : 'rules-only branch (no lookahead at this stage)'
    } — no rollout scores exist for this decision.</div>`;
    return;
  }

  const base = document.createElement('div');
  base.className = 'sim-baseline';
  base.innerHTML = `Hold (no-op) = <strong>0.00</strong> by definition — every score below is marginal against it.`
    + (record.held ? `<br><strong>The teacher held.</strong> Nothing below beat its margin.` : '');
  if (record.baseline && record.baseline.entities) {
    base.appendChild(miniBoard(record.baseline.entities));
  }
  body.appendChild(base);

  const secs = (record.horizonTicks || 0) / 10;
  record.candidates.forEach((cand, i) => {
    const row = document.createElement('div');
    row.className = 'sim-row'
      + (i === record.chosenIndex ? ' chosen' : '')
      + (cand.rejected ? ' rejected' : '');

    const head = document.createElement('div');
    head.className = 'sim-row-head';
    const label = (cand.steps || []).map(cardLabel).join('  ▸  ') || 'Hold';
    head.innerHTML = `<span>#${i + 1} ${label}</span>`
      + `<span class="sim-score">${cand.score >= 0 ? '+' : ''}${cand.score.toFixed(2)}</span>`;
    row.appendChild(head);

    if (cand.final && cand.final.entities) {
      const wrap = document.createElement('div');
      wrap.className = 'sim-boards';
      wrap.appendChild(miniBoard(gameData.ticks[currentTick].entities));
      const arrow = document.createElement('span');
      arrow.className = 'sim-arrow';
      arrow.textContent = '→';
      arrow.title = `+${secs}s`;
      wrap.appendChild(arrow);
      wrap.appendChild(miniBoard(cand.final.entities));
      row.appendChild(wrap);
    } else {
      const strip = document.createElement('div');
      strip.className = 'sim-strip';
      strip.innerHTML = `<span>not simulated</span>`;
      row.appendChild(strip);
    }

    if (cand.rejected) {
      const note = document.createElement('div');
      note.className = 'sim-strip';
      note.style.marginTop = '4px';
      note.innerHTML = `<span>✗ below margin ${cand.margin.toFixed(2)}</span>`;
      row.appendChild(note);
    } else if (i === record.chosenIndex) {
      const note = document.createElement('div');
      note.className = 'sim-strip';
      note.style.marginTop = '4px';
      note.innerHTML = `<span>✓ chosen</span>`;
      row.appendChild(note);
    }
    body.appendChild(row);
  });
}
```

- [ ] **Step 4: Wire the toggle**

```javascript
let simViewOpen = false;
function toggleSimView(force) {
  simViewOpen = (force === undefined) ? !simViewOpen : !!force;
  const p = document.getElementById('simPanel');
  p.classList.toggle('open', simViewOpen);
  p.setAttribute('aria-hidden', String(!simViewOpen));
  if (simViewOpen) renderSimView(teacherDebugForTick(gameData.ticks[currentTick]));
  // No canvas resize call is needed: .canvas-wrapper and canvas already carry
  // max-width:100% / object-fit:contain, so the arena CSS-scales down when the
  // panel takes width. The backing store is unchanged and the hover handler
  // already converts through getBoundingClientRect, so hit-testing still works.
}
document.getElementById('btnSimView').addEventListener('click', () => toggleSimView());
document.getElementById('simClose').addEventListener('click', () => toggleSimView(false));
```

Add `T` to the existing keyboard handler:

```javascript
  if (e.key === 't' || e.key === 'T') {
    if (hasTeacherDebug(gameData)) toggleSimView();
  }
```

And call `renderSimView(...)` from the per-tick update path when `simViewOpen`.

- [ ] **Step 5: Verify the four states**

Load `web/fixtures/teacher.json`, press `T`:
1. a `rollout` tick shows the baseline, ranked rows, boards on the top 4,
   score-only strips below, the chosen row accented;
2. scrub to a `no_candidates` tick → "Nothing affordable this decision.";
3. scrub to a tick with no record → "No teacher decision on this tick.";
4. the **arena shrinks** rather than being covered, and mini-boards show the
   real palette from Task 3.

Console must be empty throughout.

- [ ] **Step 6: Commit**

```bash
git add web/viewer.html
git commit -m "feat: slide-over Simulation View for the teacher's candidate rollouts"
```

---

### Task 10: Graceful-degradation sweep

The spec calls this critical, so it gets its own verification pass rather than
riding along on the feature tasks.

**Files:**
- Create: `web/fixtures/mixed.json` (generated)
- Modify: `tools/viewer_fixtures/make_fixtures.py`

- [ ] **Step 1: Generate a deliberately hostile fixture**

Add to `make_fixtures.py` and call from `main()`:

```python
def make_mixed(src, dst):
    """teacher.json with damage applied: some ticks lose teacherDebug, one
    record loses its candidates, one loses its baseline, one candidate loses
    its board, one loses its steps. Every one of these must render."""
    with open(src) as f:
        data = json.load(f)
    seen = 0
    for i, tick in enumerate(data["ticks"]):
        td = tick.get("teacherDebug")
        if not td:
            continue
        if i % 3 == 0:
            del tick["teacherDebug"]; continue
        seen += 1
        if seen == 1:
            td["candidates"] = []
        elif seen == 2:
            td["baseline"] = None
        elif seen == 3 and td.get("candidates"):
            td["candidates"][0].pop("final", None)
        elif seen == 4 and td.get("candidates"):
            td["candidates"][0]["steps"] = []
        elif seen == 5:
            td["kind"] = "rules_only"
        elif seen == 6:
            td["kind"] = "epsilon"
    with open(dst, "w") as f:
        json.dump(data, f)
```

- [ ] **Step 2: Run the sweep across all four fixtures**

For each of `modern.json`, `legacy.json`, `teacher.json`, `mixed.json`:
open the viewer, load the file, open the Simulation View if the chip exists,
and play through the whole replay at 8x.

Record for each: console errors (must be **0**), and whether any empty framed
region appears (must be **none** — a missing board renders a text strip, never
an empty bordered box).

- [ ] **Step 3: Confirm the entity assertions on every fixture**

Paste the Task 6 Step 1 assertion block on each. Expected: silent on all four.

- [ ] **Step 4: Commit**

```bash
git add tools/viewer_fixtures/make_fixtures.py web/fixtures/mixed.json
git commit -m "test: hostile fixture and the graceful-degradation sweep"
```

---

### Task 11: The upstream request for `cardId` on spawned entities

**Files:**
- Modify: `perception/UPSTREAM_REQUESTS.md`

- [ ] **Step 1: Read the file's existing item format**

```bash
grep -n "^## \|^### " perception/UPSTREAM_REQUESTS.md | tail -20
```

Match the numbering and the heading style already in use.

- [ ] **Step 2: Append the item**

The file's latest item is 19, so this is 20. Append verbatim:

```markdown
## 20. OPEN — spawned entities carry no cardId, so a replay cannot name them

`Entity.h:45` declares `int cardId = -1`. `CardFactories::applyCardMetadata`
assigns a real id for a card played from hand, but nothing does for an entity
spawned by another entity -- death-spawns (`SpawnOnDeath`,
`SpawnOnDeathForEnemyTeam`), spawner buildings (`PeriodicSpawnEffect`,
`ProximityGatedPeriodicSpawnEffect`, `CappedSpawnOnHitEffect`) and tower troops
(`TowerTroops.h`). They keep the -1.

### Consequence

`GameLogger` writes `cardId` per entity and an id-keyed `cardMeta` block, so a
replay is self-describing for every card played from hand and for nothing else.
`web/viewer.html` cannot resolve a -1 entity by id and falls back to a
symbol-keyed table that covers ~45 of the 173 registry entries. Where that
misses, the last resort was the raw symbol character -- and `'?'` is
`CardDefinition`'s default symbol (`CardRegistry.h:100`) as well as the
explicit symbol of Cannon Cart (id 69) and Guards (id 76). A card could
therefore be displayed as a literal `?`.

### Not gameplay-affecting

`cardId` is read by stats attribution (`StatsEvents`, `MatchStatistics`) and by
the logger. Nothing in `include/entities/` branches on it to decide movement,
targeting or damage, so propagating it cannot change a simulation outcome.
It WOULD change per-card stats attribution: a Skeleton spawned by a Tombstone
would begin attributing its damage to Tombstone rather than to nothing, which
`get_elixir_value_killed_by` and `get_damage_dealt_by_card` both read -- and
those feed reward shaping. That makes it reward-affecting even though it is not
physics-affecting, which is the reason to decide it deliberately.

### Options

1. **Propagate the parent's `cardId`** to spawned entities. Simplest; makes a
   Tombstone Skeleton report as "Skeleton" via the parent's registry entry only
   if the spawn's own card id is used, so in practice this means passing the
   SPAWNED card's id where the spawn effect knows it.
2. **Add a separate `spawnedByCardId`** and leave `cardId` alone. No change to
   stats attribution; the logger gains one field and the viewer one fallback.
   Safest with respect to the reward path.
3. **Do nothing.** The viewer now resolves these by symbol against the replay's
   own `cardMeta` block and never renders a bare `?`, which is what shipped on
   2026-08-21. Residual defect: symbols genuinely collide (`'M'` is Mini PEKKA
   at 1390 hp and Monk at 2214), so a -1 entity on a shared symbol may be shown
   under the wrong name and HP maximum. The viewer flags such a resolution as
   inferred rather than presenting it as certain.

Option 2 is the one that fixes the display without touching the reward path.
```

- [ ] **Step 3: Commit**

```bash
git add perception/UPSTREAM_REQUESTS.md
git commit -m "docs: upstream request for cardId on spawned entities"
```

---

## Deferred until the training run ends

Not part of this plan's execution. When the run finishes:

1. Move `tools/viewer_fixtures/teacher_debug_capture.py` to
   `python_ai/rl/teacher_debug.py`, adding the four-line script bootstrap that
   `test_package_layout.py` requires.
2. Fold `CapturingTeacher.act`'s capture into `UtilityTeacher.act` behind
   `enable_debug_capture(top_k=4)`, **default off**.
3. Add `--teacher-debug` to `python_ai/tools/make_replays.py`.
4. Port `check_action_identity.py` into `python_ai/tests/` as a real test case.
5. Run the Python suite:
   `python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q`
   Expected: 369 collected, 367 pass, 2 skipped (skip count varies with the
   unseeded opening-hand shuffle).
