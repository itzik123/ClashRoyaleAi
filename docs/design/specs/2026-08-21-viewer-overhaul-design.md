# Replay Viewer Overhaul — Design

**Date:** 2026-08-21
**Scope:** `web/viewer.html` aesthetic rebuild, entity-identity fix, and a new
Teacher Debug / simulation-rollout visualization.

---

## Constraint that shapes everything

**A training run is live.** `python_ai/opponents/teacher.py` is phase 1's
opponent, and CLAUDE.md records that editing `python_ai/` can kill a
multi-hour run — the running process holds its own copy, but `train.py`
`subprocess.Popen`s `train_selfplay.py`, and a spawned process would import
whatever is on disk.

**Therefore: no file under `python_ai/` is edited by this work.** The rollout
capture is developed as a subclass wrapper in a scratch module that overrides
`UtilityTeacher.act()`. Importing is harmless; only writing is not. That
wrapper generates *real* fixture replays from the *real* teacher, so the entire
viewer UI is built and validated against genuine data. Landing it in
`teacher.py` afterwards is a mechanical move, staged as a patch (§6).

---

## 1. Aesthetic & UI overhaul

### 1.1 Map palette

The reference frame shows one continuous lawn with a two-tone mown checker, a
bright cyan river, wood-plank bridges, and pale tan lane paths running from
each bridge to the towers.

| element | current | new |
|---|---|---|
| grass, light checker | `#223322` | `#93C756` |
| grass, dark checker | `#1a2e1a` | `#84BC46` |
| river | `#1e3a5f` | `#37BEDC` |
| bridge deck | `#8b6914` | `#C89B5C` |
| bridge plank seams | — | `#A87C46` |
| lane path | — | `#CFC08F` |
| arena surround | `#0f1117` | `#2E4A1C` deep forest |

**The red/blue territory wash is removed.** It is the single largest reason the
current viewer does not read as Clash Royale: the real arena is one green
field. Team identity comes from tower color and unit rings. Small edge labels
("BLUE TERRITORY") are kept at low contrast; the full-half tint is not.

The checker is drawn per tile from `(x + y) % 2`, matching the reference's
mown-stripe grid, and must be derived from the replay's own `boardWidth` /
`boardHeight` rather than hardcoded 18×34.

### 1.2 Theme: neutral greyscale dark mode

`--elixir-purple` is deleted, along with the header's purple gradient text.

| token | value |
|---|---|
| `--bg-primary` | `#0B0B0C` |
| `--bg-panel` | `#141416` |
| `--bg-card` | `#1C1C1F` |
| `--border` | `#26262A` |
| `--text-primary` | `#F5F5F5` |
| `--text-secondary` | `#A1A1A6` |
| `--text-muted` | `#6E6E73` |

**Team red/blue is retained, scoped strictly to team semantics** — towers, unit
rings, hand headers, elixir row labels. A two-player board rendered without
team color is unreadable. Everything that is not team identity becomes
greyscale, including the elixir bar fills (light grey, previously purple) and
all badges.

### 1.3 Right panel layout

Rebuilt on an 8px spacing scale with uppercase micro-labels and tabular
numerals:

- **Match state** — both elixir bars in one compact block, large numerals.
- **Hands** — real card tiles showing name and cost, dimmed when unaffordable
  at the current elixir. Currently they are truncated text chips (`Musket...`).
- **Tower health** — 2×3 grid of slim bars with absolute HP.
- **Entity inspector** — unchanged position, fixed contents (§2).
- **Event log** — unchanged.

The Agent Brain gauge on the left panel is **not** touched.

---

## 2. Entity inspector: the `?` bug

Three independent causes. Only the first is an engine defect.

### 2.1 `cardId = -1` on non-card-factory spawns — UPSTREAM, not fixed here

`Entity.h:45` defaults `cardId = -1`. Entities spawned outside
`CardFactories::applyCardMetadata` — death-spawns, spawner buildings, tower
troops — keep that default, so `cardMeta[cardId]` misses and the viewer falls
through to a symbol table covering ~45 of 173 registry entries.

The fix is propagating `cardId` to spawned entities in C++. Per CLAUDE.md that
is not a change to make unilaterally: it is written up in
`perception/UPSTREAM_REQUESTS.md` with evidence and blast radius, for the human
to decide.

### 2.2 Symbol index derived from the replay's own `cardMeta` — fixed here

Every replay already embeds `cardMeta`, generated live from `CardRegistry`,
and each entry carries its `symbol`. The viewer builds a **symbol → meta index
from that block at load time** and consults it before the hardcoded legacy
tables. A `cardId: -1` entity then resolves against live registry data instead
of a stale hand-maintained list.

Symbol collisions are real (`'M'` is both Mini PEKKA at 1390 HP and Monk at
2214). For a `cardId: -1` entity they are not resolvable, so the index records
the collision and the inspector marks such a resolution as inferred rather than
silently picking whichever card loaded last. This preserves the existing
`cardId`-first ordering, which is collision-free and stays the preferred path.

### 2.3 No bare `?` may ever render — fixed here

- Last-resort name becomes `Unknown (symbol 'x')`, never the raw character.
  Two registry cards ship `symbol = '?'` (Cannon Cart id 69, Guards id 76) and
  `'?'` is also `CardDefinition`'s default, so the raw-symbol fallback can
  render a literal `?` as a card name.
- `meta.maxHp || '?'` becomes `—` for `maxHp: 0`.
- Spells (`maxHp: 0` by construction) show `n/a` for HP rather than `? `.

**Open item:** the exact `?` reported could not be reproduced from the one
replay on disk (`replays/replay_ep1000.json`), where every entity resolves
correctly. The three fixes above are unconditional and cover each known path,
but the reported replay file should be checked against the result to confirm.

---

## 3. Teacher Debug: data

### 3.1 The capture mechanism (verified)

`ClashEnv::snapshot()` copies everything except the replay logger, which starts
empty and stays live. A candidate rollout on a snapshot therefore accumulates
**only its own ticks**, and `save_log()` yields a self-contained mini-replay in
the exact schema the viewer already renders — entity positions, HP, `cardId`,
`cardMeta`. Verified empirically; the parent match is unaffected.

### 3.2 Measured cost — why the top-K cap exists

| operation | cost |
|---|---|
| bare candidate rollout (today) | **0.31 ms** |
| + `save_log` round-trip → exact final board | **~32 ms** |
| `get_observation_for_team` (lossy alternative) | 2.4 ms |

The 32 ms is disk sync, not parsing; a tail-read optimization was tried and did
not help. Full-fidelity capture is therefore ~100× a bare rollout and **must
never run in the training loop.**

Candidates per decision, measured over 1,080 decisions across three contested
teacher-vs-teacher matches:

| statistic | value |
|---|---|
| mean / median | 3.60 / 3 |
| p90 / p99 / max | 9 / 12 / 13 |
| decisions with **zero** playable candidates | **22%** |
| decisions with **>4** candidates | **33%** |

The cap of 4 binds on a third of decisions. Its compute saving is modest
(~30%); its value is bounding the UI at four mini-boards. Total capture cost is
**~40 s per 360-decision replay**.

### 3.3 What is captured

`enable_debug_capture(top_k=4)`, **off by default**, zero cost when off. Inside
`act()`:

- **Every** scored candidate is recorded — cards, cells, score, the margin it
  had to beat, and whether the margin rejected it. Free: the scores are already
  computed. `score()` currently discards sub-margin candidates via `continue`,
  and "scored +2.1 but the margin was 3.0" is exactly the diagnostic wanted.
- **Chosen + next 3** are re-run with board capture, keeping only the final
  tick's entities.
- The **no-op baseline** board is captured too. `score()` is defined as
  marginal against it and returns exactly 0.0 for the no-op, so it is the
  honest comparator to render beside the candidates.
- The **decision path** is recorded: `rollout`, `rules_only` (stages 0–1, which
  never call a rollout and have no scores), `epsilon` (random action), or
  `no_candidates`.

### 3.4 Schema

Attached per decision-window tick by the existing Python annotation pass, the
same way `stateValue` already is:

```json
"teacherDebug": {
  "team": 1,
  "kind": "rollout",
  "horizonTicks": 100,
  "chosenIndex": 0,
  "baseline": { "entities": [ ... ] },
  "candidates": [
    {
      "steps": [ {"cardId": 15, "slot": 2, "x": 3, "y": 15, "delayTicks": 0} ],
      "kind": "single",
      "score": 4.21,
      "margin": 3.00,
      "rejected": false,
      "final": { "entities": [ ... ] }
    }
  ]
}
```

`final` is absent for candidates beyond `top_k`. Nothing else in the replay
changes, so the block is purely additive.

### 3.5 Placement

Its eventual home is a new `python_ai/rl/teacher_debug.py`, matching the
package's one-responsibility-per-module structure, opted into via
`make_replays.py --teacher-debug`. **Default off in the training loop**, so the
automatic every-1000-episodes replay costs exactly what it costs today.

**That file is not created during this work.** While the run is live the same
logic lives in a scratch subclass wrapper outside the package (see the
constraint section); §6 covers moving it in.

### 3.6 The invariant that must hold

**Capture ON must choose the identical action to capture OFF.** A debug hook
that perturbs the opponent's play would silently corrupt phase 1. Pinned by
test, following the precedent of the existing `max_combos = 0` action-identity
test.

The rollouts being visualized roll **both sides forward doing nothing** —
CLAUDE.md records this as a known scoring bias ("structurally cannot see the
counter-push"). The visualization does not introduce it; it makes an existing
limitation visible for the first time. Reactive rollouts are out of scope here
and would be a separate ~20×-cost change.

---

## 4. Teacher Debug: UX

Rejected: a **modal** (hides the match, and the point is comparing prediction
against what actually happened next) and a **live multi-board overlay** (that
is the clutter itself).

### 4.1 Always-on, minimal, on the arena

When the current tick carries data:

- a hollow numbered ring at each candidate's placement cell, the chosen one
  filled and accented — at most 4;
- a small `🧠 4 candidates` chip beside the tick counter.

Nothing else. This reads as annotation, not as a second UI.

### 4.2 The Simulation View — a slide-over, opened by `T` or the chip

Opens from the right; **the arena shrinks rather than being covered**, so the
real board and the predictions stay visible together. One ranked row per
candidate, chosen row accented with a left bar:

```
▎#1  Hog Rider  → (3,15)     [ NOW ] → [ +10s ]   +4.21   ✓ CHOSEN
  #2  Cannon    → (8,11)     [ NOW ] → [ +10s ]   +3.55
  #3  Skeletons → (7,14)     [ NOW ] → [ +10s ]   +1.02   ✗ below margin 3.00
  #4  Ice Spirit→ (9,13)     [ NOW ] → [ +10s ]   +0.44   ✗ below margin 3.00
  ─────────────────────────────────────────────────────────────────────
  #5  Fireball    (12,20)                         +0.31   · not simulated
  #6  The Log     (6,16)                          −0.10   · not simulated
```

With `top_k = 4`, rows #1–#4 carry boards and #5 onward are score-only. On the
67% of decisions with four or fewer candidates the divider never appears.

- Mini-boards reuse the **main arena's draw function** at reduced scale — one
  renderer, not two, so the palette work in §1 applies automatically.
- A header row shows the baseline: `Hold (no-op) = 0.00 by definition`, with
  the margin threshold drawn, so it is visible *why* a positively-scoring
  candidate was still declined.
- Rows past `top_k` collapse to a score-only strip. No empty board frames.

### 4.3 Graceful degradation — a hard requirement

| case | behavior |
|---|---|
| file has no `teacherDebug` anywhere | toggle button is not rendered at all; no chip, no rings, no console output |
| tick has none | chip and rings absent; open panel reads "no decision this tick" |
| `kind: "no_candidates"` (**22% of decisions**) | "Nothing affordable this decision" — not an empty list |
| `kind: "rules_only"` / `"epsilon"` | explains the path and shows **no scores**, because none exist |
| candidate lacks `final` | score-only strip |
| pre-`cardMeta` / pre-`result` replays | existing fallbacks unchanged |

No path may produce a console error or an empty framed region.

---

## 5. Testing

**Python** (deferred until the run ends, developed against the scratch
wrapper):
- capture is off by default and costs nothing when off;
- schema shape, `top_k` cap, and the `no_candidates` / `rules_only` / `epsilon`
  paths;
- **action identity** between capture on and off (§3.6).

**Viewer:** fixture replays covering (a) full `teacherDebug`, (b) none,
(c) mixed ticks, (d) a pre-`cardMeta` legacy file. Each is loaded headlessly
and checked for console errors and for absence of empty UI regions.

**Visual:** the arena is screenshotted against the reference frame and
iterated on, rather than declared correct from the hex values alone.

---

## 6. Sequencing and deliverables

1. **Aesthetic** — palette, greyscale theme, right-panel rebuild.
2. **Entity inspector** — §2.2 and §2.3 in the viewer; §2.1 written to
   `perception/UPSTREAM_REQUESTS.md`.
3. **Teacher Debug** — scratch capture wrapper → real fixture replays →
   Simulation View UI → staged `teacher.py` patch, applied only once the
   training run has finished.

Deliverables: a rewritten `web/viewer.html`, an `UPSTREAM_REQUESTS.md` entry, a
staged patch plus `python_ai/rl/teacher_debug.py` held outside the package
until it is safe to land, and the fixture replays.
