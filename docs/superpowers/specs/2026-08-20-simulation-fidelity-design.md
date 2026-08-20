# Four simulation-fidelity fixes — design

**Date:** 2026-08-20
**Status:** approved for implementation (coordinates already landed and visually confirmed)
**Source of truth:** a professional Clash Royale player's audit, plus the sight-range
data catalog reproduced in `include/core/CardStats.h`'s `sightRange` comment.

---

## 0. What was already true before this work

Two of the four requested items were found already implemented. They are recorded
here so nobody re-does them, and so the *actual* remaining work is legible.

**Rule A (strict sight) already ships.** `CardRegistry.h` carries 58
`withSightRange(...)` calls whose values match the catalog exactly — Hog 9.5,
Giant 7.5, P.E.K.K.A. 5.0, Bowler 4.0, Mortar/X-Bow 11.5, Firecracker 8.5,
Balloon and Skeleton Barrel 7.7, King Tower 7.0, Princess Tower 7.5.
`CombatEntity::findTarget` and `BuildingTargeter::findTarget` both filter
candidates through `effectiveSightTo()`. The brief's Example 1 (a Hog ignoring a
Cannon on the far bridge) **already passes**: the bridges are 12 tiles apart and
the Hog's effective sight is 10.9.

**TimeoutRules already implements both tiebreakers**, is bound to Python as
`resolve_timeout_outcome`, and is wired into `ClashEnv::calculateReward`.
Measured with `tools/audit/timeout_audit.cpp`:

| scenario | result |
|---|---|
| both sides passive, 6 matches | 6/6 reached tick 3600 → towers 3-3, weakest 2534 vs 2534 → `DRAW`, `reward0 = +0.0` |
| vs the C++ `HeuristicOpponent`, 6 matches | 0/6 reached the limit; all decided on a King |

The draws in the first row are *correct*: two passive teams take no damage, so
every tower sits at full 2534 and it is a genuine exact tie on both criteria —
the one path the brief also leaves as a draw.

So item 4 is not a rule defect. It is a **consumer** defect, and only in
JavaScript. See §4.

---

## 1. King Tower activation

### Rule

The King Tower has a latching `awake` state, initially `false`. While `false` it
can neither acquire a target nor fire. It stays **targetable and damageable**. It
wakes permanently on either trigger:

- it takes any damage, or
- any Princess Tower on its own team is destroyed.

### Design

`Tower` gains `bool awake`. Princess Towers construct `awake = true`; the King
constructs `false` (set by `GameManager::addTower`'s `symbol == 'R'` branch,
which already special-cases the King for `sightRange`).

**Damage trigger — latched in `Tower::update` on `hp < maxHp`, not in an override
of a damage entry point.** Damage reaches a tower through several paths
(`Projectile::applyHit`, `AreaSpell::update`, direct `performAttack`, splash,
poison-over-time), and overriding one of them would silently miss the others.
Testing the HP invariant is path-independent. Because it is a *latch* — set only,
never cleared — a subsequent heal cannot put the King back to sleep.

**Princess trigger — the King records its team's living Princess Tower count on
its first `update`, and wakes when the live count drops below that.** Recording
the initial count rather than hardcoding `< 2` keeps bare-board unit tests (a
King constructed with no Princesses) from waking instantly, which a `< 2` test
would do.

**Gate — `Tower::findTarget` returns `nullptr` while asleep.** That is the single
choke point: no acquisition means no lock, so `resolveCurrentTarget` finds
nothing and `performAttack` is never reached. No change to `Building::update`.

`Tower::snapshot()` already copy-constructs, so the flag survives `Board::deepCopy`
and therefore search rollouts, with no extra code.

### Blast radius

This is the largest gameplay change of the four. The 2026-08-20 audit measured
the King's share of damage against a lone Hog at **36.8%** (630 of 1710) after
the sight fix widened its effective reach to 9.4 tiles. All of that currently
comes from a tower that should have been dormant. Defence gets materially weaker.
That is the intended correction.

---

## 2. Tower and bridge coordinates — LANDED

Applied and visually confirmed against the player's own river-row map
`WWBBWWWWWWWWWWBBWW` before anything else was touched.

x is a **cell index** clamped to `[0, width-1]`, so the board's centre — the
fixed point of the mirror `17 - x` — is **8.5**, not 9.0. This is the x-analogue
of `extractObservationForTeam`'s `y -> 33 - y`, and the convention
`Board::isBackRowDeadZone` already used for its own centre.

| | before | after | occupies | mirrors to |
|---|---|---|---|---|
| King (both teams) | 9.0 | **8.5** | cols 7–10 (4 wide) | itself |
| Left Princess | 4.0 | **3.0** | cols 2–4 (3 wide) | 14.0 |
| Right Princess | 14.0 | 14.0 | cols 13–15 | 3.0 |
| Left bridge | 4.0 | **2.5** | cols 2–3 | 14.5 |
| Right bridge | 14.0 | **14.5** | cols 14–15 | 2.5 |

**A two-tile bridge's centre sits on the seam between tiles, not on a tile.**
That is why 2.5/14.5 and not 3.0/14.0: `clampToBoard`'s `±BRIDGE_HALF_WIDTH`
corridor then spans exactly cells 2 and 3, since cell `i` covers
`[i-0.5, i+0.5]`. Centring on a tile is what made the corridor three wide.

Supporting changes: `Board::getLeftBridge()/getRightBridge()` (read-only, so
nothing keeps a second copy — `HeuristicOpponent`'s own `LEFT_BRIDGE_X = 3.5f`
was already stale before today) and a named `Board::BRIDGE_HALF_WIDTH`.

This reverses the 2026-07-30 calibration-driven move in CLAUDE.md. That fit was
anchored on a half-tile convention error; the corrected layout is self-consistent
and mirror-symmetric, which the old one was only about the wrong centre.

### Consequence for §3

The bridge no longer shares a centre with its Princess Tower (2.5 vs 3.0). A unit
crossing at 2.5 must curve half a tile inward to reach a tower at 3.0.

---

## 3. Aggro, sight, and lane pathing

### Rule A — strict sight

**No behaviour change.** Sight stays measured **surface-to-surface**, the same
way attack range is: `effectiveSightTo = sightRange + own radius + target radius`.

Keeping the two parallel *is* the invariant: as long as `sightRange >=
attackRange`, effective sight >= effective attack range, so nothing can ever
attack what it cannot see. Breaking that parity is what produced the measured
2026-08-20 defect where a Musketeer destroyed a 3204 hp Princess Tower from 8
tiles taking **zero** damage in return.

Work here is regression tests only (§5).

### Rule B — blind lane pathing

**This is the genuinely missing behaviour.** `findTarget`'s fallback when nothing
is in sight is *closest enemy Tower by raw distance*.

**The reproduction is position-dependent, and that matters for the test.** With
team 1's left Princess destroyed, a unit in the left lane measures:

| unit at | → right Princess (14, 27) | → King (8.5, 30.5) | closest-tower picks |
|---|---|---|---|
| (2.5, 12.0) | 18.90 | 19.45 | **right Princess — wrong** |
| (2.5, 14.0) | 17.36 | 17.56 | **right Princess — wrong** |
| (2.5, 15.0) | 16.62 | 16.62 | tie |
| (2.5, 16.5) — the bridge | 15.57 | 15.23 | King — *right, by coincidence* |
| (2.5, 19.0) | 14.01 | 12.97 | King — *right, by coincidence* |

The crossover sits at y ≈ 15.0. **At the bridge mouth the broken rule already
gives the right answer**, so a test written at the bridge would pass against
unfixed code and prove nothing — the "a cross-check anchored where the error is
zero" trap CLAUDE.md records from the 2026-08-05 tile-grid refit. The regression
test must place the unit at **y ≤ 14**, behind the bridge on its own side, which
is also where a tank is actually deployed. It then walks the whole lane, so the
approach shape is exercised too.

New `include/core/LanePath.h` — pure geometry over `Board` and `Entity`. It
identifies towers through `Entity::isTower()` and `symbol == 'R'` rather than
including `Tower.h`, which avoids the `Tower.h -> Building.h -> CombatEntity.h`
include cycle.

```
lane path, unit on team T in lane L
  W0 = bridge(L)                             (2.5 | 14.5, 16.5)
  W1 = (lanePrincessX(L), princessRowY(!T))  (3.0 | 14.0, 27.0 | 6.0)
  W2 = enemy King                            (8.5, 30.5 | 2.5)
```

Two consumers:

1. **Lane-aware objective.** `findTarget`'s blind fallback becomes: *my own
   lane's* enemy Princess Tower if alive, else the enemy King. Shared by
   `CombatEntity::findTarget` and `BuildingTargeter::findTarget` so a
   building-targeter cannot drift out of agreement with a troop.

2. **Lane-aware approach.** Movement toward a **Tower** objective routes through
   the polyline and only then through the existing `Board::getNextWaypoint` for
   the river crossing — the two compose, the river logic is untouched. With the
   lane's Princess dead the unit walks up its own lane to the empty Princess slot
   and *then* angles in to the King. That is the "curve" the brief describes, and
   it is not a diagonal shortcut across the arena.

**Lane assignment is nearest-bridge, re-evaluated every tick.** This is the same
rule `getNextWaypoint` already uses to choose a crossing, so the objective and
the crossing agree by construction and there is no new per-entity state for
`Board::deepCopy` or `snapshot()` to carry.

**No "am I blind?" flag.** The rule is simply *a Tower objective is approached
along its lane*. When the tower is already in sight the polyline's last leg is
the tower itself, so the rule is a no-op there — the flag would have been state
with no observable effect.

### Path shape validation

The polyline's shape is derived analytically from the corrected geometry in §2.
It is then **verified by hand against `perception/assets/recordings/`**: sample
frames where a lone unit walks unopposed and compare its track to the engine's.

The video is used as an *oracle*, not a dependency. There is no cross-frame unit
tracker in `perception/` today and `detect/placements.py` currently raises
(stage 3 is "blocked on data"), so deriving the curve from video would be a
separate multi-day sub-project. If the hand check shows the real path differs
from the analytic one, the polyline is data and is cheap to correct; the finding
gets recorded either way.

---

## 4. Timeout verdict consumers

`TimeoutRules::resolve` is correct (§0). Its **consumers** re-derive the verdict,
and the one that survived every previous cleanup is in JavaScript.

`web/viewer.html:803`, `computeMatchResult()`:

```js
// Winner detection -- mirrors MatchRules::evaluate exactly
const blueKingAlive = finalTick.entities.some(e => e.symbol === 'R' && e.team === 0);
const redKingAlive  = finalTick.entities.some(e => e.symbol === 'R' && e.team === 1);
if (blueKingAlive && redKingAlive)
  return { winner: 'draw', reason: 'Timeout — both King Towers still standing' };
```

The comment is accurate and is also the bug. `MatchRules::evaluate` answers *"has
a King died yet?"* — it is called every tick and correctly returns `{over:false}`
right up to the limit. *"Who won?"* at the limit is `TimeoutRules::resolve`, which
the viewer never asks. A severely damaged Princess Tower is reported as a draw.

Three changes:

- **`GameLogger` writes the engine's verdict into the replay JSON** as a
  `"result"` object — `MatchRules` for a King KO, else `TimeoutRules`. Computed
  from GameLogger's own final snapshot (which already carries per-entity
  `team`/`symbol`/`hp`), so no new plumbing and no `Board` handle is needed.
- **`viewer.html` consumes `gameData.result`.** Its fallback for replays written
  before this field existed becomes a client-side port of TimeoutRules — tower
  count, then **absolute** weakest HP — not king-alive-only.
- **The guard test is extended past Python.**
  `test_match_outcome_is_the_only_scorer.py` walks `.py` files only, which is
  precisely why this defect survived in `.html`. It gains `.html`/`.js`.

**Absolute HP, not percentage.** Real Clash Royale breaks this tie on HP
*fraction*, and King 4008 vs Princess 2534 means the two disagree often — but the
brief specifies absolute and the engine already does absolute. Recorded as a
deliberate, known divergence rather than an oversight.

---

## 5. Testing

Every fix gets dedicated C++ regression tests. New files:

**`tests/core/test_king_activation.cpp`**
- a fresh King acquires no target and deals no damage with an enemy in range
- it takes damage → wakes → fires on the next tick
- a friendly Princess Tower dies → wakes, with the King itself untouched
- waking is permanent: healing back to full does not re-sleep it
- an asleep King is still targetable and damageable
- a Princess Tower is never asleep
- `snapshot()` carries the flag both ways

**`tests/core/test_lane_pathing.cpp`**
- Example 2: left Princess destroyed, Ice Golem in the left lane **at y ≤ 14**
  (see §3 — at the bridge itself the unfixed rule already answers correctly, so
  the test must be anchored where it demonstrably fails) → objective is the
  **King**, never the surviving right Princess
- the same case pinned as a *characterisation* of the old rule, so the test is
  proven to fail before the fix rather than assumed to
- Example 1, verbatim: Hog on the left bridge, Cannon on the right bridge →
  Cannon is outside effective sight, objective stays the tower
- lane's Princess alive → that Princess is the objective even when the other
  lane's is nominally closer
- the approach curves: sampled positions on the left-lane-to-King path stay in
  the left lane past the river before angling in
- `BuildingTargeter` and `CombatEntity` agree on the objective from the same
  position

**Extensions**
- `test_sight_range.cpp` — catalog values pinned for all eight `DEFAULT_DECK`
  cards; the parity invariant `effectiveSightTo >= effectiveRangeTo` asserted
  across the deck
- `test_board.cpp` — **the bridge literals become `board.getLeftBridge().x`.**
  Both deadlock sweeps hardcode `const float bridges[] = { 4.0f, 14.0f }` and
  line 27 asserts `clampToBoard` leaves `x = 4.0` in the river band, which is
  false now. Reading the bridge off the board makes them sweep the real bridges
  and survive the next move.
- `test_timeout_rules.cpp` — a case pinning that the verdict a replay would
  report matches `TimeoutRules::resolve`

### The bridge-deadlock regression, proven with instruments not assertions

The absorbing-state defects of 2026-08-09 and 2026-08-20 were both properties of
a *trajectory*, which end-state assertions cannot see. The existing instruments
are the real proof and must be re-run after the pathing change:

| instrument | bar |
|---|---|
| `tools/audit/waypoint_probe.cpp` | **0** absorbing states, extended to cover lane waypoints (was 0 over 8,661,439 positions) |
| `tools/audit/bridge_audit.cpp` | 34/34 crossings for every card; longest stall == `DEPLOY_TIME_TICKS` |
| `tools/audit/soak.cpp` | stall rate no worse than the recorded 7 per 342,563 unit-ticks, **none on or near a bridge** |

### Suite bars

- C++: 582 cases / 5,737 assertions before this work, of which exactly one is the
  `[!shouldfail]` navigation-wedge case. Runner must still exit 0.
- Python: 366 passed / 2 skipped.
- Perception: 353 passed / 1 skipped.

---

## 6. What this invalidates

**Gameplay-affecting: items 1, 2 and 3.** Every win rate, Elo figure and
placement score in CLAUDE.md becomes historical, including the `phase1_v5` run
stopped at the start of this session.

**Checkpoints still load.** There is no observation, action-space, reward or
architecture change, so `model_weights.pth` and `model_weights_selfplay.pth`
resume normally — their measured *strength* simply no longer means anything.

**Item 4 is not gameplay-affecting.** It changes only what a replay reports.

### Second copies to reconcile

The coordinate change makes several downstream copies stale. Each is derived from
the engine rather than re-hardcoded, per CLAUDE.md's no-second-copies rule:

- `HeuristicOpponent::LEFT_BRIDGE_X / RIGHT_BRIDGE_X` — 3.5/13.5, already stale
  before today; points at `Board::getLeftBridge()`
- `perception/geometry.py` — `LEFT_BRIDGE`, `RIGHT_BRIDGE`,
  `OWN/OPP_PRINCESS_LEFT`, King positions
- any `python_ai/advisors/tactics.py` bridge cell used by the advisor
