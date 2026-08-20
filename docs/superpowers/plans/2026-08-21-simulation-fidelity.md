# Simulation Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring four engine mechanics in line with real Clash Royale — King Tower dormancy, corrected arena coordinates, sight-gated lane pathing, and a timeout verdict that replay consumers actually read.

**Architecture:** Arena geometry is consolidated into one header (`ArenaLayout.h`) that `Board`, `GameManager` and a new `LanePath` all read, so no coordinate exists twice. King dormancy is a latching flag on `Tower` gated at `findTarget`. Lane pathing is a two-part change: `findTarget`'s blind fallback picks the unit's *own lane's* objective instead of the globally-closest tower, and a single call site routes movement toward a King through an intermediate lane waypoint. The timeout rule itself is unchanged — only its consumers are fixed.

**Tech Stack:** C++17 header-only engine, Catch2 (`build_python/ClashRoyaleTests.vcxproj`), MSVC via absolute MSBuild path, standalone audit instruments built by `tools/audit/build.ps1`, pybind11 `.pyd` for the Python side, vanilla JS for `web/viewer.html`.

**Spec:** [docs/superpowers/specs/2026-08-20-simulation-fidelity-design.md](../specs/2026-08-20-simulation-fidelity-design.md)

## Global Constraints

- **Machine A toolchain.** `cmake`, `cl`, `msbuild` are NOT on PATH. Build the suite with the absolute path below. Run MSBuild from the **PowerShell** tool, never Bash — Git Bash mangles `/p:Configuration=Release` into `p:Configuration=Release`.
- **Never rebuild the `.pyd` while any Python process holds it.** Check `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` first; the post-build copy fails with MSB3073 otherwise. A foreign Claude session was running `defence_gradient.py` at plan time — do not kill another session's work; ask.
- **`python_ai/` and `include/`, `src/` are normally READ-ONLY.** This work is explicitly authorised to edit them.
- **No second copy of an engine constant.** Derive from `ArenaLayout.h` (C++) or the bindings (Python). This rule has been violated three times in this repo already.
- **Two independent copies of "close enough" is a deadlock.** Any new waypoint MUST use `Board::WAYPOINT_ARRIVAL_EPS` and MUST hand back something else once the mover is within it. Two absorbing states have already shipped from exactly this.
- **Baseline before this work:** 582 cases / 5,709 assertions; 9 real failures from the already-landed coordinate change, plus the `[!shouldfail]` wedge case which now *passes* its assertions and so reports as failed.
- **Coordinates (already landed, user-approved):** King 8.5, Princess 3.0/14.0, bridges 2.5/14.5, all symmetric about `17 - x`.
- Commit after every task. Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

**Build the suite:**
```
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" build_python\ClashRoyaleTests.vcxproj /p:Configuration=Release /p:Platform=x64 /m /v:minimal
```
**Run it:** `./build_python/Release/ClashRoyaleTests.exe`
**Run one case:** `./build_python/Release/ClashRoyaleTests.exe "exact test name"`

---

### Task 1: ArenaLayout.h — one source of truth for arena geometry

**Files:**
- Create: `include/core/ArenaLayout.h`
- Modify: `include/core/Board.h` (the `leftBridge`/`rightBridge` initialisers)
- Modify: `include/core/GameManager.h` (the six `addTower` calls in `setupTowers`)
- Test: `tests/core/test_arena_layout.cpp` (create)
- Modify: `build_python/ClashRoyaleTests.vcxproj` — only if new test files are not picked up by a glob; check first with `grep -c test_board.cpp build_python/ClashRoyaleTests.vcxproj`. If files are listed explicitly, add every new test file created by this plan.

**Interfaces:**
- Consumes: nothing.
- Produces: namespace `ArenaLayout` with `WIDTH`, `HEIGHT`, `CENTER_X`, `LEFT_LANE_X`, `RIGHT_LANE_X`, `LEFT_BRIDGE_X`, `RIGHT_BRIDGE_X`, `BRIDGE_Y`, `princessY(int team)`, `kingY(int team)`, `mirrorX(float)`. Tasks 6 and 7 depend on all of these.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_arena_layout.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "ArenaLayout.h"
#include "GameManager.h"
#include "Board.h"

// The arena is symmetric about the CELL-INDEX centre (WIDTH-1)/2 = 8.5, which
// is the x-analogue of ClashEnv::extractObservationForTeam's y -> 33 - y. Every
// pair below must mirror onto the other under 17 - x, or one lane is playable
// differently from the other -- a class of bug this repo has paid for twice.
TEST_CASE("ArenaLayout is mirror-symmetric about the board centre", "[arena][geometry]") {
    REQUIRE(ArenaLayout::CENTER_X == Catch::Approx(8.5f));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::LEFT_LANE_X) == Catch::Approx(ArenaLayout::RIGHT_LANE_X));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::LEFT_BRIDGE_X) == Catch::Approx(ArenaLayout::RIGHT_BRIDGE_X));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::CENTER_X) == Catch::Approx(ArenaLayout::CENTER_X));
}

// The player's own map of the river row: WWBBWWWWWWWWWWBBWW -- bridges occupy
// columns 2-3 and 14-15, so each CENTRE sits on the seam between its two tiles.
// Centring on a tile instead is what made the corridor three wide.
TEST_CASE("bridges are two tiles wide, on the seam", "[arena][geometry]") {
    REQUIRE(ArenaLayout::LEFT_BRIDGE_X == Catch::Approx(2.5f));
    REQUIRE(ArenaLayout::RIGHT_BRIDGE_X == Catch::Approx(14.5f));

    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    const float midRiver = 16.5f;
    // Exactly columns 2,3 and 14,15 survive clampToBoard inside the river band.
    for (int x = 0; x < ArenaLayout::WIDTH; ++x) {
        Vector2D probe{ static_cast<float>(x), midRiver };
        bool walkable = board.clampToBoard(probe, false).y == Catch::Approx(midRiver);
        bool expected = (x == 2 || x == 3 || x == 14 || x == 15);
        INFO("column " << x);
        REQUIRE(walkable == expected);
    }
}

// GameManager must not carry its own copy of these numbers.
TEST_CASE("GameManager spawns its towers exactly where ArenaLayout says", "[arena][geometry]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    int kingsSeen = 0, princessSeen = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (!e->isTower()) continue;
        if (e->symbol == 'R') {
            kingsSeen++;
            REQUIRE(e->position.x == Catch::Approx(ArenaLayout::CENTER_X));
            REQUIRE(e->position.y == Catch::Approx(ArenaLayout::kingY(e->team)));
        } else {
            princessSeen++;
            bool left = e->position.x < ArenaLayout::CENTER_X;
            REQUIRE(e->position.x == Catch::Approx(left ? ArenaLayout::LEFT_LANE_X
                                                        : ArenaLayout::RIGHT_LANE_X));
            REQUIRE(e->position.y == Catch::Approx(ArenaLayout::princessY(e->team)));
        }
    }
    REQUIRE(kingsSeen == 2);
    REQUIRE(princessSeen == 4);
}

// Each Princess Tower is 3 wide (radius 1.5) and its lane's bridge is 2 wide;
// the bridge covers the tower's two OUTER columns. Pins the relationship the
// lane path in LanePath.h relies on: a unit crossing at 2.5 must curve half a
// tile inward to reach a tower at 3.0.
TEST_CASE("each bridge sits half a tile outboard of its own Princess Tower",
          "[arena][geometry]") {
    REQUIRE(ArenaLayout::LEFT_LANE_X - ArenaLayout::LEFT_BRIDGE_X == Catch::Approx(0.5f));
    REQUIRE(ArenaLayout::RIGHT_BRIDGE_X - ArenaLayout::RIGHT_LANE_X == Catch::Approx(0.5f));
}
```

- [ ] **Step 2: Run test to verify it fails**

```
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" build_python\ClashRoyaleTests.vcxproj /p:Configuration=Release /p:Platform=x64 /m /v:minimal
```
Expected: **compile error**, `cannot open include file: 'ArenaLayout.h'`.

- [ ] **Step 3: Write minimal implementation**

Create `include/core/ArenaLayout.h`:

```cpp
#pragma once

// The arena's fixed geometry, in ONE place.
//
// Before this header the same numbers lived in three: Board's leftBridge/
// rightBridge members, GameManager::setupTowers' six addTower literals, and
// HeuristicOpponent's own LEFT_BRIDGE_X/RIGHT_BRIDGE_X -- which had already
// gone stale (3.5/13.5) against Board's own values. LanePath.h needs the
// Princess COLUMNS, which existed only inside GameManager, and copying them a
// fourth time is what this header exists to prevent.
//
// COORDINATE CONVENTION. x is a CELL INDEX, clamped by Board::clampToBoard to
// [0, WIDTH-1]. So the board's centre -- and the fixed point of the mirror
// mirrorX -- is (WIDTH-1)/2 = 8.5, NOT 9.0. This is the x-analogue of
// ClashEnv::extractObservationForTeam's y -> (HEIGHT-1) - y, and it is the
// convention Board::isBackRowDeadZone already used for its own centre.
//
// Corrected 2026-08-20 from a professional player's audit: the King was at
// 9.0 and the left Princess at 4.0, symmetric about 9.0 rather than 8.5 --
// the same half-tile convention error the river carried before it was
// re-centred on 16.5.
namespace ArenaLayout {

inline constexpr int WIDTH = 18;
inline constexpr int HEIGHT = 34;

// The mirror that maps one team's frame onto the other, and one lane onto the
// other. Its fixed point is the board centre.
inline constexpr float CENTER_X = (WIDTH - 1) / 2.0f;   // 8.5
constexpr float mirrorX(float x) { return (WIDTH - 1) - x; }
constexpr float mirrorY(float y) { return (HEIGHT - 1) - y; }

// Princess Tower columns. 3 tiles wide (Tower::getCollisionRadius 1.5), so
// the left one occupies columns 2-4 and the right 13-15.
inline constexpr float LEFT_LANE_X  = 3.0f;
inline constexpr float RIGHT_LANE_X = mirrorX(LEFT_LANE_X);   // 14.0

// Bridge centres. A bridge is TWO tiles wide -- the real river row reads
//     column  012345678901234567
//             WWBBWWWWWWWWWWBBWW
// so columns 2-3 and 14-15, and each centre lands on the SEAM between its two
// tiles rather than on a tile. Cell i covers [i-0.5, i+0.5], so a centre of
// 2.5 with Board::BRIDGE_HALF_WIDTH = 1.0 spans exactly cells 2 and 3.
inline constexpr float LEFT_BRIDGE_X  = 2.5f;
inline constexpr float RIGHT_BRIDGE_X = mirrorX(LEFT_BRIDGE_X);   // 14.5
inline constexpr float BRIDGE_Y = 16.5f;                          // river band centre

// Tower rows, per team. Team 0 defends the low-y end.
inline constexpr float KING_Y_TEAM0     = 2.5f;
inline constexpr float PRINCESS_Y_TEAM0 = 6.0f;
constexpr float kingY(int team)     { return team == 0 ? KING_Y_TEAM0     : mirrorY(KING_Y_TEAM0); }
constexpr float princessY(int team) { return team == 0 ? PRINCESS_Y_TEAM0 : mirrorY(PRINCESS_Y_TEAM0); }

// Which lane a position belongs to. Nearest-bridge, matching the rule
// Board::getNextWaypoint already uses to pick a crossing -- so a unit's lane
// objective and its chosen bridge agree by construction and it cannot be
// routed to one bridge while aiming at the other lane's tower.
constexpr bool isLeftLane(float x) { return x < CENTER_X; }
constexpr float bridgeXFor(float x) { return isLeftLane(x) ? LEFT_BRIDGE_X : RIGHT_BRIDGE_X; }
constexpr float laneXFor(float x)   { return isLeftLane(x) ? LEFT_LANE_X   : RIGHT_LANE_X; }

}  // namespace ArenaLayout
```

- [ ] **Step 4: Point Board and GameManager at it**

In `include/core/Board.h`, add `#include "ArenaLayout.h"` at the top, then replace the two bridge member initialisers (keep the existing explanatory comment block above them — only the values change to references):

```cpp
    Vector2D leftBridge{ ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::BRIDGE_Y };
    Vector2D rightBridge{ ArenaLayout::RIGHT_BRIDGE_X, ArenaLayout::BRIDGE_Y };
```

In `include/core/GameManager.h`, add `#include "ArenaLayout.h"`, then replace the six `addTower` calls in `setupTowers` (around line 629) with:

```cpp
        addTower(ArenaLayout::CENTER_X, ArenaLayout::kingY(0), 4008, 0, 7.0f, 90, 10, 'R', "King Tower");
        addTower(ArenaLayout::CENTER_X, ArenaLayout::kingY(1), 4008, 1, 7.0f, 90, 10, 'R', "King Tower");

        addTower(ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(0), 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(0), 0, "Princess Tower", towerTroopStats(aiTowerTroop));
        addTower(ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(1), 1, "Princess Tower", towerTroopStats(oppTowerTroop));
        addTower(ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(1), 1, "Princess Tower", towerTroopStats(oppTowerTroop));
```

- [ ] **Step 5: Run the new tests**

Run: `./build_python/Release/ClashRoyaleTests.exe "[arena]"`
Expected: **PASS**, 4 cases.

- [ ] **Step 6: Commit**

```bash
git add include/core/ArenaLayout.h include/core/Board.h include/core/GameManager.h tests/core/test_arena_layout.cpp build_python/ClashRoyaleTests.vcxproj
git commit -m "$(cat <<'EOF'
Arena geometry had four copies; give it one header

x is a cell index in [0, 17], so the board centre -- the fixed point of the
mirror 17 - x -- is 8.5, not 9.0. The King sat at 9.0 and the left Princess at
4.0, symmetric about the wrong centre, and the bridges were centred on a tile
rather than on the seam between their two tiles, which made the clampToBoard
corridor three columns wide instead of two.

Corrected against a professional player's own map of the river row
(WWBBWWWWWWWWWWBBWW): King 8.5, Princess 3.0/14.0, bridges 2.5/14.5, every
pair mirroring onto the other under 17 - x.

ArenaLayout.h now owns all of it. Board and GameManager read it instead of
holding literals; LanePath (next) needs the Princess columns, which existed
only inside GameManager, and a fourth copy is what this prevents.

GAMEPLAY-AFFECTING. Every win rate earned before this is historical.
Checkpoints still load -- no observation, action-space or architecture change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Repair the coordinate-dependent tests

Nine existing cases assert the old coordinates as literals. They must read the
geometry off the board instead, so they sweep the *real* bridges and survive the
next move. **This is the task that protects the bridge-deadlock fix** — both
deadlock sweeps currently sweep water.

**Files:**
- Modify: `tests/core/test_board.cpp` (lines ~27, 301–324, 348–361, 372, 420–437, 462–468)
- Modify: `tests/entities/test_troop.cpp` (lines ~26–42, ~116–119)
- Modify: `tests/core/test_game_manager.cpp` (line ~25)

**Interfaces:**
- Consumes: `ArenaLayout` from Task 1; `Board::getLeftBridge()/getRightBridge()` (already added).
- Produces: nothing.

- [ ] **Step 1: Confirm the exact failure set**

Run: `./build_python/Release/ClashRoyaleTests.exe`
Expected: `582 | 572 passed | 10 failed`. Nine named cases:
```
clampToBoard does not snap positions sitting on a bridge column
getNextWaypoint routes below-to-above via the nearest bridge's start edge
getNextWaypoint routes above-to-below via the nearest bridge's end edge
getNextWaypoint picks the left bridge when it is nearer
getNextWaypoint does not strand a unit standing on the near bank
getNextWaypoint does not strand a unit standing on the far bank
GameManager construction sets up exactly the 6 expected towers
Troop routes through the nearest bridge when crossing the river
Troop::clampPosition does not snap positions sitting on a bridge column
```
The 10th is the `[!shouldfail]` wedge — leave it, Task 3 owns it.

- [ ] **Step 2: Replace every hardcoded bridge column**

In `tests/core/test_board.cpp`, both deadlock sweeps currently read:
```cpp
    const float bridges[] = { 4.0f, 14.0f };
```
Replace **both** with:
```cpp
    // Read off the board, not written down again: these sweeps exist to prove
    // no bridge mouth is an absorbing state, and a literal silently turns them
    // into a sweep of open water the moment a bridge moves (which it did on
    // 2026-08-20). See Board::getLeftBridge.
    const float bridges[] = { board.getLeftBridge().x, board.getRightBridge().x };
```

For the remaining sites, substitute as follows — in every case replace the bare
`4.0f` used as a bridge column with `board.getLeftBridge().x` and `14.0f` with
`board.getRightBridge().x`. Specifically:

- line ~27–28, `clampToBoard does not snap positions sitting on a bridge column`:
```cpp
    REQUIRE(board.clampToBoard(Vector2D{ board.getLeftBridge().x, 17.0f }, false).y == Catch::Approx(17.0f));
    REQUIRE(board.clampToBoard(Vector2D{ board.getRightBridge().x, 17.0f }, false).y == Catch::Approx(17.0f));
```
- lines ~301–324, the three `getNextWaypoint routes/picks` cases: replace the
  expected `wp.x == Catch::Approx(14.0f)` with
  `wp.x == Catch::Approx(board.getRightBridge().x)`, and
  `Catch::Approx(4.0f) // leftBridge.x` with
  `Catch::Approx(board.getLeftBridge().x)`. Also update the *input* positions in
  those cases from `Vector2D{ 4.0f, ... }` / `Vector2D{ 14.0f, ... }` to the
  matching `board.getLeftBridge().x` / `board.getRightBridge().x`.
- lines ~348, ~360, ~420, ~428, ~462, ~467: same substitution on the `here` /
  input positions.

In `tests/entities/test_troop.cpp`:
- `Troop::clampPosition does not snap positions sitting on a bridge column`
  (~line 116): the section is named `left bridge (x in [3,5])`. Rename it to
  `left bridge` and use `board.getLeftBridge().x` for the position.
- `Troop routes through the nearest bridge when crossing the river` (~line 26–42)
  asserts a hardcoded post-move x of `10.588172f`. **Do not re-derive this by
  hand.** Replace the assertion with a property that cannot go stale:
```cpp
    // The troop must be closing on the bridge it chose, not on a literal
    // recomputed for today's coordinates. Bridge choice is nearest-bridge, so
    // assert the invariant instead of the arithmetic.
    float bridgeX = board.getRightBridge().x;
    float before = std::fabs(startX - bridgeX);
    float after = std::fabs(troop->position.x - bridgeX);
    REQUIRE(after < before);
    REQUIRE(troop->position.y > startY);   // and it is heading upfield
```
  (declare `startX`/`startY` from the troop's spawn position before the step).

In `tests/core/test_game_manager.cpp` (~line 25), replace
`Catch::Approx(9.0f)` with `Catch::Approx(ArenaLayout::CENTER_X)` and add
`#include "ArenaLayout.h"`. Update the Princess assertions in the same case to
`ArenaLayout::LEFT_LANE_X` / `ArenaLayout::RIGHT_LANE_X`.

- [ ] **Step 3: Rebuild and run**

Run the MSBuild command, then `./build_python/Release/ClashRoyaleTests.exe`
Expected: `582 | 581 passed | 1 failed` — only the `[!shouldfail]` wedge, which Task 3 owns.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
Tests pinned the bridge columns as literals; read them off the board

Both bridge-deadlock sweeps hardcoded { 4.0f, 14.0f }, so after the arena
correction they swept open water -- the regression tests for two shipped
absorbing states were no longer testing a bridge at all. Nine cases across
test_board, test_troop and test_game_manager did the same in smaller ways.

They now read Board::getLeftBridge()/getRightBridge() and ArenaLayout, so they
follow the geometry instead of restating it.

test_troop's "routes through the nearest bridge" asserted a hand-computed post-
move x (10.588172f). Replaced with the invariant it was really checking -- the
troop closes on the bridge it chose and heads upfield -- which cannot go stale.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Re-anchor the navigation-wedge reproduction

The `[!shouldfail]` wedge case stopped failing: its assertions now pass, because
moving the towers changed the pinch geometry that formed one wall of the trap.
**That is not evidence the defect is fixed** — `Board::pushAwayFrom`'s opposing-
slide cancellation is untouched. Settle it with the instrument, not the test.

**Files:**
- Modify: `tests/core/test_navigation_wedge.cpp`
- Run: `tools/audit/soak.cpp`

**Interfaces:**
- Consumes: `ArenaLayout` from Task 1.
- Produces: nothing.

- [ ] **Step 1: Measure the stall rate under the new geometry**

```
powershell -File tools/audit/build.ps1 soak
./tools/audit/bin/soak.exe
```
Record: total unit-ticks, stall count, and every stall's position. The
pre-change reference is **7 stalls per 342,563 unit-ticks, none on or near a
bridge**.

- [ ] **Step 2: Branch on the measurement**

**If stalls are still reported** (the expected outcome): take the position of
the longest stall from the soak output and re-anchor the reproduction to it.
In `tests/core/test_navigation_wedge.cpp`, replace the two obstacle positions
and the troop's start position with the measured ones, keeping the
`[!shouldfail]` tag and adding to the existing comment block:

```cpp
// RE-ANCHORED 2026-08-21. The original reproduction used the pre-correction
// tower coordinates as one wall of the pinch; after the arena moved, that exact
// geometry no longer traps. The DEFECT is unchanged -- pushAwayFrom's opposing
// slides still cancel -- so this is re-pointed at a pocket that still
// reproduces, measured with tools/audit/soak.cpp on the corrected board.
```

**If soak reports ZERO stalls over at least 300,000 unit-ticks**: do NOT delete
the case and do NOT silently drop the tag. Change the tag from `[!shouldfail]`
to `[!mayfail]` and add:

```cpp
// 2026-08-21: soak.cpp measured ZERO stalls over N unit-ticks on the corrected
// arena, where the pre-correction board measured 7 per 342,563. The wedge
// mechanism in pushAwayFrom is UNCHANGED, so this is "the reproduction no
// longer reproduces", not "the defect is fixed" -- [!mayfail] keeps the case
// executable and green either way. Reopen UPSTREAM_REQUESTS.md item 18 if a
// stall is ever seen again, especially near a bridge.
```
Then report the number to the user rather than concluding the defect is gone.

- [ ] **Step 3: Rebuild and run the full suite**

Expected: `582 | 582 passed | 0 failed`, runner exits **0**.

- [ ] **Step 4: Commit**

```bash
git add tests/core/test_navigation_wedge.cpp
git commit -m "$(cat <<'EOF'
Re-anchor the collision-wedge repro to the corrected arena

The [!shouldfail] case stopped failing after the tower coordinates moved: the
old pinch used a tower as one of its two walls. pushAwayFrom's opposing-slide
cancellation is untouched, so this is the reproduction moving, not the defect
being fixed -- re-pointed at a pocket measured with tools/audit/soak.cpp.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: King Tower activation

**Files:**
- Modify: `include/entities/Tower.h`
- Modify: `include/core/GameManager.h` (`addTower`, the `symbol == 'R'` branch)
- Test: `tests/core/test_king_activation.cpp` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `Tower::isAwake() const -> bool`, `Tower::wake()`. Task 8's instruments and Task 12's notes reference `isAwake()`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_king_activation.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"
#include "Tower.h"
#include "MeleeTroop.h"

// Builds a bare board with one King for `team` and returns it.
static std::shared_ptr<Tower> makeKing(Board& board, int team) {
    auto king = std::make_shared<Tower>(board.allocateId(), ArenaLayout::CENTER_X,
                                        ArenaLayout::kingY(team), 4008, team, 7.0f, 90, 10, 'R');
    king->sightRange = 7.0f;
    board.addEntity(king);
    board.commitPendingEntities(0);
    return king;
}

static std::shared_ptr<MeleeTroop> makeEnemy(Board& board, int team, float x, float y, int hp = 5000) {
    auto t = std::make_shared<MeleeTroop>(board.allocateId(), x, y, hp, team, 'T', 0.0f, 1.0f, 10, 10);
    board.addEntity(t);
    board.commitPendingEntities(0);
    return t;
}

TEST_CASE("a Princess Tower is never dormant", "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto princess = std::make_shared<Tower>(board.allocateId(), ArenaLayout::LEFT_LANE_X,
                                            ArenaLayout::princessY(0), 2534, 0, 7.5f, 90, 8, 'P');
    REQUIRE(princess->isAwake());
}

TEST_CASE("a fresh King Tower is dormant and does not fire", "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto king = makeKing(board, 0);
    auto enemy = makeEnemy(board, 1, ArenaLayout::CENTER_X, ArenaLayout::kingY(0) + 3.0f);
    REQUIRE_FALSE(king->isAwake());

    int hpBefore = enemy->hp;
    for (int i = 0; i < 60; ++i) { king->update(board); board.commitPendingEntities(i); }
    REQUIRE(enemy->hp == hpBefore);   // never acquired, never shot
}

TEST_CASE("the King wakes when it takes any damage", "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto king = makeKing(board, 0);
    auto enemy = makeEnemy(board, 1, ArenaLayout::CENTER_X, ArenaLayout::kingY(0) + 3.0f);
    king->update(board);
    REQUIRE_FALSE(king->isAwake());

    king->takeDamage(1);
    king->update(board);
    REQUIRE(king->isAwake());

    int hpBefore = enemy->hp;
    for (int i = 0; i < 60; ++i) { king->update(board); board.commitPendingEntities(i); }
    REQUIRE(enemy->hp < hpBefore);    // now it fights back
}

TEST_CASE("the King wakes when a friendly Princess Tower is destroyed", "[king][activation]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();

    std::shared_ptr<Entity> king, princess;
    for (const auto& e : board.getEntities()) {
        if (!e->isTower() || e->team != 0) continue;
        if (e->symbol == 'R') king = e; else if (!princess) princess = e;
    }
    REQUIRE(king);
    REQUIRE(princess);
    game.step();
    REQUIRE_FALSE(std::dynamic_pointer_cast<Tower>(king)->isAwake());

    princess->takeDamage(princess->hp);     // destroy it, King itself untouched
    game.step();
    game.step();
    REQUIRE(std::dynamic_pointer_cast<Tower>(king)->isAwake());
    REQUIRE(king->hp == 4008);              // it woke without being hit
}

TEST_CASE("waking is permanent -- healing back to full does not re-sleep it",
          "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto king = makeKing(board, 0);
    king->takeDamage(100);
    king->update(board);
    REQUIRE(king->isAwake());

    king->hp = 4008;                        // fully healed
    king->update(board);
    REQUIRE(king->isAwake());               // latched, not recomputed
}

TEST_CASE("a dormant King is still targetable and damageable", "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto king = makeKing(board, 0);
    REQUIRE_FALSE(king->isAwake());
    REQUIRE(king->isTargetable());
    REQUIRE(king->isTower());
    king->takeDamage(500);
    REQUIRE(king->hp == 3508);
}

TEST_CASE("snapshot carries the King's dormancy both ways", "[king][activation]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    auto asleep = makeKing(board, 0);
    auto copyAsleep = std::dynamic_pointer_cast<Tower>(asleep->snapshot());
    REQUIRE_FALSE(copyAsleep->isAwake());

    asleep->wake();
    auto copyAwake = std::dynamic_pointer_cast<Tower>(asleep->snapshot());
    REQUIRE(copyAwake->isAwake());
}
```

- [ ] **Step 2: Run to verify it fails**

Run MSBuild, then `./build_python/Release/ClashRoyaleTests.exe "[king]"`
Expected: **compile error** — `isAwake` is not a member of `Tower`.

- [ ] **Step 3: Implement**

In `include/entities/Tower.h`, add to the class:

```cpp
public:
    // Real Clash Royale's King Tower starts DORMANT: it cannot acquire a target
    // or fire until it is activated, permanently, by either taking any damage or
    // by losing a Princess Tower on its own team. Until 2026-08-21 this engine's
    // King fired from tick 0, which made defence measurably stronger than the
    // real game's -- the 2026-08-20 audit put the King's share of the damage
    // against a lone Hog at 36.8%, all of it from a tower that should have been
    // asleep.
    //
    // Princess Towers construct awake; only the King is built asleep (see
    // GameManager::addTower's symbol == 'R' branch).
    bool isAwake() const { return awake; }
    void wake() { awake = true; }

    void update(Board& board) override {
        if (!awake) {
            // Trigger 1: any damage. Latched on the HP INVARIANT rather than by
            // overriding one damage entry point -- damage reaches a tower
            // through Projectile::applyHit, AreaSpell::update, direct
            // performAttack, splash and poison-over-time, and overriding one of
            // them would silently miss the rest. Because this only ever SETS
            // the flag, a later heal cannot put the King back to sleep.
            if (hp < maxHp) awake = true;

            // Trigger 2: a friendly Princess Tower destroyed. The count is
            // recorded on the first update rather than compared against a
            // hardcoded 2, so a King built on a bare board in a unit test (zero
            // Princesses) does not wake instantly -- which a `< 2` test would do.
            int living = 0;
            for (const auto& e : board.getEntities()) {
                if (e->isAlive() && e->isTower() && e->team == team && e->symbol != 'R') ++living;
            }
            if (initialFriendlyPrincesses < 0) initialFriendlyPrincesses = living;
            else if (living < initialFriendlyPrincesses) awake = true;
        }
        Building::update(board);
    }

protected:
    // The single choke point. No acquisition means no lock, so
    // resolveCurrentTarget finds nothing and performAttack is never reached --
    // no change to Building::update is needed. The tower stays targetable and
    // damageable, which is what lets trigger 1 ever fire.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        if (!awake) return nullptr;
        return Building::findTarget(board);
    }

private:
    bool awake = true;                    // Princess Towers; the King is set false at construction
    int initialFriendlyPrincesses = -1;   // -1 = not yet recorded
```

In `include/core/GameManager.h`, in the raw `addTower` overload, extend the
existing `symbol == 'R'` branch:

```cpp
        if (symbol == 'R') {
            tower->sightRange = 7.0f;
            // The King starts dormant -- see Tower::isAwake. This is the only
            // construction path that builds a King, so it is the only place the
            // flag needs clearing; every other Tower is a Princess and stays awake.
            tower->awake = false;
        }
```

`awake` is `private`, so add `friend class GameManager;` to `Tower`, **or**
simpler: give `Tower` a `void sleep() { awake = false; }` and call
`tower->sleep();`. Prefer `sleep()` — a friend declaration widens access to the
whole class for one field.

- [ ] **Step 4: Run to verify it passes**

Run: `./build_python/Release/ClashRoyaleTests.exe "[king]"`
Expected: **PASS**, 7 cases.

- [ ] **Step 5: Run the full suite**

Expected: `582+ passed`, runner exits 0. Deploy-time and tower-troop cases that
assumed a firing King may now fail — if any does, read the failure before
editing it: a case that broke because the King no longer shoots is *correct* to
update; one that broke for another reason is a real regression.

- [ ] **Step 6: Commit**

```bash
git add include/entities/Tower.h include/core/GameManager.h tests/core/test_king_activation.cpp build_python/ClashRoyaleTests.vcxproj
git commit -m "$(cat <<'EOF'
The King Tower never slept; give it the real game's activation state

Real Clash Royale's King is dormant until it takes damage or loses a friendly
Princess Tower. This engine's fired from tick 0 -- long-standing, recorded as
UPSTREAM_REQUESTS item 3, and the 2026-08-20 sight fix made it bite harder by
widening the King's effective reach to 9.4 tiles: its share of the damage
against a lone Hog was 36.8%, all from a tower that should have been asleep.

Latching flag on Tower, gated at findTarget so there is one choke point. The
damage trigger tests the HP invariant rather than overriding one damage entry
point, because damage arrives through five of them. The Princess trigger records
the initial count instead of comparing against a hardcoded 2, so a King built on
a bare board in a unit test does not wake instantly.

GAMEPLAY-AFFECTING, and the largest of the four: defence gets materially weaker.
Every win rate earned before this is historical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Sight-range regression tests (Rule A)

Rule A already ships correctly. This task adds **no production change** — it pins
the catalog so a future edit cannot quietly break it, and proves Example 1
end-to-end.

**Files:**
- Modify: `tests/core/test_sight_range.cpp`

**Interfaces:**
- Consumes: `ArenaLayout` from Task 1.
- Produces: nothing.

- [ ] **Step 1: Append the tests**

```cpp
// The sight-range catalog, for the eight DEFAULT_DECK cards. These are the
// player-audited values (a Clash Royale community stats breakdown of a stat the
// card info screens do not show). Pinned by card id so a registry edit that
// changes one is a test failure and not a silent rebalance.
TEST_CASE("DEFAULT_DECK sight ranges match the catalog", "[sight][registry]") {
    struct Row { int cardId; const char* name; float sight; };
    const Row rows[] = {
        { 15, "Hog Rider",  9.5f },
        {  6, "Musketeer",  6.0f },
        { 25, "Cannon",     5.5f },
        { 40, "Ice Golem",  7.0f },
        { 24, "Skeletons",  5.5f },
        { 72, "Ice Spirit", 5.5f },
    };
    for (const auto& r : rows) {
        const auto* card = CardRegistry::getInstance().getCard(r.cardId);
        INFO(r.name << " (id " << r.cardId << ")");
        REQUIRE(card != nullptr);
        REQUIRE(card->stats.sightRange == Catch::Approx(r.sight));
    }
}

// THE invariant. Sight and attack range must be measured the same way --
// surface to surface -- so that effective sight >= effective attack range and
// nothing can ever attack what it cannot see. Two conventions for one geometric
// question is what let a Musketeer destroy a 3204 hp Princess Tower from 8
// tiles taking zero damage (measured 2026-08-20).
TEST_CASE("effective sight is never shorter than effective attack range",
          "[sight][invariant]") {
    Board board(ArenaLayout::WIDTH, ArenaLayout::HEIGHT);
    for (int cardId : CardRegistry::getInstance().getAllCardIds()) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (!card || card->isSpell) continue;
        INFO("card " << cardId << " " << card->name);
        REQUIRE(card->stats.sightRange >= card->stats.attackRange);
    }
}

// Example 1 from the fidelity brief, end to end: an enemy Hog Rider crossing the
// LEFT bridge ignores a Cannon placed on the RIGHT bridge, because the Cannon is
// outside the Hog's sight, and continues to the tower.
TEST_CASE("a Hog ignores a Cannon on the far bridge", "[sight][aggro][brief]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();

    const auto* hogCard = CardRegistry::getInstance().getCard(15);
    const auto* cannonCard = CardRegistry::getInstance().getCard(25);
    REQUIRE(hogCard);
    REQUIRE(cannonCard);

    hogCard->spawnEntity(ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::BRIDGE_Y, 1, board);
    cannonCard->spawnEntity(ArenaLayout::RIGHT_BRIDGE_X, ArenaLayout::BRIDGE_Y - 2.0f, 0, board);
    board.commitPendingEntities(0);

    std::shared_ptr<Entity> hog, cannon;
    for (const auto& e : board.getEntities()) {
        if (e->team == 1 && !e->isTower()) hog = e;
        if (e->team == 0 && e->isBuilding() && !e->isTower()) cannon = e;
    }
    REQUIRE(hog);
    REQUIRE(cannon);

    // The two bridges are 12.0 tiles apart; the Hog's effective sight to a
    // building is 9.5 + 0.4 + 1.0 = 10.9. So the Cannon is out of sight, and
    // must stay out of sight for the whole approach.
    REQUIRE(hog->position.distanceTo(cannon->position) > 10.9f);

    for (int t = 0; t < 120; ++t) game.step();

    // It went down its own lane rather than across to the Cannon.
    REQUIRE(hog->position.x < ArenaLayout::CENTER_X);
    REQUIRE(cannon->hp == cannon->hp);   // untouched by the Hog
    REQUIRE(hog->position.y < ArenaLayout::BRIDGE_Y);   // it crossed, heading for team 0
}
```

Add `#include "ArenaLayout.h"` and `#include "CardRegistry.h"` if absent.

- [ ] **Step 2: Run**

Run MSBuild, then `./build_python/Release/ClashRoyaleTests.exe "[sight]"`
Expected: **PASS**. If the catalog test fails, the registry disagrees with the
brief — report the exact card and value rather than editing the test.

- [ ] **Step 3: Commit**

```bash
git add tests/core/test_sight_range.cpp
git commit -m "$(cat <<'EOF'
Pin the sight-range catalog and the sight >= attack invariant

Rule A of the fidelity brief was already implemented -- 58 per-card
withSightRange values matching the player's catalog -- but nothing pinned them,
so a registry edit could rebalance the game silently. No production change.

Also pins the invariant the 2026-08-20 audit exists for: sight and attack range
measured the same way, so effective sight >= effective attack range and nothing
can attack what it cannot see. And the brief's Example 1 end to end.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: LanePath — the lane objective (Rule B, part 1)

**Files:**
- Create: `include/core/LanePath.h`
- Modify: `include/entities/CombatEntity.h` (`findTarget`, ~line 1172)
- Modify: `include/entities/BuildingTargeter.h` (`findTarget`, ~line 35)
- Test: `tests/core/test_lane_pathing.cpp` (create)

**Interfaces:**
- Consumes: `ArenaLayout` from Task 1.
- Produces:
  - `LanePath::laneObjective(const Board& board, int myTeam, const Vector2D& from) -> std::shared_ptr<Entity>`
  - `LanePath::approachPoint(const Board&, int myTeam, const Vector2D& from, const std::shared_ptr<Entity>& target) -> Vector2D` (Task 7 implements the body; declare and return `target->position` here).

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_lane_pathing.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "ArenaLayout.h"
#include "LanePath.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"

// Destroys team `team`'s Princess Tower on the given side and returns the King.
static std::shared_ptr<Entity> killPrincess(Board& board, int team, bool left) {
    std::shared_ptr<Entity> king;
    for (const auto& e : board.getEntities()) {
        if (!e->isTower() || e->team != team) continue;
        if (e->symbol == 'R') { king = e; continue; }
        bool isLeft = e->position.x < ArenaLayout::CENTER_X;
        if (isLeft == left) e->takeDamage(e->hp);
    }
    return king;
}

// Example 2 from the fidelity brief.
//
// ANCHOR MATTERS. With the left Princess dead, "closest tower" picks the King by
// coincidence from the bridge itself (15.23 vs 15.57 tiles) and only picks the
// WRONG tower further back. Measured crossover is y ~ 15.0:
//     (2.5, 12.0) -> right Princess 18.90, King 19.45   <- broken rule is wrong
//     (2.5, 14.0) -> right Princess 17.36, King 17.56   <- broken rule is wrong
//     (2.5, 16.5) -> right Princess 15.57, King 15.23   <- broken rule is RIGHT
// So this test anchors at y = 13.0, behind the bridge, where a tank is actually
// deployed and where the defect is real. A test written at the bridge mouth
// passes against unfixed code and proves nothing.
TEST_CASE("with its lane's Princess dead a unit targets the King, not the other lane",
          "[lane][aggro][brief]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);
    REQUIRE(king);
    board.cleanDeadEntities(0);

    Vector2D from{ ArenaLayout::LEFT_BRIDGE_X, 13.0f };

    // The premise: the naive rule really would pick the wrong tower here.
    std::shared_ptr<Entity> naiveClosest;
    float best = 1e9f;
    for (const auto& e : board.getEntities()) {
        if (!e->isTower() || e->team != 1 || !e->isAlive()) continue;
        float d = from.distanceTo(e->position);
        if (d < best) { best = d; naiveClosest = e; }
    }
    REQUIRE(naiveClosest->symbol == 'P');                     // the right-lane Princess
    REQUIRE(naiveClosest->position.x > ArenaLayout::CENTER_X);

    // The rule under test picks the King instead.
    auto objective = LanePath::laneObjective(board, /*myTeam=*/0, from);
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

TEST_CASE("with its lane's Princess alive that Princess is the objective",
          "[lane][aggro]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();

    Vector2D from{ ArenaLayout::LEFT_BRIDGE_X, 13.0f };
    auto objective = LanePath::laneObjective(board, 0, from);
    REQUIRE(objective);
    REQUIRE(objective->symbol == 'P');
    REQUIRE(objective->team == 1);
    REQUIRE(objective->position.x == Catch::Approx(ArenaLayout::LEFT_LANE_X));
}

TEST_CASE("the right lane mirrors the left exactly", "[lane][aggro]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/false);
    board.cleanDeadEntities(0);

    Vector2D from{ ArenaLayout::RIGHT_BRIDGE_X, 13.0f };
    auto objective = LanePath::laneObjective(board, 0, from);
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

TEST_CASE("both enemy Princesses dead leaves the King", "[lane][aggro]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    killPrincess(board, 1, true);
    auto king = killPrincess(board, 1, false);
    board.cleanDeadEntities(0);

    auto objective = LanePath::laneObjective(board, 0, Vector2D{ ArenaLayout::LEFT_BRIDGE_X, 13.0f });
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

// A building-targeter must not drift out of agreement with a troop about which
// tower is the lane objective -- they share the rule for exactly this reason.
TEST_CASE("an Ice Golem walking the left lane never crosses to the right",
          "[lane][pathing][brief]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    killPrincess(board, 1, /*left=*/true);
    board.cleanDeadEntities(0);

    const auto* iceGolem = CardRegistry::getInstance().getCard(40);
    REQUIRE(iceGolem);
    iceGolem->spawnEntity(ArenaLayout::LEFT_BRIDGE_X, 13.0f, 0, board);
    board.commitPendingEntities(0);

    std::shared_ptr<Entity> golem;
    for (const auto& e : board.getEntities())
        if (e->team == 0 && !e->isTower() && !e->isBuilding()) golem = e;
    REQUIRE(golem);

    float maxX = golem->position.x;
    for (int t = 0; t < 600 && golem->isAlive(); ++t) {
        game.step();
        maxX = std::max(maxX, golem->position.x);
        // It must never wander into the right lane on its way to the King.
        INFO("tick " << t << " at (" << golem->position.x << ", " << golem->position.y << ")");
        REQUIRE(golem->position.x <= ArenaLayout::CENTER_X + 0.5f);
    }
    // And it must have actually gone somewhere upfield.
    REQUIRE(golem->position.y > ArenaLayout::BRIDGE_Y);
}
```

- [ ] **Step 2: Run to verify it fails**

Run MSBuild.
Expected: **compile error**, `cannot open include file: 'LanePath.h'`.

- [ ] **Step 3: Implement LanePath**

Create `include/core/LanePath.h`:

```cpp
#pragma once
#include <memory>
#include <cmath>
#include "ArenaLayout.h"
#include "Board.h"
#include "Entity.h"

// The "dirt paths". A unit with nothing inside its own sight range is BLIND, and
// a blind unit does not beeline for whatever tower happens to be nearest -- it
// walks its own lane toward the enemy base.
//
// Before this, findTarget's blind fallback was "closest enemy Tower by raw
// distance". With one enemy Princess destroyed that sends a unit diagonally
// across the arena to the OTHER lane's Princess. Measured on the corrected
// board, left Princess dead, a unit in the left lane:
//     at (2.5, 12.0)  right Princess 18.90   King 19.45   -> wrong tower
//     at (2.5, 14.0)  right Princess 17.36   King 17.56   -> wrong tower
//     at (2.5, 16.5)  right Princess 15.57   King 15.23   -> right, by luck
//
// Deliberately identified through Entity::isTower()/symbol rather than by
// including Tower.h: Tower.h -> Building.h -> CombatEntity.h, and CombatEntity
// is one of this header's consumers.
namespace LanePath {

// The tower a blind unit on `myTeam` should walk to from `from`: its OWN lane's
// enemy Princess Tower if that is still standing, else the enemy King.
//
// Lane is nearest-bridge, re-evaluated every call -- the same rule
// Board::getNextWaypoint uses to choose a crossing, so the objective and the
// crossing agree by construction and a unit can never be routed to one bridge
// while aiming at the other lane's tower. Nothing is stored on the entity, so
// Board::deepCopy and Tower::snapshot need no new field.
inline std::shared_ptr<Entity> laneObjective(const Board& board, int myTeam,
                                             const Vector2D& from) {
    const bool wantLeft = ArenaLayout::isLeftLane(from.x);
    std::shared_ptr<Entity> lanePrincess, king;

    for (const auto& e : board.getEntities()) {
        if (!e->isAlive() || !e->isTower() || e->team == myTeam) continue;
        if (e->symbol == 'R') { king = e; continue; }
        if (ArenaLayout::isLeftLane(e->position.x) == wantLeft) lanePrincess = e;
    }
    return lanePrincess ? lanePrincess : king;
}

// Where to WALK toward while approaching `target`. Task 7 fills this in.
inline Vector2D approachPoint(const Board& board, int myTeam, const Vector2D& from,
                              const std::shared_ptr<Entity>& target) {
    (void)board; (void)myTeam; (void)from;
    return target->position;
}

}  // namespace LanePath
```

- [ ] **Step 4: Wire it into both findTarget implementations**

In `include/entities/CombatEntity.h`, add `#include "LanePath.h"` near the top,
then in `findTarget` replace the `return closestInSight ? closestInSight : closestTower;`
tail with:

```cpp
        if (closestInSight) return closestInSight;

        // Blind: nothing inside sight range. Walk our OWN lane's objective
        // rather than whatever tower is nearest -- see LanePath.h. Falls back to
        // the old closest-tower answer if the lane objective is not a legal
        // target for this attacker (a Mortar's minAttackRange blind spot, an
        // air-only restriction), so no existing edge case changes behaviour.
        auto laneTarget = LanePath::laneObjective(board, team, position);
        if (laneTarget && isValidTarget(laneTarget)) return laneTarget;
        return closestTower;
```

In `include/entities/BuildingTargeter.h`, add `#include "LanePath.h"` and make
the identical replacement at the end of its `findTarget`. Its validity check is
its own loop's filter, so use:

```cpp
        if (closestInSight) return closestInSight;
        auto laneTarget = LanePath::laneObjective(board, team, position);
        if (laneTarget && laneTarget->isBuilding()) return laneTarget;
        return closestTower;
```

- [ ] **Step 5: Run to verify it passes**

Run: `./build_python/Release/ClashRoyaleTests.exe "[lane]"`
Expected: **PASS**, 5 cases.

- [ ] **Step 6: Run the full suite**

Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add include/core/LanePath.h include/entities/CombatEntity.h include/entities/BuildingTargeter.h tests/core/test_lane_pathing.cpp build_python/ClashRoyaleTests.vcxproj
git commit -m "$(cat <<'EOF'
A blind unit walked to the nearest tower, not down its own lane

findTarget's fallback with nothing in sight was "closest enemy Tower by raw
distance". With one enemy Princess destroyed that sends a unit diagonally across
the arena to the other lane's Princess instead of up its own lane to the King --
the fidelity brief's Example 2.

The reproduction is position-dependent and that shaped the test: from the bridge
mouth the broken rule already answers correctly by coincidence (King 15.23 vs
Princess 15.57), and only goes wrong further back, crossover at y ~ 15.0. The
regression test anchors at y = 13.0, where a tank is actually deployed.

Lane is nearest-bridge, re-evaluated per tick -- the same rule getNextWaypoint
uses to choose a crossing, so the two agree by construction and no per-entity
state needs snapshotting. Shared by CombatEntity and BuildingTargeter so a
building-targeter cannot drift out of agreement with a troop.

GAMEPLAY-AFFECTING.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: LanePath — the approach curve (Rule B, part 2)

With the lane's Princess dead, the objective is the King — but a straight line
from the bridge to the King cuts diagonally inward immediately. The real dirt
path runs **up the lane first**, then angles in.

**Files:**
- Modify: `include/core/LanePath.h` (`approachPoint`)
- Modify: `include/entities/CombatEntity.h:1068` (the single `moveTowards` call site)
- Modify: `tests/core/test_lane_pathing.cpp`

**Interfaces:**
- Consumes: `LanePath::laneObjective` from Task 6.
- Produces: a working `LanePath::approachPoint` with the signature declared in Task 6.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_lane_pathing.cpp`:

```cpp
// The curve. Heading for the King with its own lane's Princess dead, a unit must
// travel UP ITS LANE to the empty Princess slot before angling inward -- not cut
// the corner from the bridge. Sampled: while it is still short of the Princess
// row, it stays near its lane column.
TEST_CASE("the approach to the King curves up the lane before angling in",
          "[lane][pathing]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    killPrincess(board, 1, /*left=*/true);
    board.cleanDeadEntities(0);

    const auto* iceGolem = CardRegistry::getInstance().getCard(40);
    iceGolem->spawnEntity(ArenaLayout::LEFT_BRIDGE_X, 13.0f, 0, board);
    board.commitPendingEntities(0);
    std::shared_ptr<Entity> golem;
    for (const auto& e : board.getEntities())
        if (e->team == 0 && !e->isTower() && !e->isBuilding()) golem = e;
    REQUIRE(golem);

    bool sawPastRiver = false;
    for (int t = 0; t < 900 && golem->isAlive(); ++t) {
        game.step();
        float y = golem->position.y;
        if (y > ArenaLayout::BRIDGE_Y && y < ArenaLayout::princessY(1) - 1.0f) {
            sawPastRiver = true;
            // Still short of the Princess row: hug the lane, do not cut inward.
            INFO("tick " << t << " at (" << golem->position.x << ", " << y << ")");
            REQUIRE(golem->position.x < ArenaLayout::LEFT_LANE_X + 2.0f);
        }
    }
    REQUIRE(sawPastRiver);
}

// THE ABSORBING-STATE GUARD. Two absorbing states have already shipped in this
// engine, both from a waypoint being handed back to a mover already standing on
// it. approachPoint introduces a new intermediate waypoint, so it gets the same
// discipline: standing exactly on W1 must yield something else.
TEST_CASE("approachPoint never returns a point the mover already occupies",
          "[lane][pathing][regression]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);
    board.cleanDeadEntities(0);

    // Sweep the whole lane at finer-than-epsilon resolution, including exactly
    // on the intermediate waypoint.
    for (float y = 12.0f; y <= 31.0f; y += 0.005f) {
        Vector2D from{ ArenaLayout::LEFT_LANE_X, y };
        Vector2D wp = LanePath::approachPoint(board, 0, from, king);
        INFO("from y=" << y << " -> (" << wp.x << ", " << wp.y << ")");
        REQUIRE(from.distanceTo(wp) > Board::WAYPOINT_ARRIVAL_EPS);
    }
    // And exactly on the waypoint itself.
    Vector2D onW1{ ArenaLayout::LEFT_LANE_X, ArenaLayout::princessY(1) };
    REQUIRE(onW1.distanceTo(LanePath::approachPoint(board, 0, onW1, king))
            > Board::WAYPOINT_ARRIVAL_EPS);
}

// A tower objective in the unit's own lane is approached directly -- the curve
// must be a no-op when the Princess is alive, so this changes nothing for the
// overwhelmingly common case.
TEST_CASE("approachPoint is the identity for a live lane Princess", "[lane][pathing]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();
    Vector2D from{ ArenaLayout::LEFT_BRIDGE_X, 13.0f };
    auto objective = LanePath::laneObjective(board, 0, from);
    REQUIRE(objective->symbol == 'P');
    REQUIRE(LanePath::approachPoint(board, 0, from, objective).x
            == Catch::Approx(objective->position.x));
    REQUIRE(LanePath::approachPoint(board, 0, from, objective).y
            == Catch::Approx(objective->position.y));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./build_python/Release/ClashRoyaleTests.exe "[lane][pathing]"`
Expected: the curve case FAILS (the unit cuts inward from the bridge); the
identity case passes with the Task 6 stub.

- [ ] **Step 3: Implement approachPoint**

Replace the stub in `include/core/LanePath.h`:

```cpp
// Where to WALK toward while approaching `target`.
//
// For every target except an enemy KING this is just the target's position, so
// the overwhelmingly common case is untouched. Heading for the King, the unit
// first walks up its own lane to W1 -- the lane's Princess slot, empty by
// definition, since a live Princess would have BEEN the objective -- and only
// then angles in. That is the dirt path's curve; a straight line from the bridge
// would cut the corner across the arena.
//
//   W0 = bridge(lane)                            (2.5 | 14.5, 16.5)
//   W1 = (laneX(lane), princessY(enemy team))     (3.0 | 14.0, 27.0 | 6.0)
//   W2 = the King itself                          (8.5, 30.5 | 2.5)
//
// ABSORBING-STATE DISCIPLINE. This is a new intermediate waypoint, and two
// absorbing states have already shipped in this engine from a planner handing a
// mover the point it already stands on (Troop::moveTowards refuses to move
// inside Board::WAYPOINT_ARRIVAL_EPS, so the position never changes, so the
// waypoint never changes). W1 is therefore released as soon as the unit is
// within the epsilon of it OR has passed its row -- never handed back to
// someone standing on it. See test_lane_pathing.cpp's sweep.
inline Vector2D approachPoint(const Board& board, int myTeam, const Vector2D& from,
                              const std::shared_ptr<Entity>& target) {
    if (!target) return from;
    if (!target->isTower() || target->symbol != 'R') return target->position;

    const int enemyTeam = 1 - myTeam;
    const float w1x = ArenaLayout::laneXFor(from.x);
    const float w1y = ArenaLayout::princessY(enemyTeam);
    const Vector2D w1{ w1x, w1y };

    // Advancing toward the enemy means increasing y for team 0, decreasing for
    // team 1. Once level with the Princess row the lane leg is done.
    const bool advancingUp = (enemyTeam == 1);
    const bool reachedRow = advancingUp ? (from.y >= w1y) : (from.y <= w1y);
    if (reachedRow) return target->position;
    if (from.distanceTo(w1) <= Board::WAYPOINT_ARRIVAL_EPS) return target->position;
    return w1;
}
```

- [ ] **Step 4: Use it at the one call site**

In `include/entities/CombatEntity.h`, line ~1068, replace:
```cpp
                moveTowards(board, target->position);
```
with:
```cpp
                // Lane-aware approach: a King objective is walked to up this
                // unit's own lane, not cut across the arena. Identity for every
                // other target -- see LanePath::approachPoint.
                moveTowards(board, LanePath::approachPoint(board, team, position, target));
```

- [ ] **Step 5: Run to verify it passes**

Run: `./build_python/Release/ClashRoyaleTests.exe "[lane]"`
Expected: **PASS**, 8 cases.

- [ ] **Step 6: Run the full suite**

Expected: runner exits 0.

- [ ] **Step 7: Commit**

```bash
git add include/core/LanePath.h include/entities/CombatEntity.h tests/core/test_lane_pathing.cpp
git commit -m "$(cat <<'EOF'
Walking to the King cut the corner; follow the lane, then angle in

With its own lane's Princess destroyed a unit's objective is the King, but a
straight line from the bridge cuts diagonally inward immediately. The real dirt
path runs up the lane to the Princess slot first and only then angles in.

approachPoint is the identity for every target except an enemy King, so the
common case is untouched, and it is applied at the single moveTowards call site.

Carries the absorbing-state discipline this engine has twice paid for: W1 is
released as soon as the mover is within Board::WAYPOINT_ARRIVAL_EPS of it or has
passed its row, so it is never handed back to a unit already standing on it.
Pinned by a sweep of the whole lane at finer-than-epsilon resolution.

GAMEPLAY-AFFECTING.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Prove no new absorbing state, with instruments

End-state assertions cannot see a stall; both shipped absorbing states were
properties of a *trajectory*. The audit instruments are the real proof.

**Files:**
- Modify: `tools/audit/waypoint_probe.cpp` (extend to lane waypoints)
- Run: `tools/audit/bridge_audit.cpp`, `tools/audit/soak.cpp`

**Interfaces:**
- Consumes: `LanePath::approachPoint` from Task 7.
- Produces: recorded measurements for Task 12's CLAUDE.md entry.

- [ ] **Step 1: Extend the analytic probe**

In `tools/audit/waypoint_probe.cpp`, after the existing `getNextWaypoint` sweep,
add a second sweep composing the two planners the way the engine does:

```cpp
    // The engine composes LanePath::approachPoint with Board::getNextWaypoint.
    // Either alone can be absorbing-free while the COMPOSITION is not, so sweep
    // the composition, which is what a unit actually follows.
    std::size_t laneChecked = 0, laneAbsorbing = 0;
    for (float x = 0.0f; x <= 17.0f; x += 0.05f) {
        for (float y = 0.0f; y <= 33.0f; y += 0.05f) {
            Vector2D from{ x, y };
            for (int myTeam = 0; myTeam < 2; ++myTeam) {
                auto objective = LanePath::laneObjective(board, myTeam, from);
                if (!objective) continue;
                Vector2D approach = LanePath::approachPoint(board, myTeam, from, objective);
                Vector2D wp = board.getNextWaypoint(from, approach);
                ++laneChecked;
                if (from.distanceTo(wp) <= Board::WAYPOINT_ARRIVAL_EPS &&
                    from.distanceTo(objective->position) > Board::WAYPOINT_ARRIVAL_EPS) {
                    ++laneAbsorbing;
                    std::printf("ABSORBING lane state at (%.3f, %.3f) team %d\n", x, y, myTeam);
                }
            }
        }
    }
    std::printf("lane composition: %zu positions checked, %zu absorbing\n",
                laneChecked, laneAbsorbing);
```

Add `#include "LanePath.h"`. Build a `GameManager` inside the probe so the board
carries real towers, and run the sweep twice — once with all towers alive, once
with each enemy Princess destroyed in turn (the state that activates the curve).

- [ ] **Step 2: Build and run all three instruments**

```
powershell -File tools/audit/build.ps1 waypoint_probe
powershell -File tools/audit/build.ps1 bridge_audit
powershell -File tools/audit/build.ps1 soak
./tools/audit/bin/waypoint_probe.exe
./tools/audit/bin/bridge_audit.exe
./tools/audit/bin/soak.exe
```

Bars, all of which must hold:

| instrument | bar |
|---|---|
| `waypoint_probe` | **0** absorbing states, both the original sweep and the new lane composition |
| `bridge_audit` | **34/34** crossings for every card; longest stall == `DEPLOY_TIME_TICKS` (10, or 9 for Skeletons/Minions) |
| `soak` | stall rate no worse than 7 per 342,563 unit-ticks, and **none on or near a bridge** |

If `waypoint_probe` reports any absorbing state, STOP and fix it before
continuing — that is the exact defect class this plan is most at risk of
reintroducing. If `bridge_audit` shows any card below 34/34, the pathing change
broke crossing; do not proceed.

- [ ] **Step 3: Commit**

```bash
git add tools/audit/waypoint_probe.cpp
git commit -m "$(cat <<'EOF'
Sweep the lane planner for absorbing states, not just the bridge one

LanePath::approachPoint and Board::getNextWaypoint are composed at runtime, and
either can be absorbing-free while the composition is not. The probe now sweeps
what a unit actually follows, with all towers alive and with each enemy Princess
destroyed in turn -- the state that activates the lane curve.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: The timeout verdict, in replays and the viewer

The rule is correct. The consumer is not, and it survived every previous cleanup
because it is JavaScript and the guard test walks `.py` only.

**Files:**
- Modify: `include/core/GameLogger.h` (add a `"result"` object)
- Modify: `web/viewer.html:800-815` (`computeMatchResult`)
- Modify: `python_ai/tests/test_match_outcome_is_the_only_scorer.py`
- Modify: `tests/core/test_timeout_rules.cpp`

**Interfaces:**
- Consumes: `TimeoutRules::resolve`, `MatchRules::evaluate` (both existing).
- Produces: replay JSON key `result: { loserTeam: int, reason: string, timedOut: bool }`.

- [ ] **Step 1: Write the failing C++ test**

Append to `tests/core/test_timeout_rules.cpp`:

```cpp
// The replay JSON must carry the ENGINE's verdict. Every consumer that
// re-derived one from tower counts got it wrong -- eight times across three
// waves on the Python side, and web/viewer.html was still doing it in 2026-08-21
// because the guard test only walks .py files.
TEST_CASE("GameLogger writes the engine's own verdict into the replay",
          "[timeout][replay]") {
    GameManager game({0,1,2,3,4,5,6,7}, {0,1,2,3,4,5,6,7});
    Board& board = game.getBoard();

    // Team 0 ahead on towers: destroy one of team 1's Princesses.
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 1 && e->symbol != 'R') { e->takeDamage(e->hp); break; }
    }
    board.cleanDeadEntities(0);

    GameLogger logger;
    logger.logTick(1, game);
    std::string json = logger.resultJson(game.getBoard());

    REQUIRE(json.find("\"loserTeam\": 1") != std::string::npos);
    REQUIRE(json.find("\"timedOut\"") != std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: **compile error**, `resultJson` is not a member of `GameLogger`.

- [ ] **Step 3: Implement in GameLogger**

Add to `include/core/GameLogger.h` (include `TimeoutRules.h` and `MatchRules.h`):

```cpp
    // The match verdict, written into the replay so consumers READ it instead of
    // re-deriving it. web/viewer.html derived its own from "are both Kings
    // alive?" -- which mirrors MatchRules::evaluate exactly, and that is the bug:
    // MatchRules answers "has a King died yet?", is called every tick, and
    // correctly says "not over" right up to the limit. "Who won?" at the limit is
    // TimeoutRules::resolve, which nothing on the JS side ever asked. A match
    // ending 3-3 on towers with a Princess at 90 hp was reported as a draw.
    std::string resultJson(const Board& board) const {
        MatchRules::Outcome byKing = MatchRules::evaluate(board);
        bool timedOut = !byKing.over;
        MatchRules::Outcome outcome = timedOut ? TimeoutRules::resolve(board) : byKing;

        std::string reason;
        if (!timedOut) {
            reason = (outcome.loserTeam == -1) ? "Both King Towers fell the same tick"
                   : (outcome.loserTeam == 0)  ? "Blue's King Tower destroyed"
                                               : "Red's King Tower destroyed";
        } else if (outcome.loserTeam == -1) {
            reason = "Timeout - exact tie on towers and weakest-tower HP";
        } else {
            reason = "Timeout - decided on towers, then the weakest tower's HP";
        }

        std::ostringstream out;
        out << "{\"loserTeam\": " << outcome.loserTeam
            << ", \"timedOut\": " << (timedOut ? "true" : "false")
            << ", \"reason\": \"" << reason << "\"}";
        return out.str();
    }
```

Then in `writeToFile`, emit it before `"ticks"`:
```cpp
        file << "  \"result\": " << resultJson(finalBoard) << ",\n";
```
`writeToFile` must receive the final `Board`. If it does not already, add a
`const Board&` parameter and update its call sites (`ClashEnv` and any tool);
compile errors will name them all.

- [ ] **Step 4: Fix the viewer**

In `web/viewer.html`, replace `computeMatchResult` entirely:

```js
// Reads the verdict the ENGINE wrote into the replay. It does not re-derive one.
//
// The previous version checked only whether both King Towers were alive and
// called anything else a draw -- its comment said it "mirrors MatchRules::
// evaluate exactly", which was true and was the bug: MatchRules answers "has a
// King died yet?" and correctly says no right up to the tick limit. Who WON at
// the limit is TimeoutRules: fewer surviving towers loses; on equal counts the
// side whose weakest surviving tower has the lower ABSOLUTE hp loses; only an
// exact tie on both is a draw. A match with a badly damaged Princess Tower was
// being reported as a draw.
function computeMatchResult() {
  if (!gameData || !gameData.ticks.length) return null;

  if (gameData.result) {
    const r = gameData.result;
    if (r.loserTeam === -1) return { winner: 'draw', reason: r.reason };
    return { winner: r.loserTeam === 0 ? 'red' : 'blue', reason: r.reason };
  }

  // Replays written before the engine emitted `result`. Port of TimeoutRules,
  // NOT the old king-alive-only shortcut.
  const finalTick = gameData.ticks[gameData.ticks.length - 1];
  const towers = finalTick.entities.filter(e => e.symbol === 'R' || e.symbol === 'P');
  const kingAlive = [0, 1].map(t => towers.some(e => e.symbol === 'R' && e.team === t));
  if (!kingAlive[0] || !kingAlive[1]) {
    if (!kingAlive[0] && !kingAlive[1])
      return { winner: 'draw', reason: 'Both King Towers fell the same tick' };
    return {
      winner: kingAlive[0] ? 'blue' : 'red',
      reason: kingAlive[0] ? "Red's King Tower destroyed" : "Blue's King Tower destroyed"
    };
  }

  const count = [0, 1].map(t => towers.filter(e => e.team === t).length);
  if (count[0] !== count[1]) {
    return {
      winner: count[0] > count[1] ? 'blue' : 'red',
      reason: 'Timeout - decided on surviving tower count'
    };
  }
  const weakest = [0, 1].map(t =>
    Math.min(...towers.filter(e => e.team === t).map(e => e.hp)));
  if (weakest[0] !== weakest[1]) {
    return {
      winner: weakest[0] > weakest[1] ? 'blue' : 'red',
      reason: 'Timeout - decided on the weakest tower\'s HP'
    };
  }
  return { winner: 'draw', reason: 'Timeout - exact tie on towers and weakest-tower HP' };
}
```

- [ ] **Step 5: Extend the guard past Python**

In `python_ai/tests/test_match_outcome_is_the_only_scorer.py`, add:

```python
# The Python guard below could never have caught web/viewer.html, which scored a
# match from "are both Kings alive?" until 2026-08-21 -- it walks .py only. The
# pattern is language-independent, so the guard must be too.
JS_KING_ONLY = re.compile(
    r"symbol\s*===?\s*['\"]R['\"].*\n?.*?winner:\s*['\"]draw['\"]",
    re.MULTILINE,
)


def test_the_replay_viewer_reads_the_engines_verdict():
    viewer = os.path.join(python_ai.REPO_ROOT, "web", "viewer.html")
    text = open(viewer, encoding="utf-8", errors="replace").read()
    assert "gameData.result" in text, (
        "web/viewer.html must consume the engine's own `result` object rather "
        "than re-deriving a verdict. See GameLogger::resultJson."
    )
    assert not JS_KING_ONLY.search(text), (
        "web/viewer.html scores a timeout from King-alive alone, which ignores "
        "TimeoutRules' tower-count and weakest-tower tie-breaks and reports a "
        "draw for matches the engine calls a win."
    )
```

- [ ] **Step 6: Run everything**

```
./build_python/Release/ClashRoyaleTests.exe "[timeout]"
python_ai/venv/Scripts/python.exe -m pytest python_ai/tests/test_match_outcome_is_the_only_scorer.py -q
```
Expected: both PASS.

- [ ] **Step 7: Regenerate a replay and eyeball the viewer**

```
python_ai/venv/Scripts/python.exe python_ai/tools/make_replays.py
```
Open `web/viewer.html` on the produced replay and confirm a timed-out match with
a damaged Princess Tower now reports a winner, not a draw.

- [ ] **Step 8: Commit**

```bash
git add include/core/GameLogger.h web/viewer.html python_ai/tests/test_match_outcome_is_the_only_scorer.py tests/core/test_timeout_rules.cpp
git commit -m "$(cat <<'EOF'
The replay viewer scored timeouts from King-alive alone

TimeoutRules has been correct since it was written and is wired into the reward
path -- measured: both sides passive reaches tick 3600 and resolves, vs the
heuristic 0 of 6 matches even reach the limit. The DEFECT was in a consumer.

web/viewer.html re-derived the verdict from "are both Kings alive?" and called
everything else a draw. Its comment said it mirrors MatchRules::evaluate exactly,
which is true and is the bug: MatchRules answers "has a King died yet?" and
correctly says no right up to the limit. Who won at the limit is TimeoutRules.
A match with a Princess Tower at 90 hp was shown as "Draw. Timeout - both King
Towers still standing".

GameLogger now writes the engine's verdict into the replay and the viewer reads
it, with a client-side TimeoutRules port for older replays. The Python guard that
was meant to prevent exactly this walks .py files only, which is why it never saw
a .html file; it now checks the viewer directly.

Not gameplay-affecting -- this changes only what a replay reports.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Reconcile the downstream copies of the moved coordinates

**Files:**
- Modify: `include/core/HeuristicOpponent.h:121-122`
- Modify: `perception/geometry.py` (~lines 80–91)
- Modify: `python_ai/advisors/tactics.py` — only if it holds a bridge column; check with `grep -n "3\.5\|13\.5\|bridge_x\|BRIDGE" python_ai/advisors/tactics.py`

**Interfaces:**
- Consumes: `ArenaLayout`, `Board::getLeftBridge()/getRightBridge()`.
- Produces: nothing.

- [ ] **Step 1: Point HeuristicOpponent at the board**

Replace:
```cpp
    static constexpr float LEFT_BRIDGE_X = 3.5f;
    static constexpr float RIGHT_BRIDGE_X = 13.5f;
```
with:
```cpp
    // Read from ArenaLayout, not restated. These were 3.5/13.5 and had ALREADY
    // gone stale against Board's own 4.0/14.0 before the arena was corrected --
    // the exact failure mode CLAUDE.md's no-second-copies rule exists for.
    static constexpr float LEFT_BRIDGE_X = ArenaLayout::LEFT_BRIDGE_X;
    static constexpr float RIGHT_BRIDGE_X = ArenaLayout::RIGHT_BRIDGE_X;
```
Add `#include "ArenaLayout.h"`.

- [ ] **Step 2: Update perception's geometry**

In `perception/geometry.py`, update the constants and their comments:
```python
# Corrected 2026-08-21 with the engine (include/core/ArenaLayout.h). x is a cell
# index in [0, 17], so the board centre is 8.5, and a two-tile bridge is centred
# on the SEAM between its tiles. Not derivable from the bindings -- no binding
# exposes board entity positions -- so pinned here by the header it comes from.
LEFT_BRIDGE = (2.5, 16.5)
RIGHT_BRIDGE = (14.5, 16.5)

# Princess towers are 3 wide; the bridge covers their two OUTER columns, so the
# tower centre sits half a tile inboard of its bridge.
OWN_PRINCESS_LEFT = (3.0, 6.0)
OWN_PRINCESS_RIGHT = (14.0, 6.0)
OPP_PRINCESS_LEFT = (3.0, 27.0)
OPP_PRINCESS_RIGHT = (14.0, 27.0)
```
Also update `OWN_KING` / `OPP_KING` x from 9.0 to **8.5** (grep for them in the
same file).

- [ ] **Step 3: Run the perception suite**

```
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```
Expected: 353 passed / 1 skipped. Any calibration test that fails is asserting
the OLD geometry — read it before editing; if it encodes a landmark fit, its
expected values move with the towers and that is correct.

- [ ] **Step 4: Rebuild and run the C++ suite**

Expected: runner exits 0.

- [ ] **Step 5: Commit**

```bash
git add include/core/HeuristicOpponent.h perception/geometry.py python_ai/advisors/tactics.py
git commit -m "$(cat <<'EOF'
Reconcile the three downstream copies of the arena's coordinates

HeuristicOpponent's LEFT_BRIDGE_X/RIGHT_BRIDGE_X were 3.5/13.5 and had already
gone stale against Board's own values before the arena moved -- they now read
ArenaLayout. perception/geometry.py's bridge, Princess and King positions follow
the corrected layout, pinned by the header they come from since no binding
exposes board entity positions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Rebuild the `.pyd` and verify the Python side

**Files:**
- Modify: none (build artefact only)
- Run: `tools/audit/verify_pyd.py`, the Python suite

**Interfaces:**
- Consumes: every engine change from Tasks 1, 4, 6, 7, 9.
- Produces: a `python_ai/clash_royale_env.pyd` that actually carries this engine.

- [ ] **Step 1: Confirm nothing holds the `.pyd`**

```
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,CommandLine | Format-List
```
If anything is listed, the post-build copy will fail with MSB3073 — which reads
like a broken compile and is not. A foreign Claude session was running
`defence_gradient.py` at plan time: **ask the user before killing it.**

- [ ] **Step 2: Rebuild**

```
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" build_python\clash_royale_env.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

- [ ] **Step 3: Prove the `.pyd` carries this engine**

```
python_ai/venv/Scripts/python.exe tools/audit/verify_pyd.py
```
This exists because the C++ suite cannot tell you whether the `.pyd` was actually
replaced — it has gone stale silently twice. Expected: PASS.

- [ ] **Step 4: Add a coordinate check to the gate**

Extend `tools/audit/verify_pyd.py` so a stale `.pyd` cannot pass by accident:

```python
# The arena moved on 2026-08-21 (include/core/ArenaLayout.h). A .pyd built before
# that reports the old tower columns, which is the cheapest possible proof that
# the post-build copy actually happened.
obs = env.reset()
xs = sorted({round(e_x, 2) for e_x in _tower_xs(env)})
assert xs == [3.0, 8.5, 14.0], f"stale .pyd: tower columns are {xs}, expected [3.0, 8.5, 14.0]"
```
Implement `_tower_xs` using whatever entity accessor the bindings expose; if none
does, read the six tower HPs' layout is not enough — instead assert via
`env.get_max_placement_x()` / an existing bound geometry accessor, or add a
minimal read-only binding. Prefer an existing accessor.

- [ ] **Step 5: Run the Python suite**

```
python_ai/venv/Scripts/python.exe -m pytest python_ai/tests -q
```
Expected: 366 passed / 2 skipped, allowing for the two cases that depend on the
unseeded opening-hand shuffle. Any failure asserting old coordinates is a test to
update; any other failure is a real regression.

- [ ] **Step 6: Commit**

```bash
git add python_ai/clash_royale_env.pyd tools/audit/verify_pyd.py python_ai/tests
git commit -m "$(cat <<'EOF'
Rebuild the .pyd and make its gate notice a moved arena

The .pyd has gone stale silently twice, and the C++ suite cannot detect it. The
post-rebuild gate now also checks the tower columns, so a .pyd predating the
arena correction fails loudly instead of quietly running the old geometry.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Record it in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `perception/UPSTREAM_REQUESTS.md` (close item 3)

**Interfaces:**
- Consumes: the measurements recorded in Tasks 3 and 8.
- Produces: nothing.

- [ ] **Step 1: Add the section**

Add a `## 2026-08-21: four fidelity fixes from a player audit` section covering,
with the ACTUAL measured numbers from Tasks 3 and 8 (do not write a number you
did not measure):

- the arena correction, with the river-row map and why 8.5 is the centre
- King dormancy, and that it is the largest gameplay change of the four
- lane pathing, with the position-dependence table for Example 2
- the viewer verdict bug and why the Python guard could not see it
- **GAMEPLAY-AFFECTING: every win rate in this file is now historical**;
  checkpoints still load
- the navigation-wedge outcome from Task 3
- new suite counts

- [ ] **Step 2: Correct the stale claims this work invalidates**

Search CLAUDE.md for and update:
- "Board geometry. River `[15.5, 17.5)`, bridges at `(4, 16.5)` and `(14, 16.5)`"
- "left Princess `x` 3.0 → 4.0 ... Kings `x` 8.5 → 9.0" — now reversed, with the reason
- "**The King Tower never sleeps.**" — now false; rewrite, and note that any
  comparison against real footage no longer needs to exclude the Kings for this
  reason
- the 2026-08-20 audit's King damage-share note (5.3% → 36.8%), which described
  a tower that is now dormant

- [ ] **Step 3: Close UPSTREAM item 3**

Mark `perception/UPSTREAM_REQUESTS.md` item 3 (the never-sleeping King) as
**Applied 2026-08-21**, in the same style as item 16's existing sign-off.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md perception/UPSTREAM_REQUESTS.md
git commit -m "$(cat <<'EOF'
Record the four fidelity fixes and retire the claims they invalidate

Every win rate in CLAUDE.md is now historical: the King sleeps, the arena moved,
and blind units follow their lane. Checkpoints still load -- no observation,
action-space or architecture change.

Also corrects the entries this work falsified, including "The King Tower never
sleeps" and the 2026-07-30 tower-coordinate move, which is now reversed with the
convention error that caused it stated.

Closes UPSTREAM_REQUESTS item 3.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §1 King activation | 4 |
| §2 coordinates | 1 (landed pre-plan, consolidated in 1) |
| §3 Rule A sight | 5 |
| §3 Rule B lane objective | 6 |
| §3 Rule B approach curve | 7 |
| §3 video validation | deferred — see Open Item below |
| §4 viewer / GameLogger / guard | 9 |
| §5 test repairs + instruments | 2, 3, 8 |
| §6 second copies | 10 |
| §6 invalidation notes | 12 |

**Open item carried deliberately:** the spec's hand-validation of the lane-path
shape against `perception/assets/recordings/` is **not** a task here. It is a
measurement, not an implementation step, it cannot fail the build, and its
outcome only adjusts constants inside `LanePath::approachPoint`. Do it after
Task 8 and report; if the real path differs, that is a one-constant follow-up.

**Type consistency:** `laneObjective` and `approachPoint` keep identical
signatures between Tasks 6 and 7. `isAwake()`/`wake()`/`sleep()` are used
consistently in Task 4. `ArenaLayout::princessY/kingY/laneXFor/isLeftLane` as
declared in Task 1 are the names used in Tasks 4, 6, 7 and 8.

**Risk ranking:** Task 7 is the highest-risk change (it alters movement for every
unit heading to a King) and Task 8 is its gate. Task 4 has the largest gameplay
effect. Task 2 is the one protecting the two already-shipped absorbing-state fixes.
