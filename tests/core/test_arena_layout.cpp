#include <catch_amalgamated.hpp>
#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"
#include "ClashEnv.h"
#include <string>
#include <cmath>

// The arena is symmetric about the CELL-INDEX centre (WIDTH-1)/2 = 8.5, which is
// the x-analogue of ClashEnv::extractObservationForTeam's y -> 33 - y. Every pair
// below must mirror onto the other under 17 - x, or one lane is playable
// differently from the other -- a class of bug this repo has paid for twice (the
// river's own off-centre position, and the team-1 observation row shift).

TEST_CASE("ArenaLayout is mirror-symmetric about the board centre", "[arena][geometry]") {
    REQUIRE(ArenaLayout::CENTER_X == Catch::Approx(8.5f));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::LEFT_LANE_X) == Catch::Approx(ArenaLayout::RIGHT_LANE_X));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::LEFT_BRIDGE_X) == Catch::Approx(ArenaLayout::RIGHT_BRIDGE_X));
    REQUIRE(ArenaLayout::mirrorX(ArenaLayout::CENTER_X) == Catch::Approx(ArenaLayout::CENTER_X));

    // Rows mirror too: team 0's King at 2.5 <-> team 1's at 30.5.
    REQUIRE(ArenaLayout::mirrorY(ArenaLayout::kingY(0)) == Catch::Approx(ArenaLayout::kingY(1)));
    REQUIRE(ArenaLayout::mirrorY(ArenaLayout::princessY(0)) == Catch::Approx(ArenaLayout::princessY(1)));
}

// The player's own map of the river row:
//
//     column  012345678901234567
//             WWBBWWWWWWWWWWBBWW      (W water, B bridge)
//
// Bridges occupy columns 2-3 and 14-15, so each CENTRE sits on the seam between
// its two tiles. Centring on a TILE instead is what made clampToBoard's
// +/-BRIDGE_HALF_WIDTH corridor three columns wide instead of two.
TEST_CASE("bridges are two tiles wide, centred on the seam", "[arena][geometry]") {
    REQUIRE(ArenaLayout::LEFT_BRIDGE_X == Catch::Approx(2.5f));
    REQUIRE(ArenaLayout::RIGHT_BRIDGE_X == Catch::Approx(14.5f));

    Board board;
    const float midRiver = 16.5f;
    for (int x = 0; x < ArenaLayout::WIDTH; ++x) {
        Vector2D probe{ static_cast<float>(x), midRiver };
        bool walkable = board.clampToBoard(probe, false).y == Catch::Approx(midRiver);
        bool expected = (x == 2 || x == 3 || x == 14 || x == 15);
        INFO("column " << x << " of WWBBWWWWWWWWWWBBWW");
        REQUIRE(walkable == expected);
    }
}

// GameManager must not carry its own copy of these numbers. Asserted by PROPERTY
// (every King on centre, every Princess on its lane column) rather than by index,
// so it does not also pin the order entities happen to be spawned in.
TEST_CASE("GameManager spawns its towers exactly where ArenaLayout says", "[arena][geometry]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
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

// Board's own bridge members must come from ArenaLayout, not a second literal.
// HeuristicOpponent kept its own copy (3.5/13.5) and it had ALREADY gone stale
// against Board's 4.0/14.0 before the arena was corrected -- exactly the failure
// CLAUDE.md's no-second-copies rule exists for.
TEST_CASE("Board's bridges are ArenaLayout's bridges", "[arena][geometry]") {
    Board board;
    REQUIRE(board.getLeftBridge().x == Catch::Approx(ArenaLayout::LEFT_BRIDGE_X));
    REQUIRE(board.getRightBridge().x == Catch::Approx(ArenaLayout::RIGHT_BRIDGE_X));
    REQUIRE(board.getLeftBridge().y == Catch::Approx(ArenaLayout::BRIDGE_Y));
    REQUIRE(board.getRightBridge().y == Catch::Approx(ArenaLayout::BRIDGE_Y));
}

// Each Princess Tower is 3 wide (Tower::getCollisionRadius 1.5) and its lane's
// bridge is 2 wide, so the bridge covers the tower's two OUTER columns and the
// tower centre sits half a tile INBOARD of the bridge centre. LanePath relies on
// this: a unit crossing at 2.5 has to curve half a tile inward to reach 3.0.
TEST_CASE("each bridge sits half a tile outboard of its own Princess Tower",
          "[arena][geometry]") {
    REQUIRE(ArenaLayout::LEFT_LANE_X - ArenaLayout::LEFT_BRIDGE_X == Catch::Approx(0.5f));
    REQUIRE(ArenaLayout::RIGHT_BRIDGE_X - ArenaLayout::RIGHT_LANE_X == Catch::Approx(0.5f));
}

// Lane selection is nearest-bridge, the same rule Board::getNextWaypoint already
// uses to choose a crossing -- so a unit's lane objective and the bridge it is
// routed over agree by construction and it can never be sent to one bridge while
// aiming at the other lane's tower.
TEST_CASE("lane selection splits at the board centre and matches the bridges",
          "[arena][geometry]") {
    REQUIRE(ArenaLayout::isLeftLane(0.0f));
    REQUIRE(ArenaLayout::isLeftLane(8.49f));
    REQUIRE_FALSE(ArenaLayout::isLeftLane(ArenaLayout::CENTER_X));
    REQUIRE_FALSE(ArenaLayout::isLeftLane(17.0f));

    REQUIRE(ArenaLayout::bridgeXFor(1.0f) == Catch::Approx(ArenaLayout::LEFT_BRIDGE_X));
    REQUIRE(ArenaLayout::bridgeXFor(16.0f) == Catch::Approx(ArenaLayout::RIGHT_BRIDGE_X));
    REQUIRE(ArenaLayout::laneXFor(1.0f) == Catch::Approx(ArenaLayout::LEFT_LANE_X));
    REQUIRE(ArenaLayout::laneXFor(16.0f) == Catch::Approx(ArenaLayout::RIGHT_LANE_X));
}

// ---------------- the observation must describe the PHYSICS ----------------
//
// Observation channel 8 is the river/bridge mask: the only thing telling the
// network where it can cross. It was painted from a hardcoded
// `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)` while the physics used
// Board's leftBridge/rightBridge, and when the arena was corrected on
// 2026-08-21 only the physics moved.
//
// The result was not a small offset. Of the four real bridge columns the
// network was told TWO were water (2 and 15), and it was told two water
// columns were bridge (4 and 13) -- so the agent's map of where it could cross
// was half wrong in BOTH directions, on every observation of every tick of
// every episode. Every C++ test still passed, because nothing compared this
// channel against the rule it is supposed to describe.
//
// These cases compare it against MOVEMENT, not against a literal. A test that
// pinned the expected columns would have to be hand-edited on the next arena
// change and would go stale exactly the way the encoder did.

TEST_CASE("the river mask marks a column passable iff a unit can actually stand there",
          "[arena][observation][regression]") {
    Board board;
    ClashEnv env({ 15, 6, 25, 40, 24, 72, 33, 7 }, { 15, 6, 25, 40, 24, 72, 33, 7 });
    env.reset();

    const int W = ClashEnv::BOARD_WIDTH;
    const int plane = W * ClashEnv::BOARD_HEIGHT;
    constexpr int riverRow = 17;
    const float midRiver = 16.5f;

    for (int team = 0; team < 2; ++team) {
        std::vector<float> obs = env.getObservationForTeam(team);
        for (int x = 0; x < W; ++x) {
            // Ground truth: can a non-river-ignoring unit hold this position
            // inside the river band? That is clampToBoard's own rule, i.e. the
            // physics the mask is meant to advertise.
            Vector2D probe{ static_cast<float>(x), midRiver };
            const bool passable = std::abs(board.clampToBoard(probe, false).y - midRiver) < 1e-4f;
            const float marked = obs[8 * plane + riverRow * W + x];

            INFO("team " << team << " column " << x
                 << ": physics says " << (passable ? "BRIDGE" : "water")
                 << ", observation says " << (marked > 0.0f ? "BRIDGE" : "water"));
            REQUIRE((marked > 0.0f) == passable);
        }
    }
}

TEST_CASE("both teams see the bridge mask on the same row and columns",
          "[arena][observation]") {
    ClashEnv env({ 15, 6, 25, 40, 24, 72, 33, 7 }, { 15, 6, 25, 40, 24, 72, 33, 7 });
    env.reset();
    const int W = ClashEnv::BOARD_WIDTH;
    const int plane = W * ClashEnv::BOARD_HEIGHT;
    constexpr int riverRow = 17;

    std::vector<float> a = env.getObservationForTeam(0);
    std::vector<float> b = env.getObservationForTeam(1);
    for (int x = 0; x < W; ++x) {
        INFO("column " << x);
        REQUIRE(a[8 * plane + riverRow * W + x] == b[8 * plane + riverRow * W + x]);
    }
}

// Board::isOnBridge is the shared definition. Pin that it selects exactly the
// player's river row, so a future edit to either caller cannot quietly widen it.
TEST_CASE("isOnBridge reproduces the real river row WWBBWWWWWWWWWWBBWW",
          "[arena][geometry]") {
    Board board;
    std::string row;
    for (int x = 0; x < ArenaLayout::WIDTH; ++x)
        row += board.isOnBridge(static_cast<float>(x)) ? 'B' : 'W';
    INFO("engine row: " << row);
    REQUIRE(row == "WWBBWWWWWWWWWWBBWW");
}
