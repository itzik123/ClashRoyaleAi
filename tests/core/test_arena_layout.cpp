#include <catch_amalgamated.hpp>
#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"

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
