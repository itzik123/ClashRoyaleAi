#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "Building.h"
#include "Tower.h"
#include "MeleeTroop.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include <vector>

// ---------------- the two-obstacle collision wedge ----------------
//
// Measured 2026-08-20 by tools/audit/soak.cpp over 60 randomized full matches
// (308,464 unit-ticks): 4 units stopped dead for 50+ consecutive ticks with
// nothing inside their own attack reach. Every one was pinched in the concave
// pocket between TWO adjacent buildings, and the dump of the first is
// reproduced exactly below:
//
//   Ice Golem  at (11.360, 2.939), walking north
//   King Tower at ( 9.000, 2.500) r=2.0  -> minDist 2.4, actual distance 2.401
//   Cannon     at (12.032, 4.169) r=1.0  -> minDist 1.4, actual distance 1.401
//
// Both constraints sit exactly on their boundary. The unit steps toward its
// waypoint, Board::resolveCollisions pushes it back out of whichever circle it
// entered, and the two perpendicular slides in Board::pushAwayFrom -- which
// exist precisely to stop a unit sticking on ONE obstacle -- point in opposing
// tangential directions and cancel. Net displacement is exactly zero, forever.
//
// This is NOT the bridge trap (see test_board.cpp): the geometry is a concave
// pocket between two obstacles, it can happen anywhere on the board, and no
// waypoint is involved. It is a local minimum in reactive steering.

namespace {

constexpr int ICE_GOLEM = 40;
constexpr int CANNON = 25;

// The measured configuration, through the real GameManager tick loop and the
// real cards. Reconstructing it from a bare Board did NOT reproduce, and that
// is informative rather than incidental: the trap is an ATTRACTING fixed
// point, not a knife edge, so it has to be entered by walking into its basin.
// The soak's trace shows the approach converging geometrically --
// y = 2.9071, 2.9244, 2.9323, 2.9359, 2.9376, 2.9385, 2.9389, 2.9391, 2.9392 --
// which is why this starts the unit slightly outside it and lets it fall in.
std::shared_ptr<Entity> spawnWedgedGolem(GameManager& game) {
    // GameManager::reset() already builds the King Tower at (9.0, 2.5) r=2.0.
    CardRegistry::getInstance().getCard(CANNON)->spawnEntity(12.032f, 4.169f, 0, game.getBoard());
    CardRegistry::getInstance().getCard(ICE_GOLEM)->spawnEntity(11.58f, 2.84f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->cardId == ICE_GOLEM) return e;
    }
    return nullptr;
}

} // namespace

// KNOWN DEFECT, PINNED DELIBERATELY. `[!shouldfail]` means Catch2 expects this
// to fail: the suite stays green while the bug is on record, and the moment
// somebody fixes it this case goes RED to say so. It is not skipped -- it runs
// every time, and it is the executable form of the write-up in
// perception/UPSTREAM_REQUESTS.md.
//
// NOT FIXED because two local repairs were tried and measured, and both merely
// MOVED the equilibrium:
//   1. tangential slide, handedness flipped on a timer  -> 0.027 tiles / 120
//      ticks (fifteen ticks of progress undone by the next fifteen)
//   2. wall slide along the nearest blocker, side chosen by tangent dot
//      desired-direction                                -> 0.000001 tiles, a
//      new fixed point at (11.3102, 2.96839)
//
// That is the signature of an architectural limit rather than a bug in one
// expression: movement here is purely REACTIVE steering, which has local minima
// wherever obstacles form a concave pocket, and this pocket has no interior
// route at all -- the King's and the Cannon's minimum separations sum to 3.8
// while their centres are 3.46 apart. Escaping needs a multi-tile detour, which
// needs global planning (a flow field or A* over the 18x34 grid -- small and
// cheap) rather than a local rule. That is a redesign of the movement core,
// gameplay-affecting for every unit in every match, so it is written up as a
// proposal rather than landed here.
//
// Measured impact: 4 events in 308,464 unit-ticks (60 matches), none of them on
// or near a bridge, all in a player's own back corner beside their own tower.
TEST_CASE("a troop pinched between two buildings still makes progress",
          "[board][collision][regression][!shouldfail]") {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };
    GameManager game(deck, deck);
    auto golem = spawnWedgedGolem(game);
    REQUIRE(golem != nullptr);

    // Let it walk into the pocket and settle.
    for (int i = 0; i < 40; ++i) game.step();
    const Vector2D settled = golem->position;

    // Then give it twelve more seconds. At speed 0.08 that is 9.6 tiles of
    // unobstructed travel; even squeezing out of a concave pocket it must not
    // still be standing on the same spot.
    for (int i = 0; i < 120; ++i) game.step();

    INFO("settled = (" << settled.x << ", " << settled.y << ")");
    INFO("after   = (" << golem->position.x << ", " << golem->position.y << ")");
    REQUIRE(golem->isAlive());
    REQUIRE(settled.distanceTo(golem->position) > 1.0f);
}

TEST_CASE("ordinary unobstructed movement is exactly speed per tick",
          "[board][collision][regression]") {
    // The guard against the fix above becoming a behaviour change for every
    // other unit: with nothing in the way, a troop still walks precisely
    // speed-per-tick in a straight line toward its target.
    //
    // Both entities sit NORTH of the river on purpose. A start and a target on
    // opposite banks is not a straight line at all -- getNextWaypoint correctly
    // routes it via a bridge, which the first version of this test mistook for
    // drift (it ended at x=10.72 walking to a target at x=9.0).
    Board board;
    board.addEntity(std::make_shared<Tower>(board.allocateId(), 9.0f, 27.0f, 2500, 1,
                                            7.0f, 50, 10, 'P'));
    auto troop = std::make_shared<MeleeTroop>(board.allocateId(), 9.0f, 20.0f,
                                              1300, 0, 0.08f, 0.75f, 100, 12, 'G');
    board.addEntity(troop);
    board.commitPendingEntities();

    for (int i = 0; i < 50; ++i) {
        troop->update(board);
        board.resolveCollisions();
    }
    REQUIRE(troop->position.x == Catch::Approx(9.0f));
    REQUIRE(troop->position.y == Catch::Approx(20.0f + 50 * 0.08f).margin(1e-3));
}
