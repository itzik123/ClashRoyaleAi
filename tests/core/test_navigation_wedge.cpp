#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "ArenaLayout.h"
#include "Board.h"
#include "Building.h"
#include "Tower.h"
#include "MeleeTroop.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include <vector>

// --- the two-obstacle collision wedge ---
// A unit pinched in the concave pocket between two adjacent buildings (here the
// King Tower and a Cannon) stops dead: both constraints sit exactly on their
// boundary, and the two perpendicular slides in Board::pushAwayFrom, which
// exist to stop a unit sticking on one obstacle, point in opposite tangential
// directions and cancel. Net displacement is zero, forever.
//
// Not the bridge trap (test_board.cpp): no waypoint is involved, and it can
// happen anywhere. A local minimum in reactive steering.

namespace {

constexpr int ICE_GOLEM = 40;
constexpr int CANNON = 25;

// The measured configuration, through the real GameManager tick loop and real
// cards. The trap is an attracting fixed point entered by walking into its
// basin, so the unit starts slightly outside and falls in; a bare-Board
// reconstruction does not reproduce.
std::shared_ptr<Entity> spawnWedgedGolem(GameManager& game) {
    // GameManager::reset() builds the King; the Cannon is translated by the
    // same CENTER_X - 9.0 offset, so the pocket keeps its measured shape.
    constexpr float dx = ArenaLayout::CENTER_X - 9.0f;
    CardRegistry::getInstance().getCard(CANNON)->spawnEntity(12.032f + dx, 4.169f, 0, game.getBoard());
    CardRegistry::getInstance().getCard(ICE_GOLEM)->spawnEntity(11.58f + dx, 2.84f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->cardId == ICE_GOLEM) return e;
    }
    return nullptr;
}

} // namespace

// Known defect, pinned with `[!shouldfail]`: the suite stays green while the
// bug is on record, and goes red the moment it is fixed. Written up in
// perception/UPSTREAM_REQUESTS.md.
//
// Local repairs only move the equilibrium; this pocket has no interior route
// (the minimum separations sum to 3.8 while the centres are 3.46 apart).
// Escaping needs global planning (a flow field or A* over the grid), a
// gameplay-affecting redesign of the movement core. Rare: all measured events
// were in a player's own back corner, none near a bridge.
TEST_CASE("a troop pinched between two buildings still makes progress",
          "[board][collision][regression][!shouldfail]") {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };
    GameManager game(deck, deck);
    auto golem = spawnWedgedGolem(game);
    REQUIRE(golem != nullptr);

    // Let it walk in and settle.
    for (int i = 0; i < 40; ++i) game.step();
    const Vector2D settled = golem->position;

    // Twelve more seconds: 9.6 tiles of travel at speed 0.08. It must not still
    // be on the same spot.
    for (int i = 0; i < 120; ++i) game.step();

    INFO("settled = (" << settled.x << ", " << settled.y << ")");
    INFO("after   = (" << golem->position.x << ", " << golem->position.y << ")");
    REQUIRE(golem->isAlive());
    REQUIRE(settled.distanceTo(golem->position) > 1.0f);
}

TEST_CASE("ordinary unobstructed movement is exactly speed per tick",
          "[board][collision][regression]") {
    // With nothing in the way, a troop still walks exactly speed-per-tick in a
    // straight line. Both entities sit north of the river: across it,
    // getNextWaypoint routes via a bridge.
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
