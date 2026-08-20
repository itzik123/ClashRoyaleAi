#include <catch_amalgamated.hpp>
#include "ArenaLayout.h"
#include "LanePath.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "Tower.h"
#include <vector>
#include <algorithm>

// ---------------- blind lane pathing ----------------
//
// A unit with nothing inside its own sight range is BLIND. Rule B of the player
// audit: a blind unit does not beeline for whatever tower happens to be nearest,
// it acts "like a magnet to its predefined lane path" and walks its own lane
// toward the enemy base.
//
// Before this, CombatEntity::findTarget's blind fallback was "closest enemy
// Tower by raw distance". With one enemy Princess destroyed that sends a unit
// diagonally across the arena to the OTHER lane's Princess.

namespace {

constexpr int ICE_GOLEM = 40;

// Destroys team `team`'s Princess Tower on the given side; returns their King.
std::shared_ptr<Entity> killPrincess(Board& board, int team, bool left) {
    std::shared_ptr<Entity> king;
    for (const auto& e : board.getEntities()) {
        if (!e->isTower() || e->team != team) continue;
        if (e->symbol == 'R') { king = e; continue; }
        if (ArenaLayout::isLeftLane(e->position.x) == left) e->takeDamage(e->hp);
    }
    board.cleanDeadEntities(0);
    return king;
}

// What the OLD rule would have answered: the closest living enemy tower.
std::shared_ptr<Entity> closestEnemyTower(Board& board, int myTeam, const Vector2D& from) {
    std::shared_ptr<Entity> best;
    float bestDist = 1e9f;
    for (const auto& e : board.getEntities()) {
        if (!e->isAlive() || !e->isTower() || e->team == myTeam) continue;
        float d = from.distanceTo(e->position);
        if (d < bestDist) { bestDist = d; best = e; }
    }
    return best;
}

std::shared_ptr<Entity> spawnFor(Board& board, int cardId, int team, float x, float y) {
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(x, y, team, board);
    board.commitPendingEntities();
    for (const auto& e : board.getEntities())
        if (e->cardId == cardId && e->team == team) return e;
    return nullptr;
}

} // namespace

// ---- the premise, measured rather than asserted ----
//
// The reproduction is POSITION-DEPENDENT, and getting this wrong would produce
// a test that passes against unfixed code. With team 1's left Princess dead, a
// unit in the left lane measures:
//
//     at (2.5, 12.0)   right Princess 18.90   King 19.45   <- wrong tower
//     at (2.5, 14.0)   right Princess 17.36   King 17.56   <- wrong tower
//     at (2.5, 15.0)   right Princess 16.62   King 16.62   <- tie
//     at (2.5, 16.5)   right Princess 15.57   King 15.23   <- RIGHT, by accident
//
// The crossover is y ~ 15.0. AT THE BRIDGE MOUTH the broken rule already gives
// the correct answer, so a test written there proves nothing -- the "cross-check
// anchored where the error is zero" trap this codebase already paid for once
// with the 2026-08-05 tile-grid refit. Everything below anchors at y <= 14.
TEST_CASE("the old closest-tower rule really does pick the wrong lane at y<=14",
          "[lane][aggro][premise]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);
    REQUIRE(king);

    Vector2D from{ ArenaLayout::LEFT_BRIDGE_X, 13.0f };
    auto naive = closestEnemyTower(board, 0, from);
    REQUIRE(naive);
    REQUIRE(naive->symbol == 'P');                          // a Princess...
    REQUIRE(naive->position.x > ArenaLayout::CENTER_X);      // ...in the RIGHT lane

    // And at the bridge it would have been right anyway -- which is exactly why
    // the cases below are not anchored there.
    auto atBridge = closestEnemyTower(board, 0, Vector2D{ ArenaLayout::LEFT_BRIDGE_X,
                                                          ArenaLayout::BRIDGE_Y });
    REQUIRE(atBridge->symbol == 'R');
}

// ---- Example 2 from the brief ----
TEST_CASE("with its lane's Princess dead a unit targets the King, not the other lane",
          "[lane][aggro][brief]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);
    REQUIRE(king);

    auto objective = LanePath::laneObjective(board, /*myTeam=*/0,
                                             Vector2D{ ArenaLayout::LEFT_BRIDGE_X, 13.0f });
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

TEST_CASE("with its lane's Princess alive that Princess is the objective",
          "[lane][aggro]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    auto objective = LanePath::laneObjective(board, 0, Vector2D{ ArenaLayout::LEFT_BRIDGE_X, 13.0f });
    REQUIRE(objective);
    REQUIRE(objective->symbol == 'P');
    REQUIRE(objective->team == 1);
    REQUIRE(objective->position.x == Catch::Approx(ArenaLayout::LEFT_LANE_X));
}

TEST_CASE("the right lane mirrors the left exactly", "[lane][aggro]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/false);

    auto objective = LanePath::laneObjective(board, 0,
                                             Vector2D{ ArenaLayout::RIGHT_BRIDGE_X, 13.0f });
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

TEST_CASE("both enemy Princesses dead leaves the King", "[lane][aggro]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    killPrincess(board, 1, true);
    auto king = killPrincess(board, 1, false);

    auto objective = LanePath::laneObjective(board, 0,
                                             Vector2D{ ArenaLayout::LEFT_BRIDGE_X, 13.0f });
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

// Team 1 attacks downfield; the whole rule has to mirror or one side plays a
// different game -- the failure mode the team-1 observation bug already cost
// this project once.
TEST_CASE("the rule is side-agnostic", "[lane][aggro]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 0, /*left=*/true);

    auto objective = LanePath::laneObjective(board, /*myTeam=*/1,
                                             Vector2D{ ArenaLayout::LEFT_BRIDGE_X, 20.0f });
    REQUIRE(objective);
    REQUIRE(objective->id == king->id);
}

// ---- the end-to-end behaviour: Example 2 walked, not just resolved ----
TEST_CASE("an Ice Golem walking the left lane never crosses to the right",
          "[lane][pathing][brief]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    killPrincess(board, 1, /*left=*/true);

    auto golem = spawnFor(board, ICE_GOLEM, 0, ArenaLayout::LEFT_BRIDGE_X, 13.0f);
    REQUIRE(golem);

    float maxX = golem->position.x;
    float maxY = golem->position.y;
    for (int t = 0; t < 900 && golem->isAlive(); ++t) {
        game.step();
        maxX = std::max(maxX, golem->position.x);
        maxY = std::max(maxY, golem->position.y);
    }
    INFO("furthest right it ever got: x=" << maxX << ", furthest upfield: y=" << maxY);
    // It must have crossed the river...
    REQUIRE(maxY > ArenaLayout::BRIDGE_Y);
    // ...without ever wandering into the right lane on the way to the King.
    // The King sits at CENTER_X, so reaching it legitimately brings the unit to
    // 8.5; what it must never do is head out past that toward x=14.
    REQUIRE(maxX <= ArenaLayout::CENTER_X + 1.0f);
}

// ---------------- the approach curve ----------------
//
// Heading for the King with its own lane's Princess dead, a unit must travel UP
// ITS LANE to the empty Princess slot before angling inward -- not cut the
// corner from the bridge. That is the "curves rightward toward the King Tower"
// the brief describes.

TEST_CASE("approachPoint is the identity for everything except an enemy King",
          "[lane][pathing]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    Vector2D from{ ArenaLayout::LEFT_BRIDGE_X, 13.0f };

    // A live lane Princess: walked to directly, so the common case is untouched.
    auto princess = LanePath::laneObjective(board, 0, from);
    REQUIRE(princess->symbol == 'P');
    Vector2D wp = LanePath::approachPoint(board, 0, from, princess);
    REQUIRE(wp.x == Catch::Approx(princess->position.x));
    REQUIRE(wp.y == Catch::Approx(princess->position.y));

    // An ordinary troop in sight: likewise untouched.
    auto troop = spawnFor(board, ICE_GOLEM, 1, ArenaLayout::LEFT_BRIDGE_X, 14.0f);
    REQUIRE(troop);
    Vector2D wt = LanePath::approachPoint(board, 0, from, troop);
    REQUIRE(wt.x == Catch::Approx(troop->position.x));
    REQUIRE(wt.y == Catch::Approx(troop->position.y));
}

TEST_CASE("approaching the King routes via the lane's own Princess slot first",
          "[lane][pathing]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);

    // Short of the Princess row: aim at the lane slot, not the King.
    Vector2D early{ ArenaLayout::LEFT_BRIDGE_X, 20.0f };
    Vector2D wpEarly = LanePath::approachPoint(board, 0, early, king);
    REQUIRE(wpEarly.x == Catch::Approx(ArenaLayout::LEFT_LANE_X));
    REQUIRE(wpEarly.y == Catch::Approx(ArenaLayout::princessY(1)));

    // Level with it or beyond: the remaining leg is the angle in to the King.
    Vector2D late{ ArenaLayout::LEFT_LANE_X, ArenaLayout::princessY(1) + 0.5f };
    Vector2D wpLate = LanePath::approachPoint(board, 0, late, king);
    REQUIRE(wpLate.x == Catch::Approx(king->position.x));
    REQUIRE(wpLate.y == Catch::Approx(king->position.y));
}

// THE ABSORBING-STATE GUARD.
//
// Two absorbing states have already shipped in this engine, both from a planner
// handing a mover the point it already stands on: Troop::moveTowards refuses to
// move inside Board::WAYPOINT_ARRIVAL_EPS, so position never changes, so the
// waypoint never changes, forever. approachPoint adds a NEW intermediate
// waypoint, so it gets the same treatment -- swept at finer-than-epsilon
// resolution along the whole lane, in both lanes, for both teams.
TEST_CASE("approachPoint never returns a point the mover already occupies",
          "[lane][pathing][regression]") {
    for (int myTeam = 0; myTeam < 2; ++myTeam) {
        for (bool left : { true, false }) {
            GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
            Board& board = game.getBoard();
            auto king = killPrincess(board, 1 - myTeam, left);
            REQUIRE(king);

            const float laneX = left ? ArenaLayout::LEFT_LANE_X : ArenaLayout::RIGHT_LANE_X;
            int trapped = 0;
            Vector2D firstTrap{ -1.0f, -1.0f };

            // 0.005 is half the 0.01 epsilon, so every trap disc gets several
            // samples inside it. Sweeping the lane column is the interesting
            // line: it passes exactly through W1.
            for (float y = 0.0f; y <= 33.0f; y += 0.005f) {
                Vector2D from{ laneX, y };
                if (from.distanceTo(king->position) <= 1.0f) continue;  // legitimately arrived
                Vector2D wp = LanePath::approachPoint(board, myTeam, from, king);
                if (from.distanceTo(wp) <= Board::WAYPOINT_ARRIVAL_EPS) {
                    if (trapped == 0) firstTrap = from;
                    trapped++;
                }
            }
            INFO("team " << myTeam << (left ? " left" : " right") << " lane: "
                 << trapped << " absorbing positions, first at ("
                 << firstTrap.x << ", " << firstTrap.y << ")");
            REQUIRE(trapped == 0);

            // And exactly ON the intermediate waypoint, which is the one
            // position the sweep's step size could in principle skip.
            Vector2D onW1{ laneX, ArenaLayout::princessY(1 - myTeam) };
            REQUIRE(onW1.distanceTo(LanePath::approachPoint(board, myTeam, onW1, king))
                    > Board::WAYPOINT_ARRIVAL_EPS);
        }
    }
}

// The composition is what a unit actually follows: approachPoint feeds
// getNextWaypoint, and either can be absorbing-free while the PAIR is not.
TEST_CASE("the approachPoint + getNextWaypoint composition has no absorbing state",
          "[lane][pathing][regression]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    auto king = killPrincess(board, 1, /*left=*/true);

    int trapped = 0;
    Vector2D firstTrap{ -1.0f, -1.0f };
    for (float x = 0.0f; x <= 17.0f; x += 0.5f) {
        for (float y = 0.0f; y <= 33.0f; y += 0.005f) {
            Vector2D from{ x, y };
            if (std::fabs(board.clampToBoard(from, false).y - y) > 1e-6f) continue;
            if (from.distanceTo(king->position) <= 1.0f) continue;
            Vector2D approach = LanePath::approachPoint(board, 0, from, king);
            Vector2D wp = board.getNextWaypoint(from, approach);
            if (from.distanceTo(wp) <= Board::WAYPOINT_ARRIVAL_EPS) {
                if (trapped == 0) firstTrap = from;
                trapped++;
            }
        }
    }
    INFO(trapped << " absorbing positions, first at (" << firstTrap.x << ", " << firstTrap.y << ")");
    REQUIRE(trapped == 0);
}

// End to end, with the positional trace the two shipped deadlocks were only ever
// found by: a unit must make real progress every second, not merely finish.
TEST_CASE("an Ice Golem walking to the King never stalls en route",
          "[lane][pathing][regression]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();
    killPrincess(board, 1, /*left=*/true);

    auto golem = spawnFor(board, ICE_GOLEM, 0, ArenaLayout::LEFT_BRIDGE_X, 13.0f);
    REQUIRE(golem);

    std::shared_ptr<Entity> king;
    for (const auto& e : board.getEntities())
        if (e->isTower() && e->team == 1 && e->symbol == 'R') king = e;
    REQUIRE(king);

    // Measure stalls ONLY during the approach.
    //
    // The first version of this ran the full 900 ticks and failed at 68 ticks
    // stationary -- which was not a deadlock. The golem was at (5.85, 28.85),
    // 3.12 tiles from the King, against its own effective reach of
    // 0.75 + 0.4 (its radius) + 2.0 (King radius) = 3.15. It had ARRIVED and was
    // attacking. A unit standing still to fight is the goal, not a stall, so the
    // trace has to end where the journey does.
    const float arrived = 3.2f;
    Vector2D last = golem->position;
    int stillFor = 0, longestStall = 0, ticksToArrive = -1;

    for (int t = 0; t < 900 && golem->isAlive(); ++t) {
        game.step();
        if (golem->position.distanceTo(king->position) <= arrived) { ticksToArrive = t; break; }
        if (golem->position.distanceTo(last) < 1e-4f) stillFor++;
        else { longestStall = std::max(longestStall, stillFor); stillFor = 0; }
        last = golem->position;
    }
    longestStall = std::max(longestStall, stillFor);

    INFO("arrived at tick " << ticksToArrive << ", longest stall en route "
         << longestStall << " ticks, ended at (" << golem->position.x << ", "
         << golem->position.y << ")");
    // It has to actually get there -- a unit that never arrives would otherwise
    // pass a stall check by dying or wandering.
    REQUIRE(ticksToArrive > 0);
    // DEPLOY_TIME_TICKS of stillness at spawn is correct and expected. Anything
    // approaching a further second of it, mid-walk, is a deadlock.
    REQUIRE(longestStall < 25);
}
