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
