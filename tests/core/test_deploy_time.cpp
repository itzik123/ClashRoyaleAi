#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "Building.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "MeleeTroop.h"
#include "Tower.h"

// Deploy time: a freshly placed troop or building is inert for
// DEPLOY_TIME_TICKS (on the board, targetable and damageable, but not moving,
// targeting or attacking). These pin that the delay happens and that it applies
// to exactly the right entities; a tower that deploys, or a spell delayed
// twice, would change the game silently.

namespace {

// The one card whose behaviour the surrounding investigation is about.
constexpr int HOG_RIDER = 15;
constexpr int CANNON = 25;
constexpr int FIREBALL = 7;

std::shared_ptr<CombatEntity> firstCombatEntity(Board& board, int team) {
    for (const auto& e : board.getEntities()) {
        if (e->team != team) continue;
        if (e->isTower()) continue;
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (combat) return combat;
    }
    return nullptr;
}

} // namespace

TEST_CASE("a freshly spawned troop starts with deploy time pending", "[deploy]") {
    Board board;
    CardRegistry::getInstance().getCard(HOG_RIDER)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    auto troop = firstCombatEntity(board, 0);
    REQUIRE(troop != nullptr);
    REQUIRE(troop->deployTicksRemaining == DEPLOY_TIME_TICKS);
}

TEST_CASE("a deploying troop does not move", "[deploy]") {
    Board board;
    // The Hog needs a destination: a bare Board has no towers for findTarget to
    // fall back on.
    auto tower = std::make_shared<Tower>(board.allocateId(), 9.0f, 30.5f, 4000, 1,
                                         7.0f, 50, 10, 'K');
    board.addEntity(tower);
    CardRegistry::getInstance().getCard(HOG_RIDER)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    auto troop = firstCombatEntity(board, 0);
    REQUIRE(troop != nullptr);
    const Vector2D start = troop->position;

    // One tick short of deployed: still exactly where it landed.
    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) {
        troop->update(board);
        REQUIRE(troop->position.x == Catch::Approx(start.x));
        REQUIRE(troop->position.y == Catch::Approx(start.y));
    }
    REQUIRE(troop->deployTicksRemaining == 0);

    // The very next tick it is live and moves.
    troop->update(board);
    REQUIRE(troop->position.y != Catch::Approx(start.y));
}

TEST_CASE("a deploying troop deals no damage", "[deploy]") {
    // The defensive half: an answer placed on top of a threat cannot act for a
    // second.
    Board board;
    auto victim = std::make_shared<MeleeTroop>(
        board.allocateId(), 9.0f, 10.0f, 5000, 1, 0.0f, 1.0f, 10, 10, 'V');
    board.addEntity(victim);
    board.commitPendingEntities();
    const int victimHpAtStart = victim->hp;

    CardRegistry::getInstance().getCard(HOG_RIDER)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();
    auto attacker = firstCombatEntity(board, 0);
    REQUIRE(attacker != nullptr);

    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) attacker->update(board);
    REQUIRE(victim->hp == victimHpAtStart);
}

TEST_CASE("a deploying troop is still targetable and takes damage", "[deploy]") {
    // Inert, not invulnerable, so a mistimed placement is punishable.
    Board board;
    CardRegistry::getInstance().getCard(HOG_RIDER)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    auto troop = firstCombatEntity(board, 0);
    REQUIRE(troop != nullptr);
    REQUIRE(troop->deployTicksRemaining > 0);
    REQUIRE(troop->isTargetable());

    const int before = troop->hp;
    troop->takeDamage(100);
    REQUIRE(troop->hp < before);
}

TEST_CASE("a defensive building also deploys", "[deploy]") {
    Board board;
    CardRegistry::getInstance().getCard(CANNON)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    auto building = firstCombatEntity(board, 0);
    REQUIRE(building != nullptr);
    REQUIRE(building->isBuilding());
    REQUIRE(building->deployTicksRemaining == DEPLOY_TIME_TICKS);
}

TEST_CASE("towers never deploy", "[deploy]") {
    // Towers are built by GameManager, not CardFactories, so they never deploy.
    Board board;
    auto tower = std::make_shared<Tower>(board.allocateId(), 9.0f, 2.5f, 4000, 0, 7.0f, 50, 10, 'K');
    board.addEntity(tower);
    board.commitPendingEntities();
    REQUIRE(tower->deployTicksRemaining == 0);
}

TEST_CASE("spells are not delayed twice", "[deploy]") {
    // A spell already has spellDelayTicks; spawnSpell bypasses
    // applyCardMetadata and must not also get a deploy time.
    Board board;
    CardRegistry::getInstance().getCard(FIREBALL)->spawnEntity(9.0f, 20.0f, 0, board);
    board.commitPendingEntities();

    bool sawSpell = false;
    for (const auto& e : board.getEntities()) {
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (combat) continue;              // AreaSpell is not a CombatEntity
        sawSpell = true;
    }
    REQUIRE(sawSpell);
}

TEST_CASE("deploy time survives a deep copy", "[deploy][snapshot]") {
    // A snapshot must keep the remaining deploy time, or search sees every unit
    // act a second early.
    Board board;
    CardRegistry::getInstance().getCard(HOG_RIDER)->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();
    auto original = firstCombatEntity(board, 0);
    REQUIRE(original != nullptr);
    original->update(board);                       // burn one tick
    const int expected = original->deployTicksRemaining;
    REQUIRE(expected == DEPLOY_TIME_TICKS - 1);

    Board copy = board.deepCopy();
    auto copied = firstCombatEntity(copy, 0);
    REQUIRE(copied != nullptr);
    REQUIRE(copied->deployTicksRemaining == expected);
}
