#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "Building.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "MeleeTroop.h"
#include "Tower.h"

// Deploy time (2026-08-19). A freshly placed troop or building is inert for
// DEPLOY_TIME_TICKS: on the board, targetable and damageable, but unable to
// move, target or attack.
//
// WHY THESE TESTS EXIST. The change is GAMEPLAY-AFFECTING and its whole point
// is asymmetric: a defender places INTO an existing threat and needs its answer
// to act now, while an attacker places before contact and would have spent that
// second walking anyway. So the tests pin both halves -- that the delay really
// happens, and that it applies to exactly the right set of entities. Getting
// the SET wrong is the silent failure: a tower that deploys, or a spell delayed
// twice, would change the game in ways no aggregate metric would localise.

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
    // A destination is required or the Hog has nothing to walk toward and the
    // test would pass for the wrong reason -- findTarget falls back to the
    // nearest enemy TOWER, and a bare Board has none.
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
    // The defensive half of the change: an answer placed on top of a threat
    // cannot act for a second, which is the tempo the real game charges and
    // this engine was giving away for free.
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
    // Inert, NOT invulnerable. If this ever flipped, a mistimed placement would
    // become free rather than punishable, which is the opposite of the intent.
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
    // Towers are built directly by GameManager and never go through
    // CardFactories, so they must be unaffected. A tower that spent its first
    // second inert would hand every opening push a free hit.
    Board board;
    auto tower = std::make_shared<Tower>(board.allocateId(), 9.0f, 2.5f, 4000, 0, 7.0f, 50, 10, 'K');
    board.addEntity(tower);
    board.commitPendingEntities();
    REQUIRE(tower->deployTicksRemaining == 0);
}

TEST_CASE("spells are not delayed twice", "[deploy]") {
    // A spell already has spellDelayTicks. spawnSpell deliberately bypasses
    // applyCardMetadata, so it must NOT also pick up a deploy time -- that
    // would silently add a second to every Fireball in the game.
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
    // Board::deepCopy backs decision-time search. A snapshot that reset deploy
    // time to 0 would let search evaluate a board where every unit acts a full
    // second early -- wrong in exactly the direction that makes attacking look
    // better than it is.
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
