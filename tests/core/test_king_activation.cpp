#include <catch_amalgamated.hpp>
#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"
#include "Tower.h"
#include "MeleeTroop.h"
#include <vector>

// ---------------- King Tower activation ----------------
//
// Real Clash Royale's King Tower starts DORMANT. It cannot acquire a target or
// fire until it is activated, permanently, by either taking any damage or by
// losing a Princess Tower on its own team. This engine's King fired from tick 0
// -- a long-standing divergence (perception/UPSTREAM_REQUESTS.md item 3), and
// one the 2026-08-20 sight-range fix made bite harder by widening the King's
// effective reach from 7.0 to 9.4 tiles: its share of the damage dealt to a lone
// Hog Rider went from 5.3% to 36.8%, all of it from a tower that in the real
// game would have been asleep.

namespace {

constexpr int KING_HP = 4008;

// Where an enemy dummy stands relative to the King. Has to satisfy BOTH bounds:
//   inside  the King's effective reach  7.0 + 2.0 (King r) + 0.4 (troop r) = 9.4
//   outside the dummy's own reach       1.0 + 0.4       + 2.0             = 3.4
// The second bound is the one that bites. At 2.5 tiles the dummy hits the King
// back, which wakes it through trigger 1 -- so the "stays dormant" cases would
// pass or fail for a reason that has nothing to do with what they test.
constexpr float DUMMY_STANDOFF = 5.0f;

std::shared_ptr<Tower> makeKing(Board& board, int team) {
    auto king = std::make_shared<Tower>(board.allocateId(), ArenaLayout::CENTER_X,
                                        ArenaLayout::kingY(team), KING_HP, team,
                                        7.0f, 90, 10, 'R');
    king->sightRange = 7.0f;
    king->sleep();   // GameManager::addTower does this for every real King
    board.addEntity(king);
    board.commitPendingEntities();
    return king;
}

std::shared_ptr<MeleeTroop> makeEnemyDummy(Board& board, int team, float x, float y) {
    // Speed 0 so it stands still and the test is about the TOWER, not a chase;
    // huge hp so it survives the window and only the damage RATE differs.
    //
    // Argument order is (id, x, y, hp, team, speed, attackRange, damage,
    // attackCooldown, SYMBOL) -- symbol LAST. Getting it wrong is silent: char,
    // float and int all convert implicitly, so passing 'T' in the speed slot
    // compiles cleanly and gives the dummy speed 84, teleporting it off the
    // board every tick and out of every tower's range.
    auto t = std::make_shared<MeleeTroop>(board.allocateId(), x, y, 100000, team,
                                          0.0f, 1.0f, 1, 10, 'T');
    board.addEntity(t);
    board.commitPendingEntities();
    return t;
}

// Advances the whole board, not just the tower under test.
//
// A Tower attacks by spawning a Projectile (see Tower::performAttack), so the
// damage only lands once THAT entity is updated too. Ticking the tower alone
// looks exactly like a tower that never fired, which is the failure mode this
// helper exists to remove.
void tickAll(Board& board, int ticks) {
    for (int i = 0; i < ticks; ++i) {
        board.currentTick = i;
        std::vector<std::shared_ptr<Entity>> alive = board.getEntities();
        for (const auto& e : alive) if (e->isAlive()) e->update(board);
        board.commitPendingEntities(i);
        board.cleanDeadEntities(i);
    }
}

// Returns the requested tower out of a real GameManager's board.
std::shared_ptr<Entity> findTower(Board& board, int team, char symbol) {
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == team && e->symbol == symbol) return e;
    }
    return nullptr;
}

} // namespace

TEST_CASE("a Princess Tower is never dormant", "[king][activation]") {
    Board board;
    auto princess = std::make_shared<Tower>(board.allocateId(), ArenaLayout::LEFT_LANE_X,
                                            ArenaLayout::princessY(0), 2534, 0,
                                            7.5f, 90, 8, 'P');
    REQUIRE(princess->isAwake());
}

TEST_CASE("every King a real match builds starts dormant, every Princess awake",
          "[king][activation]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    int kings = 0, princesses = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto tower = std::dynamic_pointer_cast<Tower>(e);
        if (!tower) continue;
        if (tower->symbol == 'R') { kings++; REQUIRE_FALSE(tower->isAwake()); }
        else { princesses++; REQUIRE(tower->isAwake()); }
    }
    REQUIRE(kings == 2);
    REQUIRE(princesses == 4);
}

TEST_CASE("a dormant King acquires nothing and deals no damage", "[king][activation]") {
    Board board;
    auto king = makeKing(board, 0);
    auto enemy = makeEnemyDummy(board, 1, ArenaLayout::CENTER_X, ArenaLayout::kingY(0) + DUMMY_STANDOFF);
    REQUIRE_FALSE(king->isAwake());

    const int hpBefore = enemy->hp;
    tickAll(board, 100);

    REQUIRE(enemy->hp == hpBefore);
    REQUIRE_FALSE(king->isAwake());   // an enemy merely STANDING there is not a trigger
}

TEST_CASE("the King wakes on any damage and then fights back", "[king][activation]") {
    Board board;
    auto king = makeKing(board, 0);
    auto enemy = makeEnemyDummy(board, 1, ArenaLayout::CENTER_X, ArenaLayout::kingY(0) + DUMMY_STANDOFF);

    king->update(board);
    REQUIRE_FALSE(king->isAwake());

    // One point of damage is enough -- the trigger is "any", not a threshold.
    king->takeDamage(1);
    king->update(board);
    REQUIRE(king->isAwake());

    const int hpBefore = enemy->hp;
    tickAll(board, 100);
    REQUIRE(enemy->hp < hpBefore);
}

TEST_CASE("the King wakes when a friendly Princess Tower is destroyed, untouched itself",
          "[king][activation]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    auto king = std::dynamic_pointer_cast<Tower>(findTower(board, 0, 'R'));
    auto princess = findTower(board, 0, 'P');
    REQUIRE(king);
    REQUIRE(princess);

    game.step();
    REQUIRE_FALSE(king->isAwake());

    princess->takeDamage(princess->hp);
    game.step();
    game.step();

    REQUIRE(king->isAwake());
    REQUIRE(king->hp == KING_HP);   // it woke without being hit
}

TEST_CASE("an ENEMY Princess dying does not wake our King", "[king][activation]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
    Board& board = game.getBoard();

    auto king = std::dynamic_pointer_cast<Tower>(findTower(board, 0, 'R'));
    auto enemyPrincess = findTower(board, 1, 'P');
    REQUIRE(king);
    REQUIRE(enemyPrincess);

    game.step();
    enemyPrincess->takeDamage(enemyPrincess->hp);
    game.step();
    game.step();

    REQUIRE_FALSE(king->isAwake());
}

TEST_CASE("waking is permanent -- healing back to full does not re-sleep it",
          "[king][activation]") {
    Board board;
    auto king = makeKing(board, 0);
    king->takeDamage(100);
    king->update(board);
    REQUIRE(king->isAwake());

    king->hp = KING_HP;             // fully healed
    king->update(board);
    REQUIRE(king->isAwake());       // latched, not recomputed each tick
}

TEST_CASE("a dormant King is still targetable and damageable", "[king][activation]") {
    Board board;
    auto king = makeKing(board, 0);
    REQUIRE_FALSE(king->isAwake());
    REQUIRE(king->isTargetable());
    REQUIRE(king->isTower());
    REQUIRE(king->isAlive());
    king->takeDamage(500);
    REQUIRE(king->hp == KING_HP - 500);
}

// A King built on a BARE board has zero friendly Princesses. A rule phrased as
// "wake when fewer than 2 are alive" would wake it on its first tick; recording
// the initial count on first update is what makes this hold.
TEST_CASE("a King with no Princesses at all never wakes by itself",
          "[king][activation]") {
    Board board;
    auto king = makeKing(board, 0);
    tickAll(board, 50);
    REQUIRE_FALSE(king->isAwake());
}

TEST_CASE("snapshot carries the King's dormancy in both directions",
          "[king][activation][snapshot]") {
    Board board;
    auto asleep = makeKing(board, 0);

    auto copyAsleep = std::dynamic_pointer_cast<Tower>(asleep->snapshot());
    REQUIRE(copyAsleep);
    REQUIRE_FALSE(copyAsleep->isAwake());

    asleep->wake();
    auto copyAwake = std::dynamic_pointer_cast<Tower>(asleep->snapshot());
    REQUIRE(copyAwake);
    REQUIRE(copyAwake->isAwake());
}

// The behavioural consequence, measured rather than asserted from the mechanism.
//
// ANCHORING. The first version of this put a Hog on the far bridge and compared
// its hp after 60 ticks: both arms read 1247, because the King's effective reach
// is 9.4 tiles and the bridge is ~15 away, so the King could not have fired in
// either arm. That is a test anchored where the effect is zero -- it would have
// passed against a King with no dormancy at all. The enemy has to stand
// somewhere the King can actually shoot it.
//
// The King and a Princess Tower happen to have IDENTICAL effective reach here
// (King 7.0 + 2.0 + 0.4, Princess 7.5 + 1.5 + 0.4, both 9.4), so there is no
// square the King covers alone. The comparison is therefore made with all three
// towers present and only the King's dormancy varying -- any difference in the
// dummy's remaining hp is the King's contribution and nothing else.
TEST_CASE("a dormant King is one fewer tower shooting", "[king][activation][behaviour]") {
    auto dummyHpAfter = [](bool wakeTheKing) {
        Board board;
        auto king = makeKing(board, 0);
        for (float x : { ArenaLayout::LEFT_LANE_X, ArenaLayout::RIGHT_LANE_X }) {
            auto p = std::make_shared<Tower>(board.allocateId(), x, ArenaLayout::princessY(0),
                                             2534, 0, 7.5f, 90, 8, 'P');
            board.addEntity(p);
        }
        board.commitPendingEntities();
        if (wakeTheKing) king->wake();

        // Inside all three towers' reach: DUMMY_STANDOFF from the King, ~5.7
        // from each Princess, and outside its OWN reach of everything so it
        // cannot wake the dormant arm's King by hitting it.
        auto dummy = makeEnemyDummy(board, 1, ArenaLayout::CENTER_X, ArenaLayout::kingY(0) + DUMMY_STANDOFF);
        tickAll(board, 100);
        REQUIRE(dummy->isAlive());
        return dummy->hp;
    };

    const int withKingAsleep = dummyHpAfter(false);
    const int withKingAwake = dummyHpAfter(true);
    INFO("asleep leaves " << withKingAsleep << " hp, awake leaves " << withKingAwake);
    REQUIRE(withKingAsleep > withKingAwake);
}
