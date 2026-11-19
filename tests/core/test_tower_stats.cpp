#include <catch_amalgamated.hpp>
#include "ClashEnv.h"

// setTowerStats re-levels a tower without changing its state: a full tower
// stays full (so a sleeping King stays asleep), a damaged one keeps its
// fraction, and a destroyed one is refused.
TEST_CASE("setTowerStats keeps the hp fraction and the King asleep", "[tower][tower_stats]") {
    std::vector<int> deck = {0, 1, 2, 3, 4, 5, 6, 7};
    ClashEnv env(deck, deck);
    env.reset();
    REQUIRE(env.setTowerStats(0, 0, 6144, 128, 10));
    REQUIRE(env.getTowerMaxHp(0, 0) == 6144);
    REQUIRE(env.getTowerHp(0, 0) == 6144);

    REQUIRE(env.setTowerHp(1, 1, env.getTowerMaxHp(1, 1) / 2.0f));
    REQUIRE(env.setTowerStats(1, 1, 3584, 128, 8));
    REQUIRE(env.getTowerHp(1, 1) == Catch::Approx(1792).margin(1));

    REQUIRE(env.destroyTower(1, 2));
    REQUIRE_FALSE(env.setTowerStats(1, 2, 3584, 128, 8));
    REQUIRE_FALSE(env.setTowerStats(0, 1, 0, 128, 8));
}

// The hp checks above pass even if damage and cooldown were never written, and
// "full" is only the King's wake condition, not the King being asleep. Read
// the tower itself.
TEST_CASE("setTowerStats writes damage and cooldown, and a re-levelled King sleeps on",
          "[tower][tower_stats]") {
    std::vector<int> deck = {0, 1, 2, 3, 4, 5, 6, 7};
    GameManager game(deck, deck);
    Board& board = game.getBoard();
    Tower* king = game.findTower(0, 0);
    REQUIRE(king != nullptr);
    REQUIRE_FALSE(king->isAwake());

    REQUIRE(game.setTowerStats(0, 0, 6144, 128, 10));
    REQUIRE(king->getBaseDamage() == 128);
    REQUIRE(king->getAttackCooldown() == 10);
    king->update(board);
    REQUIRE_FALSE(king->isAwake());

    // A refused call changes nothing.
    Tower* princess = game.findTower(0, 1);
    REQUIRE(princess != nullptr);
    const int hp = princess->hp, maxHp = princess->getMaxHp();
    const int damage = princess->getBaseDamage(), cooldown = princess->getAttackCooldown();
    REQUIRE_FALSE(game.setTowerStats(0, 1, 3584, -1, 8));
    REQUIRE_FALSE(game.setTowerStats(0, 1, 3584, 128, 0));
    REQUIRE(princess->hp == hp);
    REQUIRE(princess->getMaxHp() == maxHp);
    REQUIRE(princess->getBaseDamage() == damage);
    REQUIRE(princess->getAttackCooldown() == cooldown);
}
