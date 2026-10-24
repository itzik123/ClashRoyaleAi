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
