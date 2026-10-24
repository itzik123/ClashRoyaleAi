#include <catch_amalgamated.hpp>
#include "ClashEnv.h"
#include <algorithm>

// getUnits is a read-only view: it must report what is on the board (towers,
// an injected troop with its deploy time) and change nothing.
TEST_CASE("getUnits lists the towers and an injected troop", "[clash_env][get_units]") {
    std::vector<int> deck = {0, 1, 2, 3, 4, 5, 6, 7};
    ClashEnv env(deck, deck);
    env.reset();
    auto units = env.getUnits();
    REQUIRE(std::count_if(units.begin(), units.end(), [](const UnitView& u) { return u.tower; }) == 6);
    for (const auto& u : units) {
        REQUIRE((u.cardId == GameManager::TOWER_KING_ID || u.cardId == GameManager::TOWER_PRINCESS_ID));
        REQUIRE(u.targetId == -1);   // nothing to lock onto on an empty board
    }

    env.inject(2, 3.0f, 12.0f, 0);  // Giant, team 0: pending until the next tick
    env.stepSelfPlayFast(-1, 0, 0, -1, 0, 0, 1);
    units = env.getUnits();
    auto giant = std::find_if(units.begin(), units.end(), [](const UnitView& u) { return u.cardId == 2; });
    REQUIRE(giant != units.end());
    REQUIRE(giant->team == 0);
    REQUIRE(giant->x == Catch::Approx(3.0f));   // still deploying, so it has not moved
    REQUIRE(giant->y == Catch::Approx(12.0f));
    REQUIRE(giant->deployTicks > 0);
    REQUIRE(giant->deployTicks <= DEPLOY_TIME_TICKS);
    REQUIRE_FALSE(giant->tower);
    REQUIRE(giant->targetable);
}
