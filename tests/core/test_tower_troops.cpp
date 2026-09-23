#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "TowerTroops.h"
#include "GameManager.h"
#include "CombatEntity.h"
#include "MeleeTroop.h"
#include "RoyalChefBuffEffect.h"
#include <memory>

TEST_CASE("towerTroopStats(None) exactly reproduces this engine's original hardcoded Princess Tower",
        "[tower_troops]") {
    CardStats stats = towerTroopStats(TowerTroopType::None);
    REQUIRE(stats.hp == 2534);
    REQUIRE(stats.attackRange == Catch::Approx(7.5f));
    REQUIRE(stats.damage == 90);
    REQUIRE(stats.attackCooldown == 8);
}

TEST_CASE("towerTroopStats(TowerPrincess) gives the sourced, stronger stats", "[tower_troops]") {
    CardStats stats = towerTroopStats(TowerTroopType::TowerPrincess);
    REQUIRE(stats.hp == 3204);
    REQUIRE(stats.damage == 153);
}

TEST_CASE("towerTroopStats(Cannoneer) gives the sourced stats with no splash", "[tower_troops]") {
    CardStats stats = towerTroopStats(TowerTroopType::Cannoneer);
    REQUIRE(stats.hp == 3052);
    REQUIRE(stats.damage == 109);
    REQUIRE(stats.attackCooldown == 8);
    REQUIRE(stats.splashRadius == Catch::Approx(0.0f));
}

TEST_CASE("towerTroopStats(DaggerDuchess) carries the burst-attack fields", "[tower_troops]") {
    CardStats stats = towerTroopStats(TowerTroopType::DaggerDuchess);
    REQUIRE(stats.hp == 2298);
    REQUIRE(stats.burstEveryNAttacks == 8);
    REQUIRE(stats.burstDamageMultiplier == Catch::Approx(8.0f));
}

TEST_CASE("towerTroopStats(RoyalChef) carries a periodic buff effect", "[tower_troops]") {
    CardStats stats = towerTroopStats(TowerTroopType::RoyalChef);
    REQUIRE(stats.periodicEffect != nullptr);
    REQUIRE(stats.periodicIntervalTicks == 280);
}

TEST_CASE("GameManager defaults to TowerTroopType::None -- existing behavior byte-for-byte unaffected",
        "[game_manager][tower_troops]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });

    int princessCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Princess Tower") {
            princessCount++;
            REQUIRE(e->hp == 2534);
        }
    }
    REQUIRE(princessCount == 4);
}

TEST_CASE("GameManager wires the selected Tower Troop onto both of that team's Princess Towers",
        "[game_manager][tower_troops]") {
    GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 },
        TowerTroopType::Cannoneer, TowerTroopType::TowerPrincess);

    int aiPrincessCount = 0, oppPrincessCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name != "Princess Tower") continue;
        if (e->team == 0) { aiPrincessCount++; REQUIRE(e->hp == 3052); }       // Cannoneer
        if (e->team == 1) { oppPrincessCount++; REQUIRE(e->hp == 3204); }     // Tower Princess
    }
    REQUIRE(aiPrincessCount == 2);
    REQUIRE(oppPrincessCount == 2);
}

TEST_CASE("RoyalChefBuffEffect buffs the nearest ally, not itself", "[tower_troops][royal_chef]") {
    Board board;
    auto combatAlly = std::make_shared<StationaryCombatant>(2, 3.0f, 0.0f, 1000, 0, 1.0f, 0, 100);
    spawn(board, combatAlly);

    RoyalChefBuffEffect effect(6.0f, 1.10f);
    effect.apply(board, Vector2D{ 0.0f, 0.0f }, 0); // fires from the tower's own position

    REQUIRE(combatAlly->buffTicksRemaining > 0);
    REQUIRE(combatAlly->hp == 1100); // +10% hp top-up
}


TEST_CASE("the Royal Chef feeds an ally once, not compounding forever",
          "[tower_troops][royal_chef][regression]") {
    // RoyalChefBuffEffect feeds the nearest ally `hp += hp / 10` with a
    // permanent damage buff. The real card grants "+1 Level" once per troop, so
    // a tank parked beside the tower must not be fed again and again.
    Board board;
    auto ally = std::make_shared<MeleeTroop>(1, 5.0f, 6.0f, 1000, 0, 0.1f, 1.0f, 100, 10, 'a');
    spawn(board, ally);

    RoyalChefBuffEffect chef(6.0f, 1.10f);
    chef.apply(board, Vector2D{ 5.0f, 5.0f }, 0);
    const int afterFirst = ally->hp;

    // Control: the first serving must have done something, or "did not
    // compound" is vacuous.
    REQUIRE(afterFirst > 1000);
    REQUIRE(ally->buffTicksRemaining > 0);

    for (int i = 0; i < 5; ++i) chef.apply(board, Vector2D{ 5.0f, 5.0f }, 0);

    INFO("hp after one serving = " << afterFirst << ", after six = " << ally->hp);
    REQUIRE(ally->hp == afterFirst);
}
