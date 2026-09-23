#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "MightyMinerEscapeEffect.h"
#include "AreaSpell.h"

TEST_CASE("MightyMinerEscapeEffect teleports the caster to the horizontally-mirrored X, Y unchanged", "[mighty_miner]") {
    Board board; // width 18 -> mirror axis at (18-1)/2 = 8.5
    auto champion = std::make_shared<StationaryCombatant>(1, 3.0f, 20.0f, 1000, 0, 1.0f, 10, 10);
    spawn(board, champion);

    MightyMinerEscapeEffect effect(2.5f, 332, 10, 1.8f);
    effect.apply(board, *champion);

    REQUIRE(champion->position.x == Catch::Approx(14.0f)); // (18-1) - 3 = 14
    REQUIRE(champion->position.y == Catch::Approx(20.0f)); // unchanged: a lane swap, not a flip
}

TEST_CASE("The mirror math is symmetric regardless of which side the caster starts on", "[mighty_miner]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 14.0f, 20.0f, 1000, 0, 1.0f, 10, 10);
    spawn(board, champion);

    MightyMinerEscapeEffect effect(2.5f, 332, 10, 1.8f);
    effect.apply(board, *champion);

    REQUIRE(champion->position.x == Catch::Approx(3.0f)); // (18-1) - 14 = 3
}

TEST_CASE("MightyMinerEscapeEffect leaves a bomb at the caster's ORIGINAL position, on the caster's own team", "[mighty_miner]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 3.0f, 20.0f, 1000, 1, 1.0f, 10, 10);
    spawn(board, champion);

    MightyMinerEscapeEffect effect(2.5f, 332, 10, 1.8f);
    effect.apply(board, *champion);
    board.commitPendingEntities();

    std::shared_ptr<AreaSpell> bomb;
    for (const auto& e : board.getEntities()) {
        bomb = std::dynamic_pointer_cast<AreaSpell>(e);
        if (bomb) break;
    }
    REQUIRE(bomb != nullptr);
    REQUIRE(bomb->team == 1);
    REQUIRE(bomb->position.x == Catch::Approx(3.0f));  // original, pre-teleport position
    REQUIRE(bomb->position.y == Catch::Approx(20.0f));
}

TEST_CASE("The bomb only detonates after its configured delay, not immediately", "[mighty_miner]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 3.0f, 20.0f, 1000, 0, 1.0f, 10, 10);
    auto enemy = std::make_shared<DummyEntity>(2, 3.0f, 20.5f, 1000, 1); // right at the bomb's original spot
    spawn(board, champion);
    spawn(board, enemy);

    MightyMinerEscapeEffect effect(2.5f, 332, 10, 1.8f); // 10-tick delay
    effect.apply(board, *champion);
    board.commitPendingEntities();

    std::shared_ptr<AreaSpell> bomb;
    for (const auto& e : board.getEntities()) {
        bomb = std::dynamic_pointer_cast<AreaSpell>(e);
        if (bomb) break;
    }
    REQUIRE(bomb != nullptr);

    // The delay branch decrements then returns, so a 10-tick delay takes 10
    // calls to drain and an 11th to detonate.
    for (int i = 0; i < 10; ++i) bomb->update(board);
    REQUIRE(enemy->hp == 1000); // still hasn't gone off

    bomb->update(board); // 11th call: detonates
    REQUIRE(enemy->hp == 1000 - 332);
}

TEST_CASE("The bomb hits both a ground and a flying enemy, and knocks them back", "[mighty_miner]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1000, 0, 1.0f, 10, 10);
    auto groundEnemy = std::make_shared<DummyEntity>(2, 5.0f, 6.0f, 1000, 1); // dist 1.0
    auto flyingEnemy = std::make_shared<DummyEntity>(3, 5.0f, 4.0f, 1000, 1); // dist 1.0
    flyingEnemy->isFlying = true;
    spawn(board, champion);
    spawn(board, groundEnemy);
    spawn(board, flyingEnemy);

    MightyMinerEscapeEffect effect(2.5f, 332, 0, 1.8f); // no delay, for a same-tick detonation
    effect.apply(board, *champion);
    board.commitPendingEntities();

    std::shared_ptr<AreaSpell> bomb;
    for (const auto& e : board.getEntities()) {
        bomb = std::dynamic_pointer_cast<AreaSpell>(e);
        if (bomb) break;
    }
    REQUIRE(bomb != nullptr);
    bomb->update(board);

    REQUIRE(groundEnemy->hp == 1000 - 332);
    REQUIRE(flyingEnemy->hp == 1000 - 332); // groundOnly stays false: hits air too

    // Knocked back 1.8 away from the bomb's origin (5,5).
    REQUIRE(groundEnemy->position.y == Catch::Approx(7.8f));
    REQUIRE(flyingEnemy->position.y == Catch::Approx(2.2f));
}
