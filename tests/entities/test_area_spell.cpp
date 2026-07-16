#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "AreaSpell.h"

TEST_CASE("AreaSpell is never itself a valid combat target", "[area_spell]") {
    AreaSpell spell(1, 5.0f, 5.0f, 0, 3.0f, 100, 5);
    REQUIRE_FALSE(spell.isTargetable());
}

TEST_CASE("AreaSpell waits out its delay before detonating", "[area_spell][delay]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 500, 2);

    spell.update(board); // delayTicks 2 -> 1, no detonation
    REQUIRE(spell.isAlive());
    REQUIRE(enemy->hp == 1000);

    spell.update(board); // delayTicks 1 -> 0, no detonation
    REQUIRE(spell.isAlive());
    REQUIRE(enemy->hp == 1000);

    spell.update(board); // delayTicks == 0: detonates now
    REQUIRE_FALSE(spell.isAlive());
    REQUIRE(enemy->hp == 500);
}

TEST_CASE("AreaSpell with zero delay detonates on the very first update", "[area_spell][delay]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 500, 0);
    spell.update(board);

    REQUIRE_FALSE(spell.isAlive());
    REQUIRE(enemy->hp == 500);
}

TEST_CASE("AreaSpell self-destructs on detonation even if nothing was in range", "[area_spell]") {
    Board board; // empty
    AreaSpell spell(1, 10.0f, 10.0f, 0, 3.0f, 500, 0);
    spell.update(board);
    REQUIRE_FALSE(spell.isAlive());
}

TEST_CASE("AreaSpell damages only alive, targetable, opposing-team entities within its radius", "[area_spell][radius]") {
    Board board;

    auto enemyAtBoundary = std::make_shared<DummyEntity>(1, 10.0f, 14.0f, 1000, 1); // dist == 4.0
    auto enemyJustOutside = std::make_shared<DummyEntity>(2, 10.0f, 14.1f, 1000, 1); // dist == 4.1
    auto ally = std::make_shared<DummyEntity>(3, 10.0f, 11.0f, 1000, 0); // same team as caster
    auto nonTargetableEnemy = std::make_shared<DummyEntity>(4, 10.0f, 11.0f, 1000, 1);
    nonTargetableEnemy->targetable = false;
    auto deadEnemy = std::make_shared<DummyEntity>(5, 10.0f, 11.0f, 100, 1);
    deadEnemy->takeDamage(100); // already dead before the spell resolves

    spawn(board, enemyAtBoundary);
    spawn(board, enemyJustOutside);
    spawn(board, ally);
    spawn(board, nonTargetableEnemy);
    spawn(board, deadEnemy);

    AreaSpell spell(6, 10.0f, 10.0f, 0, 4.0f, 500, 0);
    spell.update(board);

    REQUIRE(enemyAtBoundary->hp == 500);   // exactly at the radius boundary: included
    REQUIRE(enemyJustOutside->hp == 1000); // just beyond: excluded
    REQUIRE(ally->hp == 1000);             // same team: excluded
    REQUIRE(nonTargetableEnemy->hp == 1000); // not targetable: excluded
    REQUIRE(deadEnemy->hp == 0);            // already dead: left untouched, not further reduced
}
