#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Building.h"
#include "Tower.h"
#include "BuildingTargeter.h"
#include "RangedBuildingTargeter.h"
#include "Projectile.h"

// ---------------- Building ----------------

TEST_CASE("Building::getCollisionRadius is always 1.0", "[building]") {
    Building b(1, 5.0f, 5.0f, 1000, 0, 'C', 5.0f, 100, 10);
    REQUIRE(b.getCollisionRadius() == Catch::Approx(1.0f));
}

TEST_CASE("Building::isBuilding is true, unlike the Entity default", "[building]") {
    Building b(1, 5.0f, 5.0f, 1000, 0, 'C', 5.0f, 100, 10);
    REQUIRE(b.isBuilding());
    REQUIRE_FALSE(b.isTower()); // a plain Building is not a Tower

    DummyEntity notABuilding(2, 5.0f, 5.0f, 1000, 0);
    REQUIRE_FALSE(notABuilding.isBuilding());
}

TEST_CASE("Tower::isTower is true, unlike the Entity default and plain Building", "[building][tower]") {
    Tower t(1, 5.0f, 5.0f, 4008, 0, 7.0f, 90, 10, 'R');
    REQUIRE(t.isTower());
    REQUIRE(t.isBuilding()); // still a Building too (inherited)
}

TEST_CASE("Building never moves, even with a target far out of range", "[building][movement]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 50.0f, 100, 1);
    spawn(board, enemy);

    Building building(2, 5.0f, 5.0f, 1000, 0, 'C', 5.0f, 100, 10);
    for (int i = 0; i < 5; ++i) building.update(board);

    REQUIRE(building.position.x == Catch::Approx(5.0f));
    REQUIRE(building.position.y == Catch::Approx(5.0f));
}

TEST_CASE("Building::performAttack deals direct damage with no projectile", "[building][attack]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 7.0f, 100, 1); // within range
    spawn(board, enemy);

    Building building(2, 5.0f, 5.0f, 1000, 0, 'C', 5.0f, 127, 8);
    size_t countBefore = board.getEntities().size();
    building.update(board);
    board.commitPendingEntities();

    REQUIRE(enemy->hp == -27); // 100 - 127
    REQUIRE(board.getEntities().size() == countBefore);
}

TEST_CASE("Building decays over its lifetime and does not attack-decay before the interval", "[building][decay]") {
    Board board; // no enemies: isolates decay from combat

    // hp=3000, lifetime=300 => decays maxHp/(300/10) = 100 every 10 ticks.
    Building building(1, 5.0f, 5.0f, 3000, 0, 'C', 5.0f, 100, 10, 300);

    for (int i = 0; i < 9; ++i) building.update(board);
    REQUIRE(building.hp == 3000); // no decay yet, interval not reached

    building.update(board); // 10th tick: first decay
    REQUIRE(building.hp == 2900);

    for (int i = 0; i < 10; ++i) building.update(board); // 20th tick: second decay
    REQUIRE(building.hp == 2800);
}

TEST_CASE("Building fully decays to 0 exactly at its configured lifetime", "[building][decay]") {
    Board board;
    Building building(1, 5.0f, 5.0f, 3000, 0, 'C', 5.0f, 100, 10, 300);

    for (int i = 0; i < 300; ++i) building.update(board);

    REQUIRE(building.hp == 0);
    REQUIRE_FALSE(building.isAlive());
}

TEST_CASE("Building with lifetime <= 0 never decays", "[building][decay]") {
    Board board;
    Building building(1, 5.0f, 5.0f, 1000, 0, 'C', 5.0f, 100, 10, -1);

    for (int i = 0; i < 500; ++i) building.update(board);

    REQUIRE(building.hp == 1000);
}

// ---------------- Tower ----------------

TEST_CASE("Tower::getCollisionRadius depends on symbol (King vs Princess)", "[tower]") {
    Tower king(1, 9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R');
    Tower princess(2, 3.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P');

    REQUIRE(king.getCollisionRadius() == Catch::Approx(2.0f));
    REQUIRE(princess.getCollisionRadius() == Catch::Approx(1.5f));
}

TEST_CASE("Tower never decays regardless of ticks elapsed", "[tower][decay]") {
    Board board;
    Tower tower(1, 9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R');

    for (int i = 0; i < 500; ++i) tower.update(board);

    REQUIRE(tower.hp == 4008);
}

TEST_CASE("Tower::performAttack spawns a projectile instead of dealing direct damage", "[tower][attack]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 9.0f, 8.0f, 100, 1); // within the 7.0 range + radii
    spawn(board, enemy);

    Tower tower(2, 9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R');
    size_t countBefore = board.getEntities().size();
    tower.update(board);
    board.commitPendingEntities();

    REQUIRE(enemy->hp == 100); // untouched until the projectile arrives
    REQUIRE(board.getEntities().size() == countBefore + 1);
    auto projectile = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(projectile != nullptr);
}

TEST_CASE("Tower defaults to targetsAir true, since every tower defends against air", "[tower][flying]") {
    Tower king(1, 9.0f, 2.0f, 4008, 0, 7.0f, 90, 10, 'R');
    Tower princess(2, 3.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P');

    REQUIRE(king.targetsAir);
    REQUIRE(princess.targetsAir);
}

// ---------------- BuildingTargeter ----------------

TEST_CASE("BuildingTargeter ignores enemy troops and walks past them toward a building", "[building_targeter][targeting]") {
    Board board;
    auto decoyTroop = std::make_shared<DummyEntity>(1, 5.0f, 5.3f, 100, 1); // right next to it
    auto enemyBuilding = std::make_shared<Building>(2, 5.0f, 10.0f, 1000, 1, 'C', 5.0f, 10, 10); // dist 5.0: farther than the decoy, still within default sightRange
    spawn(board, decoyTroop);
    spawn(board, enemyBuilding);

    BuildingTargeter targeter(3, 5.0f, 5.0f, 3344, 0, 0.3f, 1.5f, 211, 15, 'G');
    targeter.update(board);

    REQUIRE(decoyTroop->hp == 100); // never attacked
    REQUIRE(targeter.position.y == Catch::Approx(5.3f)); // moved toward the building, not stuck fighting the troop
}

TEST_CASE("BuildingTargeter with no enemy buildings on the board never moves or attacks", "[building_targeter][targeting]") {
    Board board;
    auto decoyTroop = std::make_shared<DummyEntity>(1, 5.0f, 5.3f, 100, 1);
    spawn(board, decoyTroop);

    BuildingTargeter targeter(2, 5.0f, 5.0f, 3344, 0, 0.3f, 1.5f, 211, 15, 'G');
    targeter.update(board);

    REQUIRE(targeter.position.x == Catch::Approx(5.0f));
    REQUIRE(targeter.position.y == Catch::Approx(5.0f));
    REQUIRE(decoyTroop->hp == 100);
}

TEST_CASE("BuildingTargeter::performAttack deals direct damage with no projectile", "[building_targeter][attack]") {
    Board board;
    auto enemyBuilding = std::make_shared<Building>(1, 5.0f, 7.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, enemyBuilding);

    BuildingTargeter targeter(2, 5.0f, 5.0f, 3344, 0, 0.3f, 1.5f, 200, 15, 'G');
    size_t countBefore = board.getEntities().size();
    targeter.update(board);
    board.commitPendingEntities();

    REQUIRE(enemyBuilding->hp == 800);
    REQUIRE(board.getEntities().size() == countBefore);
}

// ---------------- RangedBuildingTargeter ----------------

TEST_CASE("RangedBuildingTargeter ignores enemy troops just like BuildingTargeter", "[ranged_building_targeter][targeting]") {
    Board board;
    auto decoyTroop = std::make_shared<DummyEntity>(1, 5.0f, 5.3f, 100, 1);
    // Placed far enough out (dist 9) to stay beyond this unit's much larger
    // 6.5 attack range (effective ~7.9), so it still has to walk toward it.
    auto enemyBuilding = std::make_shared<Building>(2, 5.0f, 14.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, decoyTroop);
    spawn(board, enemyBuilding);

    RangedBuildingTargeter targeter(3, 5.0f, 5.0f, 2544, 0, 0.3f, 6.5f, 159, 17, 'Y');
    targeter.sightRange = 10.0f; // real cards set this explicitly (own attackRange + ~0.5); the default alone wouldn't reach this far
    targeter.update(board);

    REQUIRE(decoyTroop->hp == 100);
    REQUIRE(targeter.position.y == Catch::Approx(5.3f));
}

TEST_CASE("RangedBuildingTargeter::performAttack spawns a projectile instead of dealing direct damage", "[ranged_building_targeter][attack]") {
    Board board;
    auto enemyBuilding = std::make_shared<Building>(1, 5.0f, 7.0f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, enemyBuilding);

    RangedBuildingTargeter targeter(2, 5.0f, 5.0f, 2544, 0, 0.3f, 6.5f, 159, 17, 'Y');
    size_t countBefore = board.getEntities().size();
    targeter.update(board);
    board.commitPendingEntities();

    REQUIRE(enemyBuilding->hp == 1000); // untouched until the projectile arrives
    REQUIRE(board.getEntities().size() == countBefore + 1);
    auto projectile = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(projectile != nullptr);
}
