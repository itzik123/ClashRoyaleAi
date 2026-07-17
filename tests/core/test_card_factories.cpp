#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardFactories.h"
#include "CardStats.h"
#include "MeleeTroop.h"
#include "RangedTroop.h"
#include "BuildingTargeter.h"
#include "RangedBuildingTargeter.h"
#include "Building.h"

// No registered card flies yet (this plan only wires the mechanism), so
// these hand-build a CardStats and call the factory functions directly
// instead of going through CardRegistry lookups.

namespace {
    CardStats makeTroopStats() {
        CardStats stats;
        stats.name = "TestFlier";
        stats.hp = 100;
        stats.speed = 1.0f;
        stats.attackRange = 1.0f;
        stats.damage = 10;
        stats.attackCooldown = 10;
        stats.symbol = 'K';
        return stats;
    }
}

TEST_CASE("applyCardMetadata copies isFlying/targetsAir onto the spawned entity", "[card_factories][flying]") {
    Board board;
    CardStats stats = makeTroopStats();
    stats.isFlying = true;
    stats.targetsAir = true;

    SECTION("spawnMeleeSquad") {
        CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->isFlying);
        REQUIRE(troop->targetsAir);
    }

    SECTION("spawnRangedSquad") {
        CardFactories::spawnRangedSquad(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<RangedTroop>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->isFlying);
        REQUIRE(troop->targetsAir);
    }

    SECTION("spawnMeleeBuildingTargeter") {
        CardFactories::spawnMeleeBuildingTargeter(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<BuildingTargeter>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->isFlying);
        REQUIRE(troop->targetsAir);
    }

    SECTION("spawnRangedBuildingTargeter") {
        CardFactories::spawnRangedBuildingTargeter(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<RangedBuildingTargeter>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->isFlying);
        REQUIRE(troop->targetsAir);
    }

    SECTION("spawnDefensiveBuilding") {
        CardFactories::spawnDefensiveBuilding(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board.getEntities().back());
        REQUIRE(building != nullptr);
        REQUIRE(building->isFlying);
        REQUIRE(building->targetsAir);
    }
}

TEST_CASE("Troop-shaped factories derive ignoresRiver from isFlying, even when ignoresRiver itself is false", "[card_factories][flying][river]") {
    Board board;
    CardStats stats = makeTroopStats();
    stats.isFlying = true;
    REQUIRE_FALSE(stats.ignoresRiver);

    SECTION("spawnMeleeSquad") {
        CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->riverIgnores);
    }

    SECTION("spawnRangedSquad") {
        CardFactories::spawnRangedSquad(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<RangedTroop>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->riverIgnores);
    }

    SECTION("spawnMeleeBuildingTargeter") {
        CardFactories::spawnMeleeBuildingTargeter(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<BuildingTargeter>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->riverIgnores);
    }

    SECTION("spawnRangedBuildingTargeter") {
        CardFactories::spawnRangedBuildingTargeter(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<RangedBuildingTargeter>(board.getEntities().back());
        REQUIRE(troop != nullptr);
        REQUIRE(troop->riverIgnores);
    }
}

TEST_CASE("A non-flying card leaves riverIgnores false unless set explicitly", "[card_factories][river]") {
    Board board;
    CardStats stats = makeTroopStats();
    // isFlying stays false, ignoresRiver stays false (default).

    CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities().back());
    REQUIRE(troop != nullptr);
    REQUIRE_FALSE(troop->riverIgnores);
}
