#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardFactories.h"
#include "CardStats.h"
#include "MeleeTroop.h"
#include "RangedTroop.h"
#include "BuildingTargeter.h"
#include "RangedBuildingTargeter.h"
#include "Building.h"
#include "AreaSpell.h"
#include "FreezeOnHit.h"
#include "SpawnOnDeath.h"

// These hand-build a CardStats and call the factory functions directly,
// exercising the mechanism in isolation from any single registered card's
// specific data (see test_card_registry.cpp for the end-to-end wiring).

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

TEST_CASE("CardFactories::placementRadius matches each archetype's real spawned footprint", "[card_factories][placement]") {
    REQUIRE(CardFactories::placementRadius(Archetype::MeleeSquad) == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE(CardFactories::placementRadius(Archetype::RangedSquad) == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE(CardFactories::placementRadius(Archetype::MeleeBuildingTargeter) == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE(CardFactories::placementRadius(Archetype::RangedBuildingTargeter) == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));
    REQUIRE(CardFactories::placementRadius(Archetype::DefensiveBuilding) == Catch::Approx(Building::COLLISION_RADIUS));
}

TEST_CASE("spawnDeployEffect is a no-op when spawnEffectRadius is unset (default)", "[card_factories][spawn_effect]") {
    Board board;
    CardStats stats = makeTroopStats(); // spawnEffectRadius defaults to 0

    CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1); // just the troop, no extra deploy-effect entity
}

TEST_CASE("spawnDeployEffect adds exactly one deploy-effect entity per card, regardless of squad size", "[card_factories][spawn_effect]") {
    Board board;
    CardStats stats = makeTroopStats();
    stats.spawnOffsets = { {0.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f} }; // squad of 3
    stats.withSpawnEffect(3.0f, 50);

    CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 4); // 3 troops + exactly 1 deploy-effect
    auto effect = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(effect != nullptr);
}

TEST_CASE("spawnDeployEffect's AreaSpell carries the configured damage and on-hit effect", "[card_factories][spawn_effect]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 6.0f, 5.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardStats stats = makeTroopStats();
    stats.withSpawnEffect(3.0f, 50, std::make_shared<FreezeOnHit>(5, 0.0f));

    CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto effect = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(effect != nullptr);
    effect->update(board); // zero delay: detonates immediately

    REQUIRE(enemy->hp == 950);
    REQUIRE(enemy->freezeTicks == 5);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.0f));
}

TEST_CASE("applyCardMetadata copies deathEffect onto the spawned entity", "[card_factories][death]") {
    Board board;
    CardStats stats = makeTroopStats();
    stats.withDeathEffect(std::make_shared<SpawnOnDeath>(makeTroopStats()));

    CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities().back());
    REQUIRE(troop != nullptr);
    REQUIRE(troop->deathEffect != nullptr);
}

TEST_CASE("SpawnOnDeath spawns its child CardStats at the dying entity's position and team", "[card_factories][death]") {
    Board board;
    CardStats childStats = makeTroopStats();
    childStats.name = "Child";
    SpawnOnDeath effect(childStats);

    effect.apply(board, Vector2D{ 7.0f, 8.0f }, 1);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1);
    auto child = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities()[0]);
    REQUIRE(child != nullptr);
    REQUIRE(child->name == "Child");
    REQUIRE(child->team == 1);
    REQUIRE(child->position.x == Catch::Approx(7.0f));
    REQUIRE(child->position.y == Catch::Approx(8.0f));
}

TEST_CASE("spawnSpell wires spellGroundOnly through to the spawned AreaSpell", "[card_factories][flying]") {
    Board board;
    auto flyingEnemy = std::make_shared<DummyEntity>(99, 5.0f, 5.5f, 1000, 1);
    flyingEnemy->isFlying = true;
    spawn(board, flyingEnemy);

    CardStats stats;
    stats.name = "TestGroundSpell";
    stats.archetype = Archetype::Spell;
    stats.spellRadius = 3.0f;
    stats.damage = 500;
    stats.spellDelayTicks = 0;
    stats.symbol = '*';
    stats.withGroundOnly();

    CardFactories::spawnSpell(stats, 5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    spell->update(board); // zero delay: detonates immediately

    REQUIRE(flyingEnemy->hp == 1000); // untouched: ground-only spell, flying target
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
