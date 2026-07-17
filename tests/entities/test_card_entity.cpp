#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardEntity.h"
#include "AreaSpell.h"
#include "MeleeTroop.h"
#include "Building.h"
#include "Projectile.h"

TEST_CASE("Entity::name defaults to empty", "[entity][name]") {
    DummyEntity e(1, 0.0f, 0.0f, 100, 0);
    REQUIRE(e.name.empty());
}

TEST_CASE("Entity::name can be set directly", "[entity][name]") {
    DummyEntity e(1, 0.0f, 0.0f, 100, 0);
    e.name = "Test Dummy";
    REQUIRE(e.name == "Test Dummy");
}

TEST_CASE("AreaSpell is a CardEntity", "[card_entity][hierarchy]") {
    auto spell = std::make_shared<AreaSpell>(1, 0.0f, 0.0f, 0, 3.0f, 100, 5);
    REQUIRE(std::dynamic_pointer_cast<CardEntity>(spell) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Entity>(spell) != nullptr);
}

TEST_CASE("Troops and buildings (via CombatEntity) are CardEntities", "[card_entity][hierarchy]") {
    auto troop = std::make_shared<MeleeTroop>(1, 0.0f, 0.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    auto building = std::make_shared<Building>(2, 0.0f, 0.0f, 1000, 0, 'C', 5.0f, 100, 10);

    REQUIRE(std::dynamic_pointer_cast<CardEntity>(troop) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<CardEntity>(building) != nullptr);
}

TEST_CASE("Projectile is an Entity but not a CardEntity", "[card_entity][hierarchy]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1);
    spawn(board, target);

    auto projectile = std::make_shared<Projectile>(2, 0.0f, 0.0f, 0, target, 1.5f, 50);

    REQUIRE(std::dynamic_pointer_cast<Entity>(projectile) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<CardEntity>(projectile) == nullptr);
}
