#include <catch_amalgamated.hpp>
#include "test_helpers.h"

TEST_CASE("Vector2D::distanceTo computes Euclidean distance", "[entity][vector2d]") {
    Vector2D a{ 0.0f, 0.0f };
    Vector2D b{ 3.0f, 4.0f };
    REQUIRE(a.distanceTo(b) == Catch::Approx(5.0f));
    REQUIRE(b.distanceTo(a) == Catch::Approx(5.0f));
    REQUIRE(a.distanceTo(a) == Catch::Approx(0.0f));
}

TEST_CASE("Entity constructor initializes all fields", "[entity]") {
    DummyEntity e(42, 3.5f, 7.5f, 100, 1, 'X');

    REQUIRE(e.id == 42);
    REQUIRE(e.position.x == Catch::Approx(3.5f));
    REQUIRE(e.position.y == Catch::Approx(7.5f));
    REQUIRE(e.hp == 100);
    REQUIRE(e.team == 1);
    REQUIRE(e.symbol == 'X');
    REQUIRE(e.name.empty());
    REQUIRE_FALSE(e.isFlying);
    REQUIRE(e.cardId == -1);
}

TEST_CASE("Entity::isAlive reflects hp", "[entity]") {
    DummyEntity e(1, 0, 0, 10, 0);
    REQUIRE(e.isAlive());

    e.takeDamage(10);
    REQUIRE(e.hp == 0);
    REQUIRE_FALSE(e.isAlive());

    DummyEntity e2(2, 0, 0, 5, 0);
    e2.takeDamage(999);
    REQUIRE(e2.hp < 0);
    REQUIRE_FALSE(e2.isAlive());
}

TEST_CASE("Entity::takeDamage subtracts the exact amount", "[entity]") {
    DummyEntity e(1, 0, 0, 100, 0);
    e.takeDamage(30);
    REQUIRE(e.hp == 70);
    e.takeDamage(1);
    REQUIRE(e.hp == 69);
}

TEST_CASE("Entity default isTargetable/getCollisionRadius", "[entity]") {
    DummyEntity e(1, 0, 0, 10, 0);
    REQUIRE(e.isTargetable());
    REQUIRE(e.getCollisionRadius() == Catch::Approx(0.0f));
}

TEST_CASE("Entity::isTargetable can be overridden false by subclasses", "[entity]") {
    DummyEntity e(1, 0, 0, 10, 0);
    e.targetable = false;
    REQUIRE_FALSE(e.isTargetable());
}
