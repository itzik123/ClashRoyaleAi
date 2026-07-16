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
    REQUIRE(e.freezeTicks == 0);
    REQUIRE(e.freezeSlow == Catch::Approx(1.0f));
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

TEST_CASE("Entity::applyFreeze", "[entity][freeze]") {
    DummyEntity e(1, 0, 0, 10, 0);

    SECTION("first application sets ticks and slow factor") {
        e.applyFreeze(30, 0.65f);
        REQUIRE(e.freezeTicks == 30);
        REQUIRE(e.freezeSlow == Catch::Approx(0.65f));
    }

    SECTION("a strictly longer duration overrides the current freeze") {
        e.applyFreeze(20, 0.8f);
        e.applyFreeze(30, 0.5f);
        REQUIRE(e.freezeTicks == 30);
        REQUIRE(e.freezeSlow == Catch::Approx(0.5f));
    }

    SECTION("a shorter but stronger freeze keeps the longer duration and adopts the stronger slow") {
        // Duration and strength are tracked independently: the target keeps
        // whichever duration is longer, but a stronger (lower) slow factor
        // always takes effect even if it came from the shorter application.
        e.applyFreeze(30, 0.65f);
        e.applyFreeze(10, 0.1f);
        REQUIRE(e.freezeTicks == 30);
        REQUIRE(e.freezeSlow == Catch::Approx(0.1f));
    }

    SECTION("a weaker reapplication never makes the slow effect weaker") {
        e.applyFreeze(30, 0.1f);
        e.applyFreeze(30, 0.65f);
        REQUIRE(e.freezeTicks == 30);
        REQUIRE(e.freezeSlow == Catch::Approx(0.1f));
    }
}

TEST_CASE("Entity::isTargetable can be overridden false by subclasses", "[entity]") {
    DummyEntity e(1, 0, 0, 10, 0);
    e.targetable = false;
    REQUIRE_FALSE(e.isTargetable());
}
