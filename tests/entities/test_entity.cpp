#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Building.h"

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

TEST_CASE("Entity::isBuilding defaults false, true only for Building", "[entity]") {
    DummyEntity e(1, 0, 0, 10, 0);
    REQUIRE_FALSE(e.isBuilding());

    Building b(2, 0, 0, 10, 0, 'C', 5.0f, 10, 10);
    REQUIRE(b.isBuilding());
}

// ---------------- pullToward / pushAway ----------------

TEST_CASE("pullToward moves an entity up to `distance` tiles toward a point", "[entity][pull]") {
    DummyEntity e(1, 8.0f, 5.0f, 10, 0); // dist 3.0 from (5,5), along +x
    pullToward(e, Vector2D{ 5.0f, 5.0f }, 1.5f);
    REQUIRE(e.position.x == Catch::Approx(6.5f));
    REQUIRE(e.position.y == Catch::Approx(5.0f));
}

TEST_CASE("pullToward never overshoots past the point", "[entity][pull]") {
    DummyEntity e(1, 6.0f, 5.0f, 10, 0); // dist 1.0 from (5,5)
    pullToward(e, Vector2D{ 5.0f, 5.0f }, 5.0f); // would overshoot by 4.0 if unclamped
    REQUIRE(e.position.x == Catch::Approx(5.0f));
    REQUIRE(e.position.y == Catch::Approx(5.0f));
}

TEST_CASE("pushAway moves an entity exactly `distance` tiles away from a point", "[entity][push]") {
    DummyEntity e(1, 6.0f, 5.0f, 10, 0); // dist 1.0 from (5,5), along +x
    pushAway(e, Vector2D{ 5.0f, 5.0f }, 2.0f);
    REQUIRE(e.position.x == Catch::Approx(8.0f));
    REQUIRE(e.position.y == Catch::Approx(5.0f));
}

TEST_CASE("pullToward and pushAway are both no-ops on a Building, regardless of caller",
        "[entity][pull][push][building]") {
    Building b(1, 6.0f, 5.0f, 1000, 0, 'C', 5.0f, 10, 10);

    pullToward(b, Vector2D{ 5.0f, 5.0f }, 5.0f);
    REQUIRE(b.position.x == Catch::Approx(6.0f));
    REQUIRE(b.position.y == Catch::Approx(5.0f));

    pushAway(b, Vector2D{ 5.0f, 5.0f }, 5.0f);
    REQUIRE(b.position.x == Catch::Approx(6.0f));
    REQUIRE(b.position.y == Catch::Approx(5.0f));
}
