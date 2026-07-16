#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Projectile.h"

TEST_CASE("Projectile is never itself a valid combat target", "[projectile]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1);
    spawn(board, target);
    Projectile p(2, 0.0f, 0.0f, 0, target, 1.5f, 50);
    REQUIRE_FALSE(p.isTargetable());
}

TEST_CASE("Projectile homes toward the target's current position each tick", "[projectile][movement]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 1.5f, 50);
    p.update(board);

    REQUIRE(p.position.x == Catch::Approx(0.0f));
    REQUIRE(p.position.y == Catch::Approx(1.5f));
}

TEST_CASE("Projectile re-homes when the target moves between ticks", "[projectile][movement]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 10.0f, 0.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 50);
    p.update(board); // heads toward (10, 0) -> lands at (2, 0)
    REQUIRE(p.position.x == Catch::Approx(2.0f));
    REQUIRE(p.position.y == Catch::Approx(0.0f));

    target->position = { 2.0f, 10.0f }; // target relocates
    p.update(board); // must now head toward the new position, not continue along x

    REQUIRE(p.position.x == Catch::Approx(2.0f));
    REQUIRE(p.position.y == Catch::Approx(2.0f));
}

TEST_CASE("Projectile deals damage and dies the instant it is within one tick's travel of the target", "[projectile][arrival]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1);
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 2.0f, 275); // dist(2.0) == speed(2.0)
    p.update(board);

    REQUIRE(target->hp == 725);
    REQUIRE_FALSE(p.isAlive());
    // Snaps to the impact point on the tick it lands, instead of dying
    // one step short of the target it just hit.
    REQUIRE(p.position.x == Catch::Approx(0.0f));
    REQUIRE(p.position.y == Catch::Approx(2.0f));
}

TEST_CASE("Projectile dies without dealing damage if its target already died first", "[projectile][arrival]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    target->takeDamage(100); // dead before the projectile resolves
    spawn(board, target);

    Projectile p(2, 0.0f, 0.0f, 0, target, 5.0f, 50);
    p.update(board);

    REQUIRE_FALSE(p.isAlive());
    REQUIRE(target->hp == 0); // not driven further negative
}

TEST_CASE("Projectile dies without crashing if its target object no longer exists", "[projectile][arrival]") {
    Board board;
    std::weak_ptr<Entity> weakTarget;
    {
        auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
        weakTarget = target;
        // target's shared_ptr goes out of scope here without ever being
        // added to the board, so the weak_ptr can no longer be locked.
    }

    Projectile p(2, 0.0f, 0.0f, 0, weakTarget, 5.0f, 50);
    REQUIRE_NOTHROW(p.update(board));
    REQUIRE_FALSE(p.isAlive());
}
