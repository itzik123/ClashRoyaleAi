#include <catch_amalgamated.hpp>
#include <cmath>
#include "test_helpers.h"
#include "MeleeTroop.h"
#include "RangedTroop.h"
#include "Projectile.h"
#include "Building.h"

// Troop is abstract (inherits CombatEntity's pure virtual performAttack), so
// MeleeTroop stands in for testing Troop's own movement/clamp behavior --
// MeleeTroop adds nothing on top of Troop except a trivial direct-damage
// attack, which is covered separately below.

TEST_CASE("Troop moves in a straight line when target is on the same side of the river", "[troop][movement]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 10.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 5.0f, 5.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    troop->update(board);

    // Straight vertical approach: x must not drift.
    REQUIRE(troop->position.x == Catch::Approx(5.0f));
    REQUIRE(troop->position.y == Catch::Approx(6.0f)); // moved exactly `speed` (1.0) toward target
}

TEST_CASE("Troop routes through the nearest bridge when crossing the river", "[troop][movement][river]") {
    Board board;
    // Target is across the river; neither point is near x=10, so the router
    // must pick a bridge waypoint instead of a straight line.
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 20.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 10.0f, 10.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    troop->sightRange = 10.0f; // enemy is placed at dist 10, beyond the default; this test is about river routing, not sight
    troop->update(board);

    // The right bridge is closer than the left from (10,10), so the troop must
    // have drifted toward it rather than staying at x=10.
    //
    // Asserted as an INVARIANT, not as arithmetic. This used to pin a
    // hand-computed post-move position (10.588172f), which silently became
    // wrong -- and needed re-deriving by hand -- the moment the bridge moved.
    // "It closes on the bridge it chose, and it heads upfield" is the property
    // the test was really for, and it cannot go stale.
    const float bridgeX = board.getRightBridge().x;
    REQUIRE(std::fabs(troop->position.x - bridgeX) < std::fabs(10.0f - bridgeX));
    REQUIRE(troop->position.x > 10.0f);
    REQUIRE(troop->position.y > 10.0f);
}

TEST_CASE("Troop with ignoresRiver set walks straight through the river band", "[troop][movement][river]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 20.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 10.0f, 10.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'H');
    troop->setIgnoresRiver(true);
    troop->sightRange = 10.0f; // enemy is placed at dist 10, beyond the default; this test is about river routing, not sight
    troop->update(board);

    // No bridge detour: x stays put, only y advances toward the target.
    REQUIRE(troop->position.x == Catch::Approx(10.0f));
    REQUIRE(troop->position.y == Catch::Approx(11.0f));
}

TEST_CASE("Freeze slows movement speed by the slow factor", "[troop][movement][freeze]") {
    Board board;
    // Kept on the same side of the river as the troop (y <= 15) so movement
    // is a straight line and isolates the speed effect from bridge routing.
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    troop->sightRange = 10.0f; // enemy is placed at dist 10, beyond the default; this test is about freeze, not sight
    troop->update(board); // unfrozen: moves full `speed` (1.0)
    REQUIRE(troop->position.y == Catch::Approx(1.0f));

    troop->applyFreeze(50, 0.5f);
    troop->update(board); // frozen: moves speed * 0.5
    REQUIRE(troop->position.y == Catch::Approx(1.5f));
}

TEST_CASE("Troop::clampPosition keeps troops within board bounds", "[troop][clamp]") {
    Board board; // empty: no target, so only the unconditional clamp runs

    SECTION("clamps negative x/overshoot y to the board edges") {
        auto troop = std::make_shared<MeleeTroop>(1, 0.0f, 0.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->position = { -5.0f, 40.0f };
        troop->update(board);
        REQUIRE(troop->position.x == Catch::Approx(0.0f));
        REQUIRE(troop->position.y == Catch::Approx(33.0f));
    }

    SECTION("clamps overshoot x to the right edge") {
        auto troop = std::make_shared<MeleeTroop>(1, 0.0f, 0.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->position = { 25.0f, 10.0f };
        troop->update(board);
        REQUIRE(troop->position.x == Catch::Approx(17.0f));
    }
}

TEST_CASE("Troop::clampPosition pushes non-bridge river-band positions out to the nearest bank", "[troop][clamp][river]") {
    Board board;

    SECTION("y below the 16.5 midpoint snaps down to the river start (15.5)") {
        auto troop = std::make_shared<MeleeTroop>(1, 10.0f, 16.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(15.5f));
    }

    SECTION("y at or above the 16.5 midpoint snaps up to the river end (17.5)") {
        auto troop = std::make_shared<MeleeTroop>(1, 10.0f, 17.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(17.5f));
    }
}

TEST_CASE("Troop::clampPosition does not snap positions sitting on a bridge column", "[troop][clamp][river]") {
    Board board;

    SECTION("left bridge") {
        auto troop = std::make_shared<MeleeTroop>(1, board.getLeftBridge().x, 17.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(17.0f));
    }

    SECTION("right bridge") {
        auto troop = std::make_shared<MeleeTroop>(1, board.getRightBridge().x, 17.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(17.0f));
    }
}

TEST_CASE("ignoresRiver also suppresses the river-band clamp", "[troop][clamp][river]") {
    Board board;
    auto troop = std::make_shared<MeleeTroop>(1, 10.0f, 17.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'H');
    troop->setIgnoresRiver(true);
    troop->update(board);
    REQUIRE(troop->position.y == Catch::Approx(17.0f)); // not snapped, despite being off-bridge
}

TEST_CASE("A flying troop ignores building collision and flies straight through", "[troop][movement][flying]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 10.0f, 100, 1);
    spawn(board, enemy);

    // Sits directly on the straight-line path from the troop to the enemy;
    // a grounded troop would be pushed off course by resolvePositionAgainstBuildings.
    auto obstacle = std::make_shared<Building>(3, 5.0f, 5.5f, 1000, 0, 'C', 5.0f, 10, 10);
    spawn(board, obstacle);

    auto troop = std::make_shared<MeleeTroop>(2, 5.0f, 5.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    troop->isFlying = true;
    troop->update(board);

    // Straight line toward the enemy (5,10): x unchanged, y advances by `speed` (1.0),
    // completely unaffected by the building sitting right in its path.
    REQUIRE(troop->position.x == Catch::Approx(5.0f));
    REQUIRE(troop->position.y == Catch::Approx(6.0f));
}

TEST_CASE("MeleeTroop::performAttack deals direct damage with no projectile spawned", "[melee_troop][attack]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 0.5f, 1.5f, 40, 11, 'K');
    size_t countBefore = board.getEntities().size();
    troop->update(board);
    board.commitPendingEntities();

    REQUIRE(enemy->hp == 60);
    REQUIRE(board.getEntities().size() == countBefore); // no projectile added
}

TEST_CASE("RangedTroop::performAttack spawns a projectile instead of dealing direct damage", "[ranged_troop][attack]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 4.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<RangedTroop>(2, 0.0f, 0.0f, 100, 0, 0.5f, 5.0f, 86, 12, 'A');
    size_t countBefore = board.getEntities().size();
    troop->update(board);
    board.commitPendingEntities();

    REQUIRE(enemy->hp == 100); // not yet damaged -- the projectile carries the damage
    REQUIRE(board.getEntities().size() == countBefore + 1);

    auto projectile = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(projectile != nullptr);
    REQUIRE(projectile->team == 0);
    REQUIRE_FALSE(projectile->isTargetable());
}

// ---------------- clone (Clone spell) ----------------

TEST_CASE("MeleeTroop::clone produces a fresh-id, 1-hp copy that keeps the original's combat stats", "[troop][clone]") {
    auto original = std::make_shared<MeleeTroop>(5, 3.0f, 4.0f, 500, 1, 0.5f, 1.2f, 200, 12, 'K');
    original->splashRadius = 1.5f; // any configured field should carry over

    auto copy = original->clone(999);
    auto meleeCopy = std::dynamic_pointer_cast<MeleeTroop>(copy);

    REQUIRE(meleeCopy != nullptr);
    REQUIRE(meleeCopy->id == 999);
    REQUIRE(meleeCopy->hp == 1);
    REQUIRE(meleeCopy->team == 1);
    REQUIRE(meleeCopy->position.x == Catch::Approx(3.0f));
    REQUIRE(meleeCopy->position.y == Catch::Approx(4.0f));
    REQUIRE(meleeCopy->splashRadius == Catch::Approx(1.5f));
}

TEST_CASE("Building has no clone() override -- Clone was never valid against buildings in the real game either", "[troop][clone]") {
    Building building(1, 0.0f, 0.0f, 500, 0, 'C', 5.0f, 100, 10);
    REQUIRE(building.clone(999) == nullptr);
}


TEST_CASE("a freeze slows movement for exactly as many ticks as it slows the cooldown",
          "[troop][movement][freeze][regression]") {
    // freezeTicks had TWO readers inside one update(), on opposite sides of
    // the decrement:
    //
    //   CombatEntity::update()  reads it, decrements it, drains the cooldown
    //   Troop::moveTowards()    reads it again, AFTER that decrement
    //
    // so applyFreeze(N) slowed the attack cooldown for N ticks and movement
    // for N-1. On the final tick of any freeze the unit was already moving at
    // full speed. At N = 1 -- and the engine registers 3-tick and 5-tick stuns
    // (Electro Spirit, Zap) -- the movement half of the freeze did not happen
    // at all.
    //
    // Same shape as the two absorbing states already recorded in CLAUDE.md:
    // one fact, two readers, and they disagreed.
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1);
    spawn(board, enemy);

    auto troop = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 1.0f, 1.0f, 10, 10, 'K');
    troop->sightRange = 10.0f;

    SECTION("a one-tick full stun stops movement for that tick") {
        troop->applyFreeze(1, 0.0f);
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(0.0f));
    }

    SECTION("an N-tick full stun stops movement for all N ticks") {
        constexpr int N = 4;
        troop->applyFreeze(N, 0.0f);
        for (int i = 0; i < N; ++i) troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(0.0f));
        REQUIRE(troop->freezeTicks == 0);

        // Control: it is not simply immobile -- the very next tick it walks.
        troop->update(board);
        REQUIRE(troop->position.y == Catch::Approx(1.0f));
    }
}
