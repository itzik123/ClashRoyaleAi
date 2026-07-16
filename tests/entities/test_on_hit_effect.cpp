#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "OnHitEffect.h"

TEST_CASE("FreezeOnHit applies the configured freeze to its target", "[on_hit_effect]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 0.0f, 100, 1);

    FreezeOnHit freeze(30, 0.65f);
    freeze.apply(target);

    REQUIRE(target->freezeTicks == 30);
    REQUIRE(target->freezeSlow == Catch::Approx(0.65f));
}

TEST_CASE("CombatEntity fires its on-hit effects exactly once per successful attack", "[on_hit_effect][combat_entity]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    attacker->addOnHitEffect(std::make_shared<FreezeOnHit>(30, 0.65f));

    attacker->update(board); // attacks once
    REQUIRE(attacker->attackCount == 1);
    REQUIRE(enemy->freezeTicks == 30);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.65f));

    // Cooldown just reset -- no attack (and no effect) on the very next tick.
    enemy->freezeTicks = 5; // sentinel: would be overwritten to 30 again if the effect fired
    attacker->update(board);
    REQUIRE(attacker->attackCount == 1);
    REQUIRE(enemy->freezeTicks == 5);
}

TEST_CASE("CombatEntity with no on-hit effects behaves exactly as before", "[on_hit_effect][combat_entity]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
    REQUIRE(enemy->freezeTicks == 0); // untouched: no effect was ever attached
}

TEST_CASE("Multiple on-hit effects all fire on the same attack", "[on_hit_effect][combat_entity]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    attacker->addOnHitEffect(std::make_shared<FreezeOnHit>(10, 0.9f));
    attacker->addOnHitEffect(std::make_shared<FreezeOnHit>(30, 0.65f));

    attacker->update(board);

    // Both fired; Entity::applyFreeze's own max/min merge logic decides the result.
    REQUIRE(enemy->freezeTicks == 30);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.65f));
}
