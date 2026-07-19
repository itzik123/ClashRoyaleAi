#include <catch_amalgamated.hpp>
#include "test_helpers.h"

// StationaryCombatant: attackRange, no movement, no decay -- isolates
// CombatEntity::update()'s targeting/attack/cooldown/freeze logic.

TEST_CASE("CombatEntity::findTarget ignores same-team entities", "[combat_entity][targeting]") {
    Board board;
    auto ally = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 0);
    spawn(board, ally);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 0);
}

TEST_CASE("CombatEntity::findTarget ignores dead entities", "[combat_entity][targeting]") {
    Board board;
    auto deadEnemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    deadEnemy->takeDamage(100);
    spawn(board, deadEnemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 0);
}

TEST_CASE("CombatEntity::findTarget ignores non-targetable entities", "[combat_entity][targeting]") {
    Board board;
    auto decoy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    decoy->targetable = false;
    spawn(board, decoy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 0);
}

TEST_CASE("CombatEntity::findTarget picks the closest enemy", "[combat_entity][targeting]") {
    Board board;
    auto far = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 100, 1, 'F');
    auto near = std::make_shared<DummyEntity>(2, 0.0f, 1.0f, 100, 1, 'N');
    spawn(board, far);
    spawn(board, near);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 20.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
    REQUIRE(attacker->lastTargetId == near->id);
}

// ---------------- target-lock ----------------
// Once an attacker has picked a target, it stays committed to that fight
// instead of re-running "who's closest" every tick -- matches the real
// game, where a new enemy wandering closer mid-fight doesn't steal a
// unit's attention. The lock only breaks when the target dies or leaves
// effective range (e.g. a future knockback/pull effect), for every
// attacker alike -- mobile or stationary. See
// CombatEntity::update()/resolveCurrentTarget().

TEST_CASE("An attacker stays locked onto its target even when a closer enemy shows up mid-fight", "[combat_entity][targeting][lock]") {
    Board board;
    auto original = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1, 'O');
    spawn(board, original);

    // attackCooldown 1: attacks every single update() call, so attackCount/
    // lastTargetId reflect this tick's target choice, not stale data from
    // a still-cooling-down previous hit (same reasoning as
    // makeRampingAttacker's cooldown-1 choice below).
    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 20.0f, 50, 1);
    attacker->update(board); // locks onto the only enemy around

    REQUIRE(attacker->lastTargetId == original->id);

    // A much closer enemy arrives (e.g. Skeletons walking right up next to
    // the attacker) -- the real-game rule is that this does NOT steal the
    // attacker away from the fight it's already committed to.
    auto closer = std::make_shared<DummyEntity>(3, 0.0f, 0.1f, 1000, 1, 'C');
    spawn(board, closer);

    attacker->update(board);
    REQUIRE(attacker->lastTargetId == original->id); // still locked on the original target
    REQUIRE(closer->hp == 1000); // untouched
}

TEST_CASE("An attacker acquires a new target once its locked target dies", "[combat_entity][targeting][lock]") {
    Board board;
    auto original = std::make_shared<DummyEntity>(1, 0.0f, 2.0f, 1000, 1, 'O');
    spawn(board, original);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 20.0f, 50, 1);
    attacker->update(board);
    REQUIRE(attacker->lastTargetId == original->id);

    original->takeDamage(1000); // dies
    auto replacement = std::make_shared<DummyEntity>(3, 0.0f, 3.0f, 1000, 1, 'R');
    spawn(board, replacement);

    attacker->update(board);
    REQUIRE(attacker->lastTargetId == replacement->id); // the only valid target left
}

TEST_CASE("An attacker drops its lock and re-targets once the locked target leaves range", "[combat_entity][targeting][lock]") {
    Board board;
    // attackRange 1.0 -> effective range 1.8 (implicit radii on both sides);
    // 1.5 stays comfortably inside it.
    auto original = std::make_shared<DummyEntity>(1, 0.0f, 1.5f, 1000, 1, 'O');
    spawn(board, original);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 50, 1);
    attacker->update(board); // locks onto and attacks `original`
    REQUIRE(attacker->attackCount == 1);
    REQUIRE(attacker->lastTargetId == original->id);

    original->position = { 0.0f, 50.0f }; // pulled/knocked far out of range (e.g. Tornado)
    auto inRange = std::make_shared<DummyEntity>(3, 0.0f, 1.5f, 1000, 1, 'N'); // within effective range
    spawn(board, inRange);

    attacker->update(board);
    REQUIRE(attacker->lastTargetId == inRange->id); // dropped the unreachable lock, picked up the reachable one
    REQUIRE(attacker->attackCount == 2);
}

TEST_CASE("CombatEntity attacks only within effective range (attackRange + implicit radii)", "[combat_entity][range]") {
    Board board;

    SECTION("target just inside effective range is attacked") {
        // attackRange=1.0, implicit radius 0.4 for both sides => effective 1.8
        auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.7f, 100, 1);
        spawn(board, enemy);

        auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 50, 10);
        attacker->update(board);

        REQUIRE(attacker->attackCount == 1);
    }

    SECTION("target outside effective range is not attacked") {
        auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 100, 1);
        spawn(board, enemy);

        auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 50, 10);
        attacker->update(board);

        REQUIRE(attacker->attackCount == 0);
        REQUIRE(enemy->hp == 100);
    }
}

TEST_CASE("CombatEntity attack deals exactly the configured damage and resets cooldown", "[combat_entity][attack]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 37, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
    REQUIRE(enemy->hp == 63);

    // Cooldown just reset to 10 -- no second attack on the very next tick.
    attacker->update(board);
    REQUIRE(attacker->attackCount == 1);
    REQUIRE(enemy->hp == 63);
}

TEST_CASE("CombatEntity re-attacks once cooldown fully elapses", "[combat_entity][cooldown]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 10, 3);
    attacker->update(board); // tick 1: attacks, cooldown -> 3
    REQUIRE(attacker->attackCount == 1);

    attacker->update(board); // cooldown 3 -> 2
    attacker->update(board); // cooldown 2 -> 1
    REQUIRE(attacker->attackCount == 1);

    attacker->update(board); // cooldown 1 -> 0, attacks again
    REQUIRE(attacker->attackCount == 2);
}

TEST_CASE("Freeze slows cooldown recovery instead of stopping attacks outright", "[combat_entity][freeze]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    attacker->update(board); // attacks, cooldown -> 10
    REQUIRE(attacker->attackCount == 1);

    attacker->applyFreeze(100, 0.5f); // frozen: cooldown now drains at 0.5/tick, not 1.0/tick

    for (int i = 0; i < 19; ++i) attacker->update(board); // 19 * 0.5 = 9.5 drained, not yet 0
    REQUIRE(attacker->attackCount == 1);

    attacker->update(board); // one more 0.5 drains -> cooldown reaches 0, attacks
    REQUIRE(attacker->attackCount == 2);
}

TEST_CASE("Freeze ticks count down independently of cooldown", "[combat_entity][freeze]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    attacker->applyFreeze(5, 0.5f);

    for (int i = 0; i < 5; ++i) attacker->update(board);
    REQUIRE(attacker->freezeTicks == 0);
}

TEST_CASE("No enemies on the board: no attack, no crash", "[combat_entity][targeting]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100, 0, 5.0f, 10, 10);
    REQUIRE_NOTHROW(attacker->update(board));
    REQUIRE(attacker->attackCount == 0);
}

TEST_CASE("Stationary combatant never moves, even when a target exists out of range", "[combat_entity][movement]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 50.0f, 100, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 3.0f, 3.0f, 100, 0, 1.0f, 10, 10);
    attacker->update(board);

    REQUIRE(attacker->position.x == Catch::Approx(3.0f));
    REQUIRE(attacker->position.y == Catch::Approx(3.0f));
}

// ---------------- ground / air ----------------

TEST_CASE("A ground-only attacker (targetsAir false) ignores a flying enemy", "[combat_entity][flying]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    enemy->isFlying = true;
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 0);
}

TEST_CASE("An attacker with targetsAir hits a flying enemy", "[combat_entity][flying]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1);
    enemy->isFlying = true;
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->targetsAir = true;
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
}

TEST_CASE("An attacker with targetsAir still attacks grounded enemies", "[combat_entity][flying]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1); // isFlying stays false
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->targetsAir = true;
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
}

TEST_CASE("findTarget skips a closer flying enemy entirely rather than deprioritizing it", "[combat_entity][flying]") {
    Board board;
    auto flyingCloser = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100, 1, 'F');
    flyingCloser->isFlying = true;
    auto groundFarther = std::make_shared<DummyEntity>(2, 0.0f, 3.0f, 100, 1, 'G');
    spawn(board, flyingCloser);
    spawn(board, groundFarther);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
    REQUIRE(attacker->lastTargetId == groundFarther->id);
}

// ---------------- death effects ----------------

TEST_CASE("CombatEntity::onDeath fires its configured death effect with its own position and team", "[combat_entity][death]") {
    auto entity = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 100, 1, 5.0f, 10, 10);
    auto effect = std::make_shared<RecordingDeathEffect>();
    entity->deathEffect = effect;

    Board board;
    entity->onDeath(board);

    REQUIRE(effect->applied);
    REQUIRE(effect->lastPosition.x == Catch::Approx(3.0f));
    REQUIRE(effect->lastPosition.y == Catch::Approx(4.0f));
    REQUIRE(effect->lastTeam == 1);
}

TEST_CASE("CombatEntity::onDeath is a no-op when no death effect is set (default)", "[combat_entity][death]") {
    auto entity = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 100, 1, 5.0f, 10, 10);
    Board board;
    REQUIRE_NOTHROW(entity->onDeath(board));
}

// ---------------- ramping damage ----------------

namespace {
    std::shared_ptr<StationaryCombatant> makeRampingAttacker() {
        // attackCooldown = 1 so it attacks every single tick, keeping
        // "number of update() calls" aligned with ticksOnTarget.
        auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 1000, 1);
        attacker->rampMidTick = 20;
        attacker->rampFullTick = 40;
        attacker->rampStartFraction = 0.05f;
        attacker->rampMidFraction = 0.1875f;
        return attacker;
    }
}

TEST_CASE("Ramping damage deals rampStartFraction before rampMidTick", "[combat_entity][ramp]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1);
    spawn(board, enemy);
    auto attacker = makeRampingAttacker();

    attacker->update(board); // ticksOnTarget starts at 0 for a freshly-locked target
    REQUIRE(enemy->hp == 1000000 - 50); // 1000 * 0.05
}

TEST_CASE("Ramping damage deals rampMidFraction from rampMidTick to rampFullTick", "[combat_entity][ramp]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1);
    spawn(board, enemy);
    auto attacker = makeRampingAttacker();

    for (int i = 0; i < 20; ++i) attacker->update(board); // ticksOnTarget 0..19 spent
    int hpBefore = enemy->hp;
    attacker->update(board); // 21st call: ticksOnTarget == 20 -> stage 2
    REQUIRE(hpBefore - enemy->hp == 187); // 1000 * 0.1875, truncated
}

TEST_CASE("Ramping damage reaches full damage at rampFullTick", "[combat_entity][ramp]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1);
    spawn(board, enemy);
    auto attacker = makeRampingAttacker();

    for (int i = 0; i < 40; ++i) attacker->update(board); // ticksOnTarget 0..39 spent
    int hpBefore = enemy->hp;
    attacker->update(board); // 41st call: ticksOnTarget == 40 -> full damage
    REQUIRE(hpBefore - enemy->hp == 1000);
}

TEST_CASE("Ramping damage resets to stage 1 when the attacker switches targets", "[combat_entity][ramp]") {
    Board board;
    auto enemyA = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1, 'A');
    spawn(board, enemyA);
    auto attacker = makeRampingAttacker();

    for (int i = 0; i < 40; ++i) attacker->update(board); // fully ramped against enemyA

    // enemyA dies -- the attacker's target-lock (see CombatEntity::update)
    // only lets go once the locked target is no longer valid, so a switch
    // has to be earned this way now; a still-alive enemyA would keep the
    // attacker locked on even with a closer enemy nearby.
    enemyA->takeDamage(enemyA->hp);
    auto enemyB = std::make_shared<DummyEntity>(3, 0.0f, 0.5f, 1000000, 1, 'B');
    spawn(board, enemyB);

    int hpBefore = enemyB->hp;
    attacker->update(board); // target switches to enemyB: ramp resets
    REQUIRE(hpBefore - enemyB->hp == 50); // stage 1 damage against the new target
}

TEST_CASE("Ramping damage resets when the attacker is frozen, even mid-ramp", "[combat_entity][ramp]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1);
    spawn(board, enemy);
    auto attacker = makeRampingAttacker();

    for (int i = 0; i < 40; ++i) attacker->update(board); // fully ramped

    attacker->applyFreeze(1, 1.0f); // duration matters here, not slow strength
    int hpBefore = enemy->hp;
    attacker->update(board); // spent frozen -> ramp resets even though the target didn't change

    REQUIRE(hpBefore - enemy->hp == 50); // back to stage 1, not full damage
}

TEST_CASE("A card without ramping configured (rampFullTick 0, the default) always deals flat damage", "[combat_entity][ramp]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 1000, 1);
    for (int i = 0; i < 50; ++i) attacker->update(board);

    REQUIRE(enemy->hp == 1000000 - 50 * 1000); // every one of the 50 hits dealt the full 1000
}

// ---------------- split-target attacks ----------------

TEST_CASE("Split-target attacker deals full damage when only one enemy is in range", "[combat_entity][split]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1);
    spawn(board, enemy);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->maxSplitTargets = 2;
    attacker->update(board);

    REQUIRE(enemy->hp == 0); // full 1000, not halved
}

TEST_CASE("Split-target attacker halves damage across the 2 closest enemies when both are in range", "[combat_entity][split]") {
    Board board;
    auto near = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'N');
    auto far = std::make_shared<DummyEntity>(2, 0.0f, 2.0f, 1000, 1, 'F');
    spawn(board, near);
    spawn(board, far);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->maxSplitTargets = 2;
    attacker->update(board);

    REQUIRE(near->hp == 500);
    REQUIRE(far->hp == 500);
}

TEST_CASE("Split-target attacker only ever hits the closest maxSplitTargets, not every enemy in range", "[combat_entity][split]") {
    Board board;
    auto a = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'A');
    auto b = std::make_shared<DummyEntity>(2, 0.0f, 2.0f, 1000, 1, 'B');
    auto c = std::make_shared<DummyEntity>(3, 0.0f, 3.0f, 1000, 1, 'C'); // 3rd-closest: must stay untouched
    spawn(board, a);
    spawn(board, b);
    spawn(board, c);

    auto attacker = std::make_shared<StationaryCombatant>(4, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->maxSplitTargets = 2;
    attacker->update(board);

    REQUIRE(a->hp == 500);
    REQUIRE(b->hp == 500);
    REQUIRE(c->hp == 1000); // untouched: only the 2 closest are hit
}

TEST_CASE("A card without split targets configured (maxSplitTargets 1, the default) hits only its single closest enemy", "[combat_entity][split]") {
    Board board;
    auto near = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'N');
    auto far = std::make_shared<DummyEntity>(2, 0.0f, 2.0f, 1000, 1, 'F');
    spawn(board, near);
    spawn(board, far);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->update(board); // maxSplitTargets left at its default (1)

    REQUIRE(near->hp == 0);   // full damage, the normal single-target case
    REQUIRE(far->hp == 1000); // never targeted at all
}
