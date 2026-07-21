#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "MeleeTroop.h"

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

// ---------------- splash damage ----------------
// See CombatEntity::applySplashDamage. Distinct from split-target above:
// splash hits everyone within a fixed radius of the primary target's
// position (an area), not "the N closest enemies to the attacker" (a count).

TEST_CASE("Splash attacker damages a nearby second enemy in addition to its primary target", "[combat_entity][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'P');
    auto nearby = std::make_shared<DummyEntity>(2, 0.5f, 1.0f, 1000, 1, 'N'); // 0.5 from primary
    spawn(board, primary);
    spawn(board, nearby);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->splashRadius = 1.5f;
    attacker->update(board); // locks onto and hits the closer `primary`

    REQUIRE(primary->hp == 0);
    REQUIRE(nearby->hp == 0); // caught in the splash too, same full damage
}

TEST_CASE("Splash damage doesn't reach an enemy outside the splash radius", "[combat_entity][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'P');
    auto faraway = std::make_shared<DummyEntity>(2, 0.0f, 10.0f, 1000, 1, 'F');
    spawn(board, primary);
    spawn(board, faraway);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->splashRadius = 1.5f;
    attacker->update(board);

    REQUIRE(primary->hp == 0);
    REQUIRE(faraway->hp == 1000); // outside the 1.5 radius: untouched
}

TEST_CASE("Splash damage never lands on the attacker's own team", "[combat_entity][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'P');
    auto ally = std::make_shared<DummyEntity>(2, 0.3f, 1.0f, 1000, 0, 'X'); // same team as the attacker, well within radius
    spawn(board, primary);
    spawn(board, ally);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->splashRadius = 1.5f;
    attacker->update(board);

    REQUIRE(primary->hp == 0);
    REQUIRE(ally->hp == 1000); // friendly fire is not a thing here
}

TEST_CASE("Splash damage doesn't double-hit the primary target", "[combat_entity][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'P');
    spawn(board, primary);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 10.0f, 300, 10);
    attacker->splashRadius = 1.5f; // primary is at distance 0 from itself -- must not be hit twice
    attacker->update(board);

    REQUIRE(primary->hp == 700); // 1000 - 300, not 1000 - 600
}

TEST_CASE("A card without splash configured (splashRadius 0, the default) only ever damages its single target", "[combat_entity][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 1000, 1, 'P');
    auto nearby = std::make_shared<DummyEntity>(2, 0.5f, 1.0f, 1000, 1, 'N');
    spawn(board, primary);
    spawn(board, nearby);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 10.0f, 1000, 10);
    attacker->update(board); // splashRadius left at its default (0.0f)

    REQUIRE(primary->hp == 0);
    REQUIRE(nearby->hp == 1000); // no splash: never touched
}

// ---------------- shields ----------------
// See CombatEntity::takeDamage. Applies to any damage source (direct hit,
// splash, spell) since it's implemented in takeDamage() itself.

TEST_CASE("Shield absorbs damage before real hp, dollar for dollar", "[combat_entity][shield]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 500, 0, 1.0f, 0, 10);
    defender->shieldHp = 100;

    defender->takeDamage(60);
    REQUIRE(defender->shieldHp == 40);
    REQUIRE(defender->hp == 500); // untouched while shield still has capacity
}

TEST_CASE("Damage exceeding the remaining shield spills over onto real hp, not double-counted", "[combat_entity][shield]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 500, 0, 1.0f, 0, 10);
    defender->shieldHp = 100;

    defender->takeDamage(150); // 100 breaks the shield, 50 carries over

    REQUIRE(defender->shieldHp == 0);
    REQUIRE(defender->hp == 450); // 500 - 50, not 500 - 150
}

TEST_CASE("Once a shield is depleted, further damage goes straight to real hp", "[combat_entity][shield]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 500, 0, 1.0f, 0, 10);
    defender->shieldHp = 100;

    defender->takeDamage(100); // breaks the shield exactly
    REQUIRE(defender->shieldHp == 0);
    REQUIRE(defender->hp == 500);

    defender->takeDamage(80);
    REQUIRE(defender->hp == 420); // 500 - 80, no shield left to absorb any of it
}

TEST_CASE("A card without a shield (shieldHp 0, the default) takes damage directly", "[combat_entity][shield]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 500, 0, 1.0f, 0, 10);
    defender->takeDamage(80);
    REQUIRE(defender->hp == 420);
}

// ---------------- charge/dash bonus damage ----------------
// Needs an attacker that actually moves (StationaryCombatant never does),
// so these use the real MeleeTroop with setIgnoresRiver(true) to keep
// movement a straight line, sidestepping bridge/waypoint routing.

TEST_CASE("An attacker deals bonus charge damage after moving far enough without attacking", "[combat_entity][charge]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 4.0f, 1.0f, 100, 10, 'p');
    attacker->setIgnoresRiver(true);
    attacker->chargeThreshold = 3.0f;
    attacker->chargeMultiplier = 2.0f;

    attacker->update(board); // moves 4.0 toward the target (still out of the 1.8 effective range): chargeProgress = 4.0
    REQUIRE(target->hp == 10000); // not in range yet, no hit landed

    attacker->update(board); // now within range (dist 1.0 <= 1.8): charged hit lands
    REQUIRE(target->hp == 10000 - 200); // 100 base * 2.0 charge multiplier
}

TEST_CASE("An attacker deals normal damage when it hasn't moved far enough to charge", "[combat_entity][charge]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1); // already within range at spawn
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 4.0f, 1.0f, 100, 10, 'p');
    attacker->setIgnoresRiver(true);
    attacker->chargeThreshold = 3.0f;
    attacker->chargeMultiplier = 2.0f;

    attacker->update(board); // in range immediately: attacks without ever having moved

    REQUIRE(target->hp == 10000 - 100); // base damage only, not charged
}

TEST_CASE("Charge resets after landing a hit -- the next attack isn't charged again for free", "[combat_entity][charge]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 4.0f, 1.0f, 100, 1, 'p'); // cooldown 1: attacks every tick once in range
    attacker->setIgnoresRiver(true);
    attacker->chargeThreshold = 3.0f;
    attacker->chargeMultiplier = 2.0f;

    attacker->update(board); // moves 4.0: charged
    attacker->update(board); // in range: charged hit (200)
    REQUIRE(target->hp == 10000 - 200);

    attacker->update(board); // still in range, cooldown just expired: attacks again, but charge was reset to 0 by the previous hit
    REQUIRE(target->hp == 10000 - 200 - 100); // second hit is base damage, not charged
}

TEST_CASE("A card without charge configured (chargeThreshold 0, the default) never deals bonus damage", "[combat_entity][charge]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 4.0f, 1.0f, 100, 10, 'p');
    attacker->setIgnoresRiver(true); // chargeThreshold left at its default (0.0f)

    attacker->update(board); // moves 4.0
    attacker->update(board); // in range: attacks

    REQUIRE(target->hp == 10000 - 100); // never charged, regardless of distance moved
}

// ---------------- enrage ----------------

TEST_CASE("Enrage heals a small amount with every landed hit", "[combat_entity][enrage]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 10.0f, 50, 10);
    attacker->hp = 500; // already damaged, well under enrageMaxHp
    attacker->enrageMaxHp = 1000;
    attacker->enrageHealPerHit = 20;

    attacker->update(board); // lands a hit
    REQUIRE(attacker->hp == 520); // 500 + 20
}

TEST_CASE("Enrage's self-heal is capped at enrageMaxHp, never overhealing", "[combat_entity][enrage]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 10.0f, 50, 10);
    attacker->hp = 995; // close to the cap
    attacker->enrageMaxHp = 1000;
    attacker->enrageHealPerHit = 20;

    attacker->update(board); // fresh attacker: cooldown starts at 0, attacks immediately

    REQUIRE(attacker->hp == 1000); // capped at enrageMaxHp, not 1015
}

TEST_CASE("Enrage shortens attack cooldown the more damaged this attacker already is", "[combat_entity][enrage]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);

    auto halfHp = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 10.0f, 50, 10);
    halfHp->hp = 500; // 50% hp
    halfHp->enrageMaxHp = 1000; // no heal configured -- isolates the cooldown effect

    auto fullHp = std::make_shared<StationaryCombatant>(3, 5.0f, 0.0f, 1000, 0, 10.0f, 50, 10);
    fullHp->enrageMaxHp = 1000; // 100% hp: enrage formula reduces to no-op

    for (int i = 0; i < 20; ++i) {
        halfHp->update(board);
        fullHp->update(board);
    }

    REQUIRE(halfHp->attackCount > fullHp->attackCount); // more damaged -> attacks more often in the same window
}

TEST_CASE("A card without enrage configured (enrageMaxHp 0, the default) neither heals nor speeds up", "[combat_entity][enrage]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 10.0f, 50, 10);
    attacker->hp = 500; // enrageMaxHp left at its default (0): mechanic stays off regardless

    attacker->update(board);
    REQUIRE(attacker->hp == 500); // no self-heal
}

// ---------------- parry ----------------

TEST_CASE("Parry fully negates the first incoming hit", "[combat_entity][parry]") {
    Board board;
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 0, 10);
    defender->parryIntervalTicks = 35;

    defender->takeDamage(200); // ready at spawn: fully negated

    REQUIRE(defender->hp == 1000);
}

TEST_CASE("Once consumed, parry doesn't negate the next hit until its interval elapses", "[combat_entity][parry]") {
    Board board;
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 0, 10);
    defender->parryIntervalTicks = 35;

    defender->takeDamage(200); // consumes the parry: negated
    REQUIRE(defender->hp == 1000);

    defender->takeDamage(200); // parry not ready again yet: applies normally
    REQUIRE(defender->hp == 800);
}

TEST_CASE("Parry becomes ready again once its interval elapses", "[combat_entity][parry]") {
    Board board;
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 0, 10);
    defender->parryIntervalTicks = 35;

    defender->takeDamage(200); // consumes the parry
    REQUIRE(defender->hp == 1000);

    for (int i = 0; i < 35; ++i) defender->update(board); // interval elapses, empty board (no attack fires)

    defender->takeDamage(200); // ready again: negated
    REQUIRE(defender->hp == 1000);
}

TEST_CASE("A card without parry configured (parryIntervalTicks 0, the default) always takes normal damage", "[combat_entity][parry]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 0, 10);
    defender->takeDamage(200);
    REQUIRE(defender->hp == 800);
}

// ---------------- hook ----------------

TEST_CASE("Hook instantly pulls an out-of-range target to just inside melee range, dealing no damage that tick", "[combat_entity][hook]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    // attackRange 1.0 -> effective range 1.8 (implicit radii on both sides)
    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 1);
    attacker->hookRange = 6.5f;

    attacker->update(board); // dist 5.0: beyond effective range, within hookRange -- hooks instead of chasing

    REQUIRE(target->position.y == Catch::Approx(1.7f)); // pulled 3.3 (5.0 - 1.8 + 0.1) toward the attacker
    REQUIRE(target->hp == 10000); // no damage on the hook tick itself
    REQUIRE(attacker->attackCount == 0); // hook isn't counted as a landed attack
}

TEST_CASE("After being hooked into range, the next attack lands normally", "[combat_entity][hook]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 1);
    attacker->hookRange = 6.5f;

    attacker->update(board); // hooks: pulls target to (0, 1.7), no damage
    attacker->update(board); // now within effective range 1.8: normal attack lands

    REQUIRE(target->hp == 10000 - 100);
    REQUIRE(attacker->attackCount == 1);
}

TEST_CASE("A target beyond hookRange is neither hooked nor chased by a stationary attacker", "[combat_entity][hook]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 20.0f, 10000, 1); // well beyond hookRange
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 1);
    attacker->hookRange = 6.5f;

    attacker->update(board);

    REQUIRE(target->position.y == Catch::Approx(20.0f)); // untouched
    REQUIRE(target->hp == 10000);
}

TEST_CASE("A card without hook configured (hookRange 0, the default) never pulls anyone", "[combat_entity][hook]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 1);
    attacker->update(board); // hookRange left at its default (0.0f)

    REQUIRE(target->position.y == Catch::Approx(5.0f)); // untouched
    REQUIRE(target->hp == 10000);
}

// ---------------- invisibility ----------------

TEST_CASE("An invisible unit is untargetable from the moment it spawns", "[combat_entity][invisibility]") {
    auto ghost = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 100, 10);
    ghost->startsInvisible = true;
    ghost->revealTicksAfterAttack = 5;

    REQUIRE_FALSE(ghost->isTargetable());
}

TEST_CASE("An invisible unit becomes targetable right after landing a hit", "[combat_entity][invisibility]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto ghost = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 1.0f, 100, 10);
    ghost->startsInvisible = true;
    ghost->revealTicksAfterAttack = 5;

    ghost->update(board); // in range, attacks: becomes visible

    REQUIRE(ghost->isTargetable());
}

TEST_CASE("An invisible unit goes invisible again once its reveal window elapses", "[combat_entity][invisibility]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto ghost = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 1000, 0, 1.0f, 100, 10);
    ghost->startsInvisible = true;
    ghost->revealTicksAfterAttack = 5;

    ghost->update(board); // attacks: visible for 5 ticks
    REQUIRE(ghost->isTargetable());

    for (int i = 0; i < 5; ++i) ghost->update(board); // reveal window elapses

    REQUIRE_FALSE(ghost->isTargetable());
}

TEST_CASE("findTarget skips an invisible unit entirely, even if it's the closest candidate", "[combat_entity][invisibility]") {
    Board board;
    auto ghost = std::make_shared<StationaryCombatant>(1, 0.0f, 1.0f, 1000, 1, 1.0f, 0, 10);
    ghost->startsInvisible = true;
    spawn(board, ghost);

    auto visible = std::make_shared<DummyEntity>(2, 0.0f, 10.0f, 1000, 1); // much farther away
    spawn(board, visible);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 20.0f, 50, 10);
    attacker->update(board);

    REQUIRE(attacker->lastTargetId == visible->id); // skips the invisible, closer ghost entirely
}

TEST_CASE("A card without invisibility configured (startsInvisible false, the default) is always targetable", "[combat_entity][invisibility]") {
    auto unit = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 100, 10);
    REQUIRE(unit->isTargetable());
}

// ---------------- periodic effects ----------------

TEST_CASE("Periodic effect doesn't fire before a full interval has elapsed", "[combat_entity][periodic]") {
    Board board;
    auto unit = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 1000, 0, 1.0f, 0, 10);
    auto effect = std::make_shared<RecordingPeriodicEffect>();
    unit->periodicEffect = effect;
    unit->periodicIntervalTicks = 5;
    unit->periodicTicksUntilNext = 5; // matches CardFactories::applyCardMetadata's initialization

    for (int i = 0; i < 4; ++i) unit->update(board);

    REQUIRE(effect->applyCount == 0);
}

TEST_CASE("Periodic effect fires at its own position/team once the interval elapses, then repeats", "[combat_entity][periodic]") {
    Board board;
    auto unit = std::make_shared<StationaryCombatant>(1, 3.0f, 4.0f, 1000, 0, 1.0f, 0, 10);
    auto effect = std::make_shared<RecordingPeriodicEffect>();
    unit->periodicEffect = effect;
    unit->periodicIntervalTicks = 5;
    unit->periodicTicksUntilNext = 5;

    for (int i = 0; i < 5; ++i) unit->update(board); // interval elapses on the 5th tick

    REQUIRE(effect->applyCount == 1);
    REQUIRE(effect->lastPosition.x == Catch::Approx(3.0f));
    REQUIRE(effect->lastPosition.y == Catch::Approx(4.0f));
    REQUIRE(effect->lastTeam == 0);

    for (int i = 0; i < 5; ++i) unit->update(board); // a second full interval

    REQUIRE(effect->applyCount == 2); // fires again, not just once ever
}

TEST_CASE("A card without a periodic effect configured (periodicIntervalTicks 0, the default) never fires one", "[combat_entity][periodic]") {
    Board board;
    auto unit = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 1.0f, 0, 10);
    for (int i = 0; i < 50; ++i) unit->update(board);
    REQUIRE_NOTHROW(unit->update(board)); // no periodicEffect set at all: must not crash trying to fire one
}

// ---------------- buffs and curses ----------------

TEST_CASE("A damage buff multiplies getCurrentDamage while active, then expires", "[combat_entity][buff]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 1);
    attacker->applyBuff(1.5f, 3);

    attacker->update(board); // tick 1: buffed (3 ticks remaining before this decrements to 2)
    REQUIRE(target->hp == 100000 - 150); // 100 * 1.5

    for (int i = 0; i < 2; ++i) attacker->update(board); // buff expires (2 more decrements: 2 -> 1 -> 0)
    int hpBefore = target->hp;
    attacker->update(board);
    REQUIRE(hpBefore - target->hp == 100); // back to base damage, unbuffed
}

TEST_CASE("A curse multiplies incoming damage in takeDamage while active, then expires", "[combat_entity][curse]") {
    auto defender = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100000, 0, 1.0f, 0, 10);
    defender->applyCurse(1.5f, 2);

    defender->takeDamage(100);
    REQUIRE(defender->hp == 100000 - 150); // 100 * 1.5, curse ticks not decremented by takeDamage itself

    Board board;
    defender->update(board); // curseTicksRemaining 2 -> 1
    defender->update(board); // curseTicksRemaining 1 -> 0: expired

    int hpBefore = defender->hp;
    defender->takeDamage(100);
    REQUIRE(hpBefore - defender->hp == 100); // back to normal, uncursed
}

TEST_CASE("applyAreaBuff buffs only same-team CombatEntity candidates within radius, up to maxTargets", "[combat_entity][buff]") {
    Board board;
    auto near = std::make_shared<StationaryCombatant>(1, 1.0f, 0.0f, 100, 0, 1.0f, 0, 10);   // ally, close
    auto mid = std::make_shared<StationaryCombatant>(2, 2.0f, 0.0f, 100, 0, 1.0f, 0, 10);    // ally, farther
    auto enemy = std::make_shared<StationaryCombatant>(3, 1.5f, 0.0f, 100, 1, 1.0f, 0, 10);  // enemy, in radius
    spawn(board, near);
    spawn(board, mid);
    spawn(board, enemy);

    applyAreaBuff(board, Vector2D{ 0.0f, 0.0f }, 5.0f, -1, 0, 2.0f, 30, 1); // only room for 1 target

    REQUIRE(near->buffTicksRemaining == 30); // closest ally: buffed
    REQUIRE(mid->buffTicksRemaining == 0);   // farther ally: capped out by maxTargets
    REQUIRE(enemy->buffTicksRemaining == 0); // wrong team: never a candidate
}

TEST_CASE("applyAreaHeal heals only same-team entities within radius", "[combat_entity][heal]") {
    Board board;
    auto ally = std::make_shared<DummyEntity>(1, 1.0f, 0.0f, 500, 0);
    auto enemy = std::make_shared<DummyEntity>(2, 1.0f, 0.0f, 500, 1);
    auto farAlly = std::make_shared<DummyEntity>(3, 20.0f, 0.0f, 500, 0);
    spawn(board, ally);
    spawn(board, enemy);
    spawn(board, farAlly);

    applyAreaHeal(board, Vector2D{ 0.0f, 0.0f }, 5.0f, -1, 0, 50);

    REQUIRE(ally->hp == 550);
    REQUIRE(enemy->hp == 500);    // wrong team: untouched
    REQUIRE(farAlly->hp == 500);  // outside radius: untouched
}

TEST_CASE("Ally buff aura fires every Nth landed attack, not every attack", "[combat_entity][aura]") {
    Board board;
    auto enemyTarget = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, enemyTarget);
    auto ally = std::make_shared<StationaryCombatant>(2, 3.0f, 0.0f, 100, 0, 1.0f, 0, 10);
    spawn(board, ally);

    auto buffer = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 1.0f, 10, 1);
    buffer->auraRadius = 5.0f;
    buffer->auraEveryNAttacks = 3;
    buffer->auraBuffMultiplier = 2.0f;
    buffer->auraBuffDurationTicks = 20;
    buffer->auraMaxTargets = 5;

    buffer->update(board); // attack 1
    buffer->update(board); // attack 2
    REQUIRE(ally->buffTicksRemaining == 0); // not yet, only 2 landed attacks so far

    buffer->update(board); // attack 3: aura fires
    REQUIRE(ally->buffTicksRemaining == 20);
}

TEST_CASE("Heal aura fires with every landed attack", "[combat_entity][aura]") {
    Board board;
    auto enemyTarget = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 100000, 1);
    spawn(board, enemyTarget);
    auto ally = std::make_shared<DummyEntity>(2, 3.0f, 0.0f, 500, 0);
    spawn(board, ally);

    auto healer = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 1.0f, 10, 1);
    healer->auraRadius = 5.0f;
    healer->healAllyAmount = 30;

    healer->update(board); // lands a hit: heals nearby allies too

    REQUIRE(ally->hp == 530);
}

// ---------------- kamikaze (dieAfterFirstHit) ----------------

TEST_CASE("An attacker with dieAfterFirstHit dies the instant its attack lands", "[combat_entity][kamikaze]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 10);
    attacker->dieAfterFirstHit = true;

    attacker->update(board);

    REQUIRE(target->hp == 10000 - 100); // the hit still lands
    REQUIRE_FALSE(attacker->isAlive());  // but the attacker dies right after
}

TEST_CASE("A card without dieAfterFirstHit configured (the default) survives to attack again", "[combat_entity][kamikaze]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 100, 10);
    attacker->update(board);

    REQUIRE(attacker->isAlive());
}

// ---------------- dash invulnerability (Bandit) ----------------

TEST_CASE("chargeGrantsInvulnerability blocks damage once past half the charge threshold", "[combat_entity][invulnerability]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 500, 0, 1.0f, 1.0f, 10, 10, 'u');
    attacker->chargeThreshold = 4.0f;
    attacker->chargeMultiplier = 2.0f;
    attacker->chargeGrantsInvulnerability = true;

    attacker->update(board); // moves 1.0 toward target: chargeProgress 1.0, below half (2.0)
    attacker->takeDamage(50);
    REQUIRE(attacker->hp == 450); // not yet invulnerable

    attacker->update(board); // moves another 1.0: chargeProgress 2.0, at half -- invulnerable now
    attacker->takeDamage(50);
    REQUIRE(attacker->hp == 450); // fully blocked
}

TEST_CASE("A charging attacker without chargeGrantsInvulnerability takes damage normally", "[combat_entity][invulnerability]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 10.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 500, 0, 1.0f, 1.0f, 10, 10, 'u');
    attacker->chargeThreshold = 4.0f;
    attacker->chargeMultiplier = 2.0f;
    // chargeGrantsInvulnerability left false (the default)

    attacker->update(board);
    attacker->update(board);
    attacker->takeDamage(50);
    REQUIRE(attacker->hp == 450); // still vulnerable, even fully charged
}

// ---------------- stun resets cooldown (Sparky) ----------------

TEST_CASE("resetCooldownOnFreeze fully restarts the attack cooldown instead of just slowing it", "[combat_entity][freeze]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->resetCooldownOnFreeze = true;

    attacker->update(board); // lands the attack, currentCooldown -> 10
    attacker->update(board); // decrements to 9 (unfrozen tick)

    attacker->applyFreeze(5, 0.5f);
    REQUIRE(attacker->attackCount == 1); // hasn't fired again yet

    // Cooldown was reset to the full 10 (not left at the 9 it had already
    // drained to) -- with 5 frozen ticks draining at freezeSlow (0.5/tick:
    // 10 -> 7.5) followed by normal ticks (-1/tick) until it clears, the
    // next attack needs 13 total update() calls after the freeze lands.
    // Without the reset it would only need 12 (draining from 9 instead of
    // 10) -- asserting exactly 13, not merely "eventually," is what
    // actually distinguishes reset-on-freeze from plain freeze.
    for (int i = 0; i < 12; ++i) attacker->update(board);
    REQUIRE(attacker->attackCount == 1); // still not ready -- would already be ready without the reset
    attacker->update(board);
    REQUIRE(attacker->attackCount == 2); // ready on the 13th
}

// ---------------- recoil on attack (Firecracker) ----------------

TEST_CASE("recoilDistance pushes the attacker away from its target after a landed hit", "[combat_entity][recoil]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1);
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->recoilDistance = 1.0f;

    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
    REQUIRE(attacker->position.y == Catch::Approx(-1.0f)); // kicked 1.0 away from the target at (0,1)
}

// ---------------- minimum attack range (Mortar) ----------------

TEST_CASE("minAttackRange rejects a target sitting inside the blind spot", "[combat_entity][min_range]") {
    Board board;
    auto tooClose = std::make_shared<DummyEntity>(1, 0.0f, 1.0f, 10000, 1); // dist 1.0
    spawn(board, tooClose);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->minAttackRange = 3.0f;

    attacker->update(board);

    REQUIRE(attacker->attackCount == 0); // inside the blind spot: not a valid target at all
}

TEST_CASE("minAttackRange still allows a target beyond the blind spot but within range", "[combat_entity][min_range]") {
    Board board;
    auto farEnough = std::make_shared<DummyEntity>(1, 0.0f, 4.0f, 10000, 1); // dist 4.0
    spawn(board, farEnough);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 5.0f, 50, 10);
    attacker->minAttackRange = 3.0f;

    attacker->update(board);

    REQUIRE(attacker->attackCount == 1);
}

// ---------------- HP-threshold transform (Cannon Cart) ----------------

TEST_CASE("A transform-capable attacker grounds itself once at or below the HP threshold", "[combat_entity][transform]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    attacker->transformAtHpFraction = 0.5f;
    attacker->transformCheckMaxHp = 1000;
    attacker->transformLifetimeTicks = 20;
    attacker->transformBecomesStationary = true;

    attacker->takeDamage(600); // down to 400/1000 = 40%, at/below the 50% threshold
    REQUIRE_FALSE(attacker->hasTransformed);

    attacker->update(board); // no enemies on the board -- just exercises the transform check
    REQUIRE(attacker->hasTransformed);
    REQUIRE(attacker->freezeTicks > 0); // grounded via the freeze(0.0f) trick
    REQUIRE(attacker->freezeSlow == Catch::Approx(0.0f));
}

TEST_CASE("A transformed attacker self-destructs once its post-transform lifetime runs out", "[combat_entity][transform]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    attacker->transformAtHpFraction = 0.5f;
    attacker->transformCheckMaxHp = 1000;
    attacker->transformLifetimeTicks = 3;
    attacker->transformBecomesStationary = true;

    attacker->takeDamage(600);
    // The triggering update() both starts the transform AND ticks the
    // countdown once in the same call (transformTicksRemaining 3 -> 2),
    // so exactly `transformLifetimeTicks` total update() calls (not one
    // more) elapse before it self-destructs.
    attacker->update(board); // remaining 3 -> 2
    REQUIRE(attacker->isAlive());
    attacker->update(board); // remaining 2 -> 1
    REQUIRE(attacker->isAlive());
    attacker->update(board); // remaining 1 -> 0: self-destructs
    REQUIRE_FALSE(attacker->isAlive());
}

// ---------------- archetype-swap transform (Goblin Demolisher) ----------------

TEST_CASE("transformKillsSelf kills the entity and fires transformDeathEffect, separately from the normal deathEffect", "[combat_entity][transform]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    attacker->transformAtHpFraction = 0.5f;
    attacker->transformCheckMaxHp = 1000;
    attacker->transformKillsSelf = true;
    auto transformEffect = std::make_shared<RecordingDeathEffect>();
    auto normalDeathEffect = std::make_shared<RecordingDeathEffect>();
    attacker->transformDeathEffect = transformEffect;
    attacker->deathEffect = normalDeathEffect;
    spawn(board, attacker); // needs to be on the board for cleanDeadEntities to find it below

    attacker->takeDamage(600); // 400/1000 = 40%, at/below the 50% threshold
    attacker->update(board);

    REQUIRE_FALSE(attacker->isAlive());
    REQUIRE(transformEffect->applied); // fires immediately, from inside update()
    REQUIRE_FALSE(normalDeathEffect->applied); // NOT fired yet -- that's Board::cleanDeadEntities' job

    board.cleanDeadEntities();
    REQUIRE(normalDeathEffect->applied); // the ordinary death pipeline still runs too, independently
}

TEST_CASE("A lethal hit that skips straight past the transform threshold does not fire transformDeathEffect", "[combat_entity][transform]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    attacker->transformAtHpFraction = 0.5f;
    attacker->transformCheckMaxHp = 1000;
    attacker->transformKillsSelf = true;
    auto transformEffect = std::make_shared<RecordingDeathEffect>();
    attacker->transformDeathEffect = transformEffect;

    // One hit straight from 100% to 0% -- this attacker is never observed
    // alive at <=50%, so its own update() (where the transform check
    // lives) never gets a chance to run again; GameManager::step() only
    // calls update() on entities that are still isAlive().
    attacker->takeDamage(1000);
    REQUIRE_FALSE(attacker->isAlive());
    REQUIRE_FALSE(transformEffect->applied); // no bonus kamikaze spawn from an ordinary kill
}

// ---------------- periodic jump (Mega Knight) ----------------

TEST_CASE("Jump instantly closes distance and lands a boosted splash hit instead of walking in", "[combat_entity][jump]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 4.0f, 10000, 1); // dist 4.0, beyond attack range
    auto bystander = std::make_shared<DummyEntity>(2, 1.0f, 4.0f, 10000, 1); // near the target
    spawn(board, target);
    spawn(board, bystander);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 1.2f, 268, 17);
    attacker->jumpMinRange = 3.5f;
    attacker->jumpMaxRange = 5.0f;
    attacker->jumpDamageMultiplier = 2.0f;
    attacker->jumpSplashRadius = 2.2f;

    attacker->update(board);

    REQUIRE(target->hp == 10000 - 536);    // 268 * 2.0
    REQUIRE(bystander->hp == 10000 - 536); // caught by the landing splash too
    REQUIRE(attacker->position.y > 0.0f);  // jumped toward the target instead of staying put
    REQUIRE(attacker->attackCount == 0);   // went through the jump branch, not performAttack()
}

TEST_CASE("A target outside the jump window is walked toward normally, not jumped to", "[combat_entity][jump]") {
    Board board;
    auto tooFar = std::make_shared<DummyEntity>(1, 0.0f, 8.0f, 10000, 1); // dist 8.0, past jumpMaxRange
    spawn(board, tooFar);

    auto attacker = std::make_shared<MeleeTroop>(2, 0.0f, 0.0f, 100, 0, 1.0f, 1.2f, 268, 17, 'X');
    attacker->jumpMinRange = 3.5f;
    attacker->jumpMaxRange = 5.0f;
    attacker->jumpDamageMultiplier = 2.0f;
    attacker->jumpSplashRadius = 2.2f;

    attacker->update(board);

    REQUIRE(tooFar->hp == 10000);             // no jump damage landed
    REQUIRE(attacker->position.y == Catch::Approx(1.0f)); // just walked its normal `speed` (1.0) instead
}

// ---------------- piercing-line splash (Bowler, Magic Archer) ----------------

TEST_CASE("applyLineSplashDamage hits entities within the line's width along its length", "[combat_entity][line_splash]") {
    Board board;
    auto onLine = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 1000, 1);        // directly on the firing line
    auto offToSide = std::make_shared<DummyEntity>(2, 3.0f, 5.0f, 1000, 1);     // same distance along, far to the side
    auto behindShooter = std::make_shared<DummyEntity>(3, 0.0f, -2.0f, 1000, 1); // wrong direction from the origin
    auto pastEnd = std::make_shared<DummyEntity>(4, 0.0f, 20.0f, 1000, 1);      // beyond the line's range
    spawn(board, onLine);
    spawn(board, offToSide);
    spawn(board, behindShooter);
    spawn(board, pastEnd);

    applyLineSplashDamage(board, Vector2D{ 0.0f, 0.0f }, Vector2D{ 0.0f, 10.0f }, 11.0f, 1.0f, -1, 99, 0, 5, 100);

    REQUIRE(onLine->hp == 900);
    REQUIRE(offToSide->hp == 1000);     // too far off to the side
    REQUIRE(behindShooter->hp == 1000); // clamped projection excludes the wrong direction
    REQUIRE(pastEnd->hp == 1000);       // beyond the line's total range
}

TEST_CASE("applyLineSplashDamage excludes the primary target (already damaged directly) via excludeId", "[combat_entity][line_splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 0.0f, 5.0f, 1000, 1);
    spawn(board, primary);

    applyLineSplashDamage(board, Vector2D{ 0.0f, 0.0f }, Vector2D{ 0.0f, 10.0f }, 11.0f, 1.0f, 1, 99, 0, 5, 100);

    REQUIRE(primary->hp == 1000); // excluded by id, not hit a second time
}

// ---------------- range-based damage falloff (Hunter) ----------------

TEST_CASE("rangeFalloff deals full damage at point-blank range", "[combat_entity][range_falloff]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 0.5f, 1000, 1); // dist 0.5, well inside range 4.0

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 4.0f, 84, 22);
    attacker->rangeFalloff = true;
    attacker->rangeFalloffMinFraction = 0.5f;
    spawn(board, target);

    attacker->update(board);

    REQUIRE(target->hp > 1000 - 84); // less than full 84 would land at dist 0, but very close to it at 0.5
    REQUIRE(target->hp <= 1000 - 84 * 0.5); // never weaker than the min fraction
}

TEST_CASE("rangeFalloff deals the minimum fraction of damage right at max range", "[combat_entity][range_falloff]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 4.0f, 1000, 1); // dist exactly at attackRange 4.0
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 4.0f, 84, 22);
    attacker->rangeFalloff = true;
    attacker->rangeFalloffMinFraction = 0.5f;

    attacker->update(board);

    REQUIRE(target->hp == 1000 - 42); // 84 * 0.5, the minimum fraction
}

TEST_CASE("A card without rangeFalloff configured (the default) always deals full damage", "[combat_entity][range_falloff]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 0.0f, 4.0f, 1000, 1); // dist exactly at attackRange
    spawn(board, target);

    auto attacker = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 4.0f, 84, 22);
    attacker->update(board);

    REQUIRE(target->hp == 1000 - 84); // no falloff: full damage regardless of distance
}

TEST_CASE("A card without transformAtHpFraction configured (the default) never transforms", "[combat_entity][transform]") {
    Board board;
    auto attacker = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 50, 10);
    attacker->takeDamage(999);
    attacker->update(board);
    REQUIRE_FALSE(attacker->hasTransformed);
    REQUIRE(attacker->isAlive());
}

// ---------------- split-target full damage (Electro Dragon) ----------------

TEST_CASE("splitTargetsFullDamage gives every split target the full damage instead of dividing it", "[combat_entity][split]") {
    Board board;
    auto enemyA = std::make_shared<DummyEntity>(1, 1.0f, 0.0f, 1000, 1);
    auto enemyB = std::make_shared<DummyEntity>(2, -1.0f, 0.0f, 1000, 1);
    spawn(board, enemyA);
    spawn(board, enemyB);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 5.0f, 100, 10);
    attacker->maxSplitTargets = 2;
    attacker->splitTargetsFullDamage = true;

    attacker->update(board);

    REQUIRE(enemyA->hp == 900); // full 100, not divided by the 2 targets hit
    REQUIRE(enemyB->hp == 900);
}

TEST_CASE("Without splitTargetsFullDamage, split-target damage is divided as before", "[combat_entity][split]") {
    Board board;
    auto enemyA = std::make_shared<DummyEntity>(1, 1.0f, 0.0f, 1000, 1);
    auto enemyB = std::make_shared<DummyEntity>(2, -1.0f, 0.0f, 1000, 1);
    spawn(board, enemyA);
    spawn(board, enemyB);

    auto attacker = std::make_shared<StationaryCombatant>(3, 0.0f, 0.0f, 100, 0, 5.0f, 100, 10);
    attacker->maxSplitTargets = 2;

    attacker->update(board);

    REQUIRE(enemyA->hp == 950); // 100 / 2 targets
    REQUIRE(enemyB->hp == 950);
}

// ---------------- Champion activated ability ----------------

TEST_CASE("activateAbility fires the configured effect and starts the cooldown", "[combat_entity][champion]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100, 0, 1.0f, 10, 10);
    auto effect = std::make_shared<RecordingAbilityEffect>();
    champion->abilityEffect = effect;
    champion->abilityCooldownTicks = 50;

    bool fired = champion->activateAbility(board);

    REQUIRE(fired);
    REQUIRE(effect->applyCount == 1);
    REQUIRE(champion->abilityCooldownRemaining == 50);
}

TEST_CASE("activateAbility is a no-op while on cooldown", "[combat_entity][champion]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100, 0, 1.0f, 10, 10);
    auto effect = std::make_shared<RecordingAbilityEffect>();
    champion->abilityEffect = effect;
    champion->abilityCooldownTicks = 50;

    champion->activateAbility(board);
    bool firedAgain = champion->activateAbility(board);

    REQUIRE_FALSE(firedAgain);
    REQUIRE(effect->applyCount == 1); // still just the one, real, activation
}

TEST_CASE("The ability becomes available again once its cooldown fully elapses", "[combat_entity][champion]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 100.0f, 100.0f, 100, 1); // far away: never actually attacked
    auto champion = std::make_shared<StationaryCombatant>(2, 0.0f, 0.0f, 100, 0, 1.0f, 10, 10);
    spawn(board, target);
    auto effect = std::make_shared<RecordingAbilityEffect>();
    champion->abilityEffect = effect;
    champion->abilityCooldownTicks = 3;

    champion->activateAbility(board);
    REQUIRE(effect->applyCount == 1);

    champion->update(board); // cooldown 3 -> 2
    champion->update(board); // 2 -> 1
    REQUIRE_FALSE(champion->activateAbility(board));
    REQUIRE(effect->applyCount == 1); // still not ready

    champion->update(board); // 1 -> 0: ready again
    REQUIRE(champion->activateAbility(board));
    REQUIRE(effect->applyCount == 2);
}

TEST_CASE("A card without an ability configured (the default) never fires anything", "[combat_entity][champion]") {
    Board board;
    auto champion = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 100, 0, 1.0f, 10, 10);

    bool fired = champion->activateAbility(board);

    REQUIRE_FALSE(fired);
    REQUIRE_FALSE(champion->isChampion);
    REQUIRE(champion->abilityElixirCost == Catch::Approx(0.0f));
    REQUIRE(champion->abilityCooldownTicks == 0);
}
