#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "AreaSpell.h"
#include "ArenaLayout.h"
#include "CardRegistry.h"
#include "FreezeOnHit.h"
#include "MeleeTroop.h"
#include "Building.h"

TEST_CASE("AreaSpell is never itself a valid combat target", "[area_spell]") {
    AreaSpell spell(1, 5.0f, 5.0f, 0, 3.0f, 100, 5);
    REQUIRE_FALSE(spell.isTargetable());
}

TEST_CASE("AreaSpell waits out its delay before detonating", "[area_spell][delay]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 500, 2);

    spell.update(board); // delayTicks 2 -> 1, no detonation
    REQUIRE(spell.isAlive());
    REQUIRE(enemy->hp == 1000);

    spell.update(board); // delayTicks 1 -> 0, no detonation
    REQUIRE(spell.isAlive());
    REQUIRE(enemy->hp == 1000);

    spell.update(board); // delayTicks == 0: detonates now
    REQUIRE_FALSE(spell.isAlive());
    REQUIRE(enemy->hp == 500);
}

TEST_CASE("AreaSpell with zero delay detonates on the very first update", "[area_spell][delay]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 500, 0);
    spell.update(board);

    REQUIRE_FALSE(spell.isAlive());
    REQUIRE(enemy->hp == 500);
}

TEST_CASE("AreaSpell self-destructs on detonation even if nothing was in range", "[area_spell]") {
    Board board; // empty
    AreaSpell spell(1, 10.0f, 10.0f, 0, 3.0f, 500, 0);
    spell.update(board);
    REQUIRE_FALSE(spell.isAlive());
}

TEST_CASE("AreaSpell damages only alive, targetable, opposing-team entities within its radius", "[area_spell][radius]") {
    Board board;

    auto enemyAtBoundary = std::make_shared<DummyEntity>(1, 10.0f, 14.0f, 1000, 1); // dist == 4.0
    auto enemyJustOutside = std::make_shared<DummyEntity>(2, 10.0f, 14.1f, 1000, 1); // dist == 4.1
    auto ally = std::make_shared<DummyEntity>(3, 10.0f, 11.0f, 1000, 0); // same team as caster
    auto nonTargetableEnemy = std::make_shared<DummyEntity>(4, 10.0f, 11.0f, 1000, 1);
    nonTargetableEnemy->targetable = false;
    auto deadEnemy = std::make_shared<DummyEntity>(5, 10.0f, 11.0f, 100, 1);
    deadEnemy->takeDamage(100); // already dead before the spell resolves

    spawn(board, enemyAtBoundary);
    spawn(board, enemyJustOutside);
    spawn(board, ally);
    spawn(board, nonTargetableEnemy);
    spawn(board, deadEnemy);

    AreaSpell spell(6, 10.0f, 10.0f, 0, 4.0f, 500, 0);
    spell.update(board);

    REQUIRE(enemyAtBoundary->hp == 500);   // exactly at the radius boundary: included
    REQUIRE(enemyJustOutside->hp == 1000); // just beyond: excluded
    REQUIRE(ally->hp == 1000);             // same team: excluded
    REQUIRE(nonTargetableEnemy->hp == 1000); // not targetable: excluded
    REQUIRE(deadEnemy->hp == 0);            // already dead: left untouched, not further reduced
}

TEST_CASE("AreaSpell applies its on-hit effect to CombatEntity targets within radius", "[area_spell][on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 10.0f, 11.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 100, 0, '*', std::make_shared<FreezeOnHit>(5, 0.0f));
    spell.update(board);

    REQUIRE(enemy->hp == 900);
    REQUIRE(enemy->freezeTicks == 5);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.0f));
}

TEST_CASE("AreaSpell with no on-hit effect (default) leaves targets unfrozen", "[area_spell][on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 10.0f, 11.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 100, 0); // no onHit passed -- defaults to nullptr
    spell.update(board);

    REQUIRE(enemy->hp == 900);
    REQUIRE(enemy->freezeTicks == 0);
}

TEST_CASE("AreaSpell hits flying targets by default (groundOnly false, matching most spells)", "[area_spell][flying]") {
    Board board;
    auto flyingEnemy = std::make_shared<DummyEntity>(1, 10.0f, 11.0f, 1000, 1);
    flyingEnemy->isFlying = true;
    spawn(board, flyingEnemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 500, 0); // groundOnly defaults false
    spell.update(board);

    REQUIRE(flyingEnemy->hp == 500);
}

TEST_CASE("AreaSpell with groundOnly skips flying targets but still hits grounded ones", "[area_spell][flying]") {
    Board board;
    auto flyingEnemy = std::make_shared<DummyEntity>(1, 10.0f, 11.0f, 1000, 1);
    flyingEnemy->isFlying = true;
    auto groundedEnemy = std::make_shared<DummyEntity>(2, 10.0f, 12.0f, 1000, 1); // isFlying stays false
    spawn(board, flyingEnemy);
    spawn(board, groundedEnemy);

    AreaSpell spell(3, 10.0f, 10.0f, 0, 3.0f, 500, 0, '*', nullptr, true); // groundOnly = true
    spell.update(board);

    REQUIRE(flyingEnemy->hp == 1000);  // untouched: flying, and this spell is ground-only
    REQUIRE(groundedEnemy->hp == 500); // still hit: not flying
}

// ---------------- multi-tick spells (Poison, Arrows) ----------------

TEST_CASE("AreaSpell with remainingHits > 1 applies damage repeatedly, waiting tickInterval between hits", "[area_spell][repeat]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 100, 0, '*', nullptr, false, 3, 2); // 3 hits, 2-tick gap

    spell.update(board); // 1st application lands immediately (delayTicks starts at 0)
    REQUIRE(enemy->hp == 900);
    REQUIRE(spell.isAlive()); // 2 more hits left

    spell.update(board); // gap tick (delayTicks 2 -> 1)
    spell.update(board); // gap tick (1 -> 0)
    REQUIRE(enemy->hp == 900); // still no 2nd hit

    spell.update(board); // 2nd application
    REQUIRE(enemy->hp == 800);
    REQUIRE(spell.isAlive());

    spell.update(board); // gap
    spell.update(board); // gap
    spell.update(board); // 3rd (final) application
    REQUIRE(enemy->hp == 700);
    REQUIRE_FALSE(spell.isAlive()); // remainingHits exhausted
}

TEST_CASE("AreaSpell with remainingHits > 1 re-evaluates who's in radius on each application", "[area_spell][repeat]") {
    Board board;
    auto entity = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 1000, 1);
    spawn(board, entity);

    AreaSpell spell(2, 10.0f, 10.0f, 0, 3.0f, 100, 0, '*', nullptr, false, 3, 1);
    spell.update(board); // hit 1: still in radius
    REQUIRE(entity->hp == 900);

    entity->position = { 10.0f, 20.0f }; // walks well outside the radius before the next application
    spell.update(board); // gap tick (delayTicks 1 -> 0)
    spell.update(board); // 2nd application would land here, but the entity has left
    REQUIRE(entity->hp == 900); // untouched: no longer in radius, just like the real spell
}

// ---------------- knockback / pull ----------------

TEST_CASE("Positive knockback pushes a hit entity directly away from the spell's position", "[area_spell][knockback]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 6.0f, 5.0f, 1000, 1); // dist 1.0 from (5,5), along +x
    spawn(board, enemy);

    AreaSpell spell(2, 5.0f, 5.0f, 0, 2.5f, 100, 0, '*', nullptr, false, 1, 0, false, 1.0f, 0, 1.0f); // knockback 1.0
    spell.update(board);

    REQUIRE(enemy->hp == 900);
    REQUIRE(enemy->position.x == Catch::Approx(7.0f)); // pushed 1.0 further away from (5,5)
    REQUIRE(enemy->position.y == Catch::Approx(5.0f));
}

TEST_CASE("Negative knockback pulls a hit entity toward the spell's position instead", "[area_spell][knockback]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 8.0f, 5.0f, 1000, 1); // dist 3.0 from (5,5), along +x
    spawn(board, enemy);

    AreaSpell spell(2, 5.0f, 5.0f, 0, 5.5f, 100, 0, '*', nullptr, false, 1, 0, false, 1.0f, 0, -1.5f); // pulls 1.5
    spell.update(board);

    REQUIRE(enemy->hp == 900);
    REQUIRE(enemy->position.x == Catch::Approx(6.5f)); // pulled 1.5 toward (5,5)
    REQUIRE(enemy->position.y == Catch::Approx(5.0f));
}

TEST_CASE("A spell without knockback configured (0.0f, the default) never repositions anyone", "[area_spell][knockback]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 6.0f, 5.0f, 1000, 1);
    spawn(board, enemy);

    AreaSpell spell(2, 5.0f, 5.0f, 0, 2.5f, 100, 0, '*');
    spell.update(board);

    REQUIRE(enemy->position.x == Catch::Approx(6.0f)); // untouched
    REQUIRE(enemy->position.y == Catch::Approx(5.0f));
}

TEST_CASE("Tornado's pull damages a building but never drags it out of position", "[area_spell][knockback][building]") {
    Board board;
    // Real-game rule: Tornado has damaged buildings since a 2020 balance
    // update, but has never been able to displace them -- buildings are
    // stationary regardless of which spell's knockback hits them.
    auto building = std::make_shared<Building>(1, 8.0f, 5.0f, 1000, 1, 'B', 5.0f, 10, 10); // dist 3.0 from (5,5)
    spawn(board, building);

    AreaSpell spell(2, 5.0f, 5.0f, 0, 5.5f, 100, 0, '*', nullptr, false, 1, 0, false, 1.0f, 0, -1.5f); // pulls 1.5
    spell.update(board);

    REQUIRE(building->hp == 900); // still takes the damage
    REQUIRE(building->position.x == Catch::Approx(8.0f)); // never dragged toward (5,5)
    REQUIRE(building->position.y == Catch::Approx(5.0f));
}

TEST_CASE("Positive knockback likewise never pushes a building away", "[area_spell][knockback][building]") {
    Board board;
    auto building = std::make_shared<Building>(1, 6.0f, 5.0f, 1000, 1, 'B', 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, building);

    AreaSpell spell(2, 5.0f, 5.0f, 0, 2.5f, 100, 0, '*', nullptr, false, 1, 0, false, 1.0f, 0, 1.0f); // pushes 1.0
    spell.update(board);

    REQUIRE(building->hp == 900);
    REQUIRE(building->position.x == Catch::Approx(6.0f)); // untouched
    REQUIRE(building->position.y == Catch::Approx(5.0f));
}

// ---------------- clone ----------------

TEST_CASE("A clonesAllies spell duplicates an allied troop in radius, not an enemy one", "[area_spell][clone]") {
    Board board;
    auto ally = std::make_shared<MeleeTroop>(1, 6.0f, 5.0f, 500, 0, 0.5f, 1.2f, 200, 12, 'K'); // dist 1.0, same team
    auto enemy = std::make_shared<MeleeTroop>(2, 5.0f, 6.0f, 500, 1, 0.5f, 1.2f, 200, 12, 'K'); // dist 1.0, other team
    spawn(board, ally);
    spawn(board, enemy);

    // buffsAllies=false, buffMultiplier/duration unused, knockback=0, spawnOnDetonate=nullptr, clonesAllies=true
    AreaSpell spell(3, 5.0f, 5.0f, 0, 2.5f, 0, 0, '*', nullptr, false, 1, 0, false, 1.0f, 0, 0.0f, nullptr, true);
    spell.update(board);
    board.commitPendingEntities();

    int allyCount = 0, enemyCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->id == ally->id || e->id == enemy->id) continue; // originals
        if (e->team == 0) allyCount++;
        if (e->team == 1) enemyCount++;
    }
    REQUIRE(allyCount == 1);  // the ally got cloned
    REQUIRE(enemyCount == 0); // the enemy did not
    REQUIRE(ally->hp == 500); // the original is untouched
}

// ---------------- top-N-highest-HP targeting (Vines) ----------------

TEST_CASE("targetTopHpCount only affects the N highest-HP entities in radius", "[area_spell][vines]") {
    Board board;
    auto low = std::make_shared<DummyEntity>(1, 9.0f, 10.0f, 300, 1);
    auto mid = std::make_shared<DummyEntity>(2, 10.0f, 9.0f, 600, 1);
    auto high = std::make_shared<DummyEntity>(3, 11.0f, 10.0f, 900, 1);
    spawn(board, low);
    spawn(board, mid);
    spawn(board, high);

    // buffsAllies=false, remainingHits=1, tickInterval=0, knockback=0,
    // spawnOnDetonate=nullptr, clonesAllies=false, targetTopHpCount=2
    AreaSpell spell(4, 10.0f, 10.0f, 0, 5.0f, 100, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 2);
    spell.update(board);

    REQUIRE(low->hp == 300);  // lowest HP of the three: left alone
    REQUIRE(mid->hp == 500);  // one of the top 2: hit
    REQUIRE(high->hp == 800); // the highest: hit
}

TEST_CASE("targetTopHpCount of 0 (the default) hits everyone in radius, same as before", "[area_spell][vines]") {
    Board board;
    auto a = std::make_shared<DummyEntity>(1, 9.0f, 10.0f, 300, 1);
    auto b = std::make_shared<DummyEntity>(2, 11.0f, 10.0f, 300, 1);
    spawn(board, a);
    spawn(board, b);

    AreaSpell spell(3, 10.0f, 10.0f, 0, 5.0f, 100, 0);
    spell.update(board);

    REQUIRE(a->hp == 200);
    REQUIRE(b->hp == 200);
}

// ---------------- tiered target-count damage (Void) ----------------

TEST_CASE("tieredDamage applies the single-target tier when only one entity is caught", "[area_spell][void]") {
    Board board;
    auto only = std::make_shared<DummyEntity>(1, 10.0f, 10.0f, 10000, 1);
    spawn(board, only);

    // tierSingleDamage=340, tierFewDamage=160, tierManyDamage=76
    AreaSpell spell(2, 10.0f, 10.0f, 0, 5.0f, 0, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 0, true, 340, 160, 76);
    spell.update(board);

    REQUIRE(only->hp == 10000 - 340);
}

TEST_CASE("tieredDamage applies the few-targets tier for 2-4 entities caught", "[area_spell][void]") {
    Board board;
    for (int i = 0; i < 3; ++i) {
        spawn(board, std::make_shared<DummyEntity>(i + 1, 10.0f + i * 0.1f, 10.0f, 10000, 1));
    }

    AreaSpell spell(10, 10.0f, 10.0f, 0, 5.0f, 0, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 0, true, 340, 160, 76);
    spell.update(board);

    for (const auto& e : board.getEntities()) {
        if (e->id == 10) continue; // the spell itself
        REQUIRE(e->hp == 10000 - 160);
    }
}

TEST_CASE("tieredDamage applies the many-targets tier for 5+ entities caught", "[area_spell][void]") {
    Board board;
    for (int i = 0; i < 5; ++i) {
        spawn(board, std::make_shared<DummyEntity>(i + 1, 10.0f + i * 0.1f, 10.0f, 10000, 1));
    }

    AreaSpell spell(10, 10.0f, 10.0f, 0, 5.0f, 0, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 0, true, 340, 160, 76);
    spell.update(board);

    for (const auto& e : board.getEntities()) {
        if (e->id == 10) continue;
        REQUIRE(e->hp == 10000 - 76);
    }
}

// ============================================================================
// Rolling sweep -- The Log, Barbarian Barrel (2026-08-28)
// ============================================================================
// These two are dynamic bodies sweeping a rectangular corridor, not static
// circles. The cases below pin the four things that can each go silently
// wrong: the corridor's WIDTH (a full width, not a radius), its REACH measured
// to a target's surface, the once-per-roll damage rule, and the lateral throw.

static AreaSpell makeRoller(int id, float x, float y, int team, int damage,
                            float range, float width, float speed, float knock) {
    AreaSpell s(id, x, y, team, /*radius*/ 0.0f, damage, /*delayTicks*/ 0, 'o');
    s.configureRoll(range, width, speed, knock);
    return s;
}

TEST_CASE("a rolling spell advances along its team's attack direction", "[area_spell][roll]") {
    Board board;
    AreaSpell team0 = makeRoller(1, 9.0f, 10.0f, 0, 100, 4.0f, 3.0f, 1.0f, 0.0f);
    AreaSpell team1 = makeRoller(2, 9.0f, 20.0f, 1, 100, 4.0f, 3.0f, 1.0f, 0.0f);

    team0.update(board);
    team1.update(board);

    // Team 0 defends the low-y half and therefore rolls toward +y; team 1 the
    // mirror. Derived from the team, never passed in -- a spell has no aim.
    REQUIRE(team0.position.y == Catch::Approx(11.0f));
    REQUIRE(team1.position.y == Catch::Approx(19.0f));
}

TEST_CASE("a rolling spell sweeps everything along its corridor, once each", "[area_spell][roll]") {
    Board board;
    auto nearTarget = std::make_shared<DummyEntity>(1, 9.0f, 12.0f, 1000, 1);
    auto farTarget = std::make_shared<DummyEntity>(2, 9.0f, 16.0f, 1000, 1);
    spawn(board, nearTarget);
    spawn(board, farTarget);

    AreaSpell log = makeRoller(3, 9.0f, 10.0f, 0, 200, 8.0f, 3.9f, 1.0f, 0.0f);

    for (int i = 0; i < 8; ++i) log.update(board);

    // Both were in the corridor and both were rolled over exactly ONCE, even
    // though the log spent several ticks on top of each. A per-tick reapply
    // would read 1000 - 200*n here.
    REQUIRE(nearTarget->hp == 800);
    REQUIRE(farTarget->hp == 800);
    REQUIRE_FALSE(log.isAlive());
}

TEST_CASE("the corridor width is a FULL width, not a radius", "[area_spell][roll]") {
    Board board;
    // The Log's 3.9 is a width, so its half-width is 1.95. A DummyEntity has
    // no collision radius of its own and so borrows IMPLICIT_TROOP_RADIUS.
    const float halfWidth = 3.9f / 2.0f;
    const float r = Entity::IMPLICIT_TROOP_RADIUS;

    auto inside = std::make_shared<DummyEntity>(1, 9.0f + halfWidth + r - 0.05f, 14.0f, 1000, 1);
    auto outside = std::make_shared<DummyEntity>(2, 9.0f + halfWidth + r + 0.05f, 14.0f, 1000, 1);
    spawn(board, inside);
    spawn(board, outside);

    AreaSpell log = makeRoller(3, 9.0f, 10.0f, 0, 200, 8.0f, 3.9f, 1.0f, 0.0f);
    for (int i = 0; i < 8; ++i) log.update(board);

    REQUIRE(inside->hp == 800);
    // Read as a radius, 3.9 would make the corridor 7.8 wide and catch this.
    REQUIRE(outside->hp == 1000);
}

TEST_CASE("a rolling spell stops at its range and reaches no further", "[area_spell][roll]") {
    Board board;
    auto within = std::make_shared<DummyEntity>(1, 9.0f, 10.0f + 4.0f, 1000, 1);
    auto beyond = std::make_shared<DummyEntity>(2, 9.0f, 10.0f + 9.0f, 1000, 1);
    spawn(board, within);
    spawn(board, beyond);

    AreaSpell log = makeRoller(3, 9.0f, 10.0f, 0, 200, 5.0f, 3.9f, 1.0f, 0.0f);
    for (int i = 0; i < 12; ++i) log.update(board);

    REQUIRE(within->hp == 800);
    REQUIRE(beyond->hp == 1000);
    REQUIRE_FALSE(log.isAlive());
    // It travelled exactly its range, not one tick's worth further.
    REQUIRE(log.position.y == Catch::Approx(15.0f));
}

TEST_CASE("The Log's 10.1 range reaches a Princess Tower from the bridge", "[area_spell][roll][bridge]") {
    // The interaction the number exists for. Asserted against ArenaLayout
    // rather than a hardcoded 9.0, so moving the arena moves this test with it
    // instead of silently invalidating it -- the failure mode CLAUDE.md
    // records for every other copy of this geometry.
    const float bridgeY = ArenaLayout::BRIDGE_Y;
    const float towerY = ArenaLayout::princessY(1);
    const float towerRadius = 1.5f;

    const float toSurface = (towerY - towerRadius) - bridgeY;
    const float toCentre = towerY - bridgeY;

    REQUIRE(toSurface < 10.1f);   // the roll reaches the tower
    REQUIRE(toCentre > 10.1f);    // ...and does not reach its centre

    Board board;
    auto tower = std::make_shared<Building>(1, ArenaLayout::LEFT_LANE_X, towerY, 2534, 1, 'T', 7.5f, 50, 10);
    spawn(board, tower);

    AreaSpell log = makeRoller(2, ArenaLayout::LEFT_LANE_X, bridgeY, 0, 269, 10.1f, 3.9f, 1.0f, 0.8f);
    for (int i = 0; i < 12; ++i) log.update(board);

    REQUIRE(tower->hp == 2534 - 269);
}

TEST_CASE("The Log throws edge-caught units SIDEWAYS and centred ones FORWARD", "[area_spell][roll][knockback]") {
    // The tactical point of the card: it splits a group apart rather than
    // shunting it back as one block. A radial pushAway from the log's centre
    // cannot express this, which is why pushAlong exists.
    Board board;
    const float halfWidth = 3.9f / 2.0f;
    auto centre = std::make_shared<DummyEntity>(1, 9.0f, 14.0f, 1000, 1);
    auto leftEdge = std::make_shared<DummyEntity>(2, 9.0f - halfWidth, 14.0f, 1000, 1);
    auto rightEdge = std::make_shared<DummyEntity>(3, 9.0f + halfWidth, 14.0f, 1000, 1);
    spawn(board, centre);
    spawn(board, leftEdge);
    spawn(board, rightEdge);

    AreaSpell log = makeRoller(4, 9.0f, 10.0f, 0, 100, 8.0f, 3.9f, 1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) log.update(board);

    // Dead centre: thrown straight along the roll, no lateral drift at all.
    REQUIRE(centre->position.x == Catch::Approx(9.0f));
    REQUIRE(centre->position.y == Catch::Approx(15.0f));

    // At the edges: thrown purely sideways, AWAY from the axis and in
    // OPPOSITE directions -- which is what separates a group.
    REQUIRE(leftEdge->position.x == Catch::Approx(9.0f - halfWidth - 1.0f));
    REQUIRE(leftEdge->position.y == Catch::Approx(14.0f));
    REQUIRE(rightEdge->position.x == Catch::Approx(9.0f + halfWidth + 1.0f));
    REQUIRE(rightEdge->position.y == Catch::Approx(14.0f));

    REQUIRE(rightEdge->position.x - leftEdge->position.x
            > halfWidth * 2.0f);   // strictly further apart than they started
}

TEST_CASE("a rolling spell never moves a building, only damages it", "[area_spell][roll][knockback]") {
    // exemptFromForcedMovement, enforced inside pushAlong exactly as it is for
    // pushAway/pullToward -- a Log must not shove a Cannon out of its lane.
    Board board;
    auto cannon = std::make_shared<Building>(1, 10.0f, 14.0f, 824, 1, 'C', 5.5f, 60, 10);
    spawn(board, cannon);

    AreaSpell log = makeRoller(2, 9.0f, 10.0f, 0, 200, 8.0f, 3.9f, 1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) log.update(board);

    REQUIRE(cannon->hp == 624);
    REQUIRE(cannon->position.x == Catch::Approx(10.0f));
    REQUIRE(cannon->position.y == Catch::Approx(14.0f));
}

TEST_CASE("a rolling spell is ground-only when configured so", "[area_spell][roll]") {
    Board board;
    auto flyer = std::make_shared<DummyEntity>(1, 9.0f, 14.0f, 1000, 1);
    flyer->isFlying = true;
    spawn(board, flyer);

    AreaSpell log(2, 9.0f, 10.0f, 0, 0.0f, 200, 0, 'o', nullptr, /*groundOnly*/ true);
    log.configureRoll(8.0f, 3.9f, 1.0f, 0.0f);
    for (int i = 0; i < 8; ++i) log.update(board);

    REQUIRE(flyer->hp == 1000);
}

TEST_CASE("a rolling spell ignores what is behind its spawn point", "[area_spell][roll]") {
    Board board;
    auto behind = std::make_shared<DummyEntity>(1, 9.0f, 6.0f, 1000, 1);
    spawn(board, behind);

    AreaSpell log = makeRoller(2, 9.0f, 10.0f, 0, 200, 8.0f, 3.9f, 1.0f, 0.0f);
    for (int i = 0; i < 8; ++i) log.update(board);

    REQUIRE(behind->hp == 1000);
}

TEST_CASE("the registry gives The Log and Barbarian Barrel their real corridors", "[area_spell][roll][registry]") {
    // The registration itself, so a future edit that drops withRollingSweep
    // turns these back into static circles LOUDLY rather than silently.
    // Asserted on CardDefinition rather than CardStats because that is the
    // struct GameLogger's cardMeta block is built from -- so this also pins
    // the shape the replay hands web/viewer.html, which cannot derive it.
    const auto& registry = CardRegistry::getInstance();

    const CardDefinition* log = registry.getCard(33);
    REQUIRE(log != nullptr);
    REQUIRE(log->name == "The Log");
    REQUIRE(log->rollWidth == Catch::Approx(3.9f));
    REQUIRE(log->rollRange == Catch::Approx(10.1f));

    const CardDefinition* barrel = registry.getCard(101);
    REQUIRE(barrel != nullptr);
    REQUIRE(barrel->name == "Barbarian Barrel");
    REQUIRE(barrel->rollWidth == Catch::Approx(2.6f));
    REQUIRE(barrel->rollRange == Catch::Approx(4.5f));

    // Every non-rolling spell must stay at zero, or the viewer would draw a
    // rectangle for a Fireball.
    const CardDefinition* fireball = registry.getCard(7);
    REQUIRE(fireball != nullptr);
    REQUIRE(fireball->rollRange == Catch::Approx(0.0f));
}
