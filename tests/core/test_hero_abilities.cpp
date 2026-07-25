#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardRegistry.h"
#include "GameManager.h"
#include "HeroMiniPekkaBoostEffect.h"
#include "SpawnOnAbility.h"
#include "HeroKnightTauntEffect.h"

// ---------------- registry wiring: one sanity check per Hero ----------------
// Stage 1 pilot pair -- both use zero new engine primitives, proving the
// generalized isHero/championSlots path end-to-end (see CardStats::isHero's
// own comment) before later stages introduce genuinely new mechanics.

TEST_CASE("Hero Mini P.E.K.K.A. (170) is registered with base Mini PEKKA's stats and a one-use ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(170);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    REQUIRE_FALSE(def->isChampion); // Heroes are NOT Champions -- additive flag, not a rename
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->isHero);
    REQUIRE_FALSE(hero->isChampion);
    REQUIRE(hero->hp == 1390); // base Mini PEKKA's own hp, copied verbatim -- see CardRegistry.h id 5
    REQUIRE(hero->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(hero->abilityCooldownTicks == 0); // no repeating cooldown -- one-use, see abilityUsesRemaining
    REQUIRE(hero->abilityUsesRemaining == 1);
}

TEST_CASE("Hero Musketeer (168) is registered with base Musketeer's stats and a repeating-cooldown ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(168);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    REQUIRE_FALSE(def->isChampion);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 721); // base Musketeer's own hp, copied verbatim -- see CardRegistry.h id 6
    REQUIRE(hero->sightRange == Catch::Approx(6.0f));
    REQUIRE(hero->abilityElixirCost == Catch::Approx(3.0f));
    REQUIRE(hero->abilityCooldownTicks == 220);
    REQUIRE(hero->abilityUsesRemaining == -1); // unlimited activations, gated only by cooldown/elixir
}

// ---------------- HeroMiniPekkaBoostEffect ----------------

TEST_CASE("HeroMiniPekkaBoostEffect grants a one-time hp+damage boost", "[hero_mini_pekka]") {
    Board board;
    auto hero = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1390, 0, 0.8f, 755, 16);

    HeroMiniPekkaBoostEffect effect(210, 1.15f);
    effect.apply(board, *hero);

    REQUIRE(hero->hp == 1390 + 210);
    REQUIRE(hero->buffDamageMultiplier == Catch::Approx(1.15f));
    REQUIRE(hero->buffTicksRemaining == 999999);
}

// ---------------- SpawnOnAbility reused for Hero Musketeer's turret ----------------

TEST_CASE("Hero Musketeer's turret stats self-destruct after their fixed lifetime, not before",
        "[hero_musketeer]") {
    Board board;
    // Mirrors CardRegistry::heroMusketeerTurretStats() (private to that
    // class) -- only the fixed-lifetime-via-withHpTransform(1.0f,...)
    // mechanism is under test here, not the exact registered stat values.
    CardStats turretStats = CardStats();
    turretStats.name = "Trusty Turret";
    turretStats.archetype = Archetype::DefensiveBuilding;
    turretStats.hp = 200; turretStats.symbol = 't';
    turretStats.attackRange = 4.0f; turretStats.damage = 90; turretStats.attackCooldown = 5;
    turretStats.withTargetsAir().withHpTransform(1.0f, 100, false);

    SpawnOnAbility effect(turretStats);
    auto caster = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1000, 0, 1.0f, 100, 10);
    effect.apply(board, *caster);
    board.commitPendingEntities();

    auto turret = std::dynamic_pointer_cast<CombatEntity>(board.getEntities().back());
    REQUIRE(turret != nullptr);
    REQUIRE(turret->isAlive());

    for (int i = 0; i < 99; ++i) turret->update(board); // 99 ticks in: not yet expired
    REQUIRE(turret->isAlive());

    turret->update(board); // the 100th tick: transform threshold reached, self-destructs
    REQUIRE_FALSE(turret->isAlive());
}

// ---------------- validateDeckSlots: Hero shares Champion's slot rules ----------------

TEST_CASE("validateDeckSlots accepts a Hero in slot 1 or 2, rejects it elsewhere", "[card_registry][hero][deck_slots]") {
    REQUIRE(validateDeckSlots({ 1, 170, 2, 3, 4, 5, 6, 7 }).empty());   // slot 1 (Heroic)
    REQUIRE(validateDeckSlots({ 1, 2, 170, 3, 4, 5, 6, 7 }).empty());   // slot 2 (Wild Card)
    REQUIRE_FALSE(validateDeckSlots({ 170, 1, 2, 3, 4, 5, 6, 7 }).empty()); // slot 0: rejected
    REQUIRE_FALSE(validateDeckSlots({ 1, 2, 3, 170, 4, 5, 6, 7 }).empty()); // slot 3: rejected
}

TEST_CASE("validateDeckSlots accepts one Champion + one Hero across slots 1 and 2, either arrangement",
        "[card_registry][hero][deck_slots]") {
    REQUIRE(validateDeckSlots({ 1, 115, 170, 3, 4, 5, 6, 7 }).empty()); // Champion slot1, Hero slot2
    REQUIRE(validateDeckSlots({ 1, 170, 115, 3, 4, 5, 6, 7 }).empty()); // Hero slot1, Champion slot2
}

TEST_CASE("validateDeckSlots accepts two Heroes, one per special slot", "[card_registry][hero][deck_slots]") {
    REQUIRE(validateDeckSlots({ 1, 170, 168, 3, 4, 5, 6, 7 }).empty());
}

// ---------------- GameManager: the generalized Champion machinery works for isHero too ----------------

TEST_CASE("activateChampionAbility fires for a Hero (not just a Champion) deployed in slot 1", "[game_manager][hero]") {
    GameManager game({ 1, 168, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Hero Musketeer -> slot 1
    game.playerAI.hand[1] = 168; // force into hand -- opening hand is randomized
    game.playerAI.elixir = 100.0f;
    game.playCard(0, 168, 9.0f, 10.0f);
    game.step(); // commits the pending entity so findChampionInSlot can see it

    REQUIRE(game.isChampionAbilityReady(0, 1));
    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1)); // now on cooldown, same as any Champion
}

TEST_CASE("A Hero's one-use ability (usesLimit=1) can't be reactivated once spent", "[game_manager][hero]") {
    GameManager game({ 1, 170, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 }); // Hero Mini P.E.K.K.A -> slot 1
    game.playerAI.hand[1] = 170;
    game.playerAI.elixir = 100.0f;
    game.playCard(0, 170, 9.0f, 10.0f);
    game.step();

    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1)); // usesRemaining exhausted, no cooldown to wait out
}

// ---------------- Hero Knight (166): taunt + expiring shield ----------------

TEST_CASE("Hero Knight (166) is registered with base Knight's stats and the Taunt ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(166);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    REQUIRE_FALSE(def->isChampion);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 1766); // base Knight's own hp, copied verbatim -- see CardRegistry.h id 0
    REQUIRE(hero->abilityElixirCost == Catch::Approx(2.0f));
    REQUIRE(hero->abilityCooldownTicks == 250);
}

TEST_CASE("HeroKnightTauntEffect forces nearby enemies to retarget and grants an expiring shield",
        "[hero_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1766, 0, 1.2f, 202, 12);
    // applyTauntNearby only forces entities it can dynamic_pointer_cast to
    // CombatEntity -- a plain DummyEntity (: Entity, not : CombatEntity)
    // would never pick up forcedTargetEntityId at all.
    auto enemy = std::make_shared<StationaryCombatant>(2, 6.0f, 5.0f, 10000, 1, 1.0f, 100, 10); // dist 1.0, within radius 6.5
    spawn(board, knight);
    spawn(board, enemy);

    HeroKnightTauntEffect effect(880, 50, 6.5f);
    effect.apply(board, *knight);

    REQUIRE(knight->shieldHp == 880);
    REQUIRE(knight->shieldExpiresTicksRemaining == 50);
    REQUIRE(enemy->forcedTargetEntityId == 1);
    REQUIRE(enemy->forcedTargetTicksRemaining == 50);
}

TEST_CASE("A taunted CombatEntity's update() locks onto the taunter regardless of its own normal targeting",
        "[hero_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1766, 0, 1.2f, 202, 12);
    // Wide attack range so both potential targets are already in range this
    // same tick -- isolates the assertion to targeting, not movement/chase.
    auto tauntedAttacker = std::make_shared<StationaryCombatant>(2, 6.0f, 5.0f, 1000, 1, 5.0f, 100, 10);
    // Closer than the Knight -- what tauntedAttacker's own ordinary
    // findTarget() would pick if the taunt weren't overriding it.
    auto trueDecoy = std::make_shared<DummyEntity>(3, 6.5f, 5.0f, 1000, 0);
    spawn(board, knight);
    spawn(board, tauntedAttacker);
    spawn(board, trueDecoy);

    tauntedAttacker->forcedTargetEntityId = knight->id;
    tauntedAttacker->forcedTargetTicksRemaining = 50;
    tauntedAttacker->update(board);

    REQUIRE(tauntedAttacker->lastTargetId == knight->id); // forced onto the Knight, not the closer trueDecoy
}

TEST_CASE("A fixed-duration shield clears once shieldExpiresTicksRemaining runs out, taunt with it", "[hero_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1766, 0, 1.2f, 202, 12);
    spawn(board, knight);

    HeroKnightTauntEffect effect(880, 3, 6.5f); // short 3-tick duration for a fast test
    effect.apply(board, *knight);
    REQUIRE(knight->shieldHp == 880);

    knight->update(board);
    knight->update(board);
    REQUIRE(knight->shieldHp == 880); // still active after 2 of 3 ticks

    knight->update(board); // the 3rd tick: duration elapses
    REQUIRE(knight->shieldHp == 0);
    REQUIRE(knight->shieldExpiresTicksRemaining == 0);
}
