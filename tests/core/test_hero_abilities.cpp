#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardRegistry.h"
#include "GameManager.h"
#include "HeroMiniPekkaBoostEffect.h"
#include "SpawnOnAbility.h"
#include "HeroKnightTauntEffect.h"
#include "HeroWizardFieryFlightEffect.h"
#include "HeroGiantHurlEffect.h"
#include "HeroMegaMinionWarpEffect.h"
#include "HeroMagicArcherTripleThreatEffect.h"
#include "HeroIceGolemSnowstormEffect.h"
#include "AreaSpell.h"
#include "HeroBarbarianBarrelRerollEffect.h"
#include "TargetingHelpers.h"
#include "Tower.h"

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

// ---------------- Hero Wizard (167): temporary flight + on-hit tornado pulse ----------------

TEST_CASE("Hero Wizard (167) is registered with base Wizard's stats and the Fiery Flight ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(167);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    REQUIRE_FALSE(def->isChampion);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 755); // base Wizard's own hp, copied verbatim -- see CardRegistry.h id 11
    REQUIRE(hero->splashRadius == Catch::Approx(1.5f));
    REQUIRE(hero->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(hero->abilityCooldownTicks == 200);
    REQUIRE_FALSE(hero->isFlying); // not yet activated
}

TEST_CASE("HeroWizardFieryFlightEffect grants flight for a fixed duration, then reverts it", "[hero_wizard]") {
    Board board;
    auto wizard = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 755, 0, 5.5f, 281, 14);
    spawn(board, wizard);
    REQUIRE_FALSE(wizard->isFlying);

    HeroWizardFieryFlightEffect effect(3, 4.0f, 20, 0.5f); // short 3-tick duration for a fast test
    effect.apply(board, *wizard);
    REQUIRE(wizard->isFlying);

    wizard->update(board);
    wizard->update(board);
    REQUIRE(wizard->isFlying); // still flying after 2 of 3 ticks

    wizard->update(board); // the 3rd tick: window elapses
    REQUIRE_FALSE(wizard->isFlying);
}

TEST_CASE("A landed attack during the flight window pulses damage/pull centered on the target, not the caster",
        "[hero_wizard]") {
    Board board;
    auto wizard = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 755, 0, 10.0f, 281, 14); // wide range: lands a hit this tick
    auto target = std::make_shared<StationaryCombatant>(2, 5.0f, 0.0f, 10000, 1, 1.0f, 100, 10);
    // Bystander near the TARGET (not near the caster) -- proves the pulse
    // is centered on target->position, unlike onHitPullRadius's self-centered pull.
    auto bystander = std::make_shared<StationaryCombatant>(3, 5.5f, 0.0f, 10000, 1, 1.0f, 100, 10);
    spawn(board, wizard);
    spawn(board, target);
    spawn(board, bystander);

    HeroWizardFieryFlightEffect effect(50, 4.0f, 20, 0.5f);
    effect.apply(board, *wizard);

    wizard->update(board); // lands the attack (currentCooldown starts at 0) -- fires the pulse too

    REQUIRE(bystander->hp < 10000); // caught by the pulse's splash damage, despite never being wizard's own attack target
}

// ---------------- findHpExtremeEnemy ----------------

TEST_CASE("findHpExtremeEnemy picks the correct entity among mixed HP and excludes Towers", "[targeting_helpers]") {
    Board board;
    auto lowHp = std::make_shared<DummyEntity>(1, 1.0f, 0.0f, 100, 1);
    auto midHp = std::make_shared<DummyEntity>(2, 2.0f, 0.0f, 500, 1);
    auto highHp = std::make_shared<DummyEntity>(3, 3.0f, 0.0f, 900, 1);
    auto tower = std::make_shared<Tower>(4, 4.0f, 0.0f, 5000, 1, 7.0f, 90, 10, 'R'); // higher hp than everything else, but excluded
    auto ally = std::make_shared<DummyEntity>(5, 0.5f, 0.0f, 50, 0); // own team, excluded regardless of hp
    spawn(board, lowHp);
    spawn(board, midHp);
    spawn(board, highHp);
    spawn(board, tower);
    spawn(board, ally);

    auto highest = findHpExtremeEnemy(board, Vector2D{ 0.0f, 0.0f }, 0.0f, 0, /*wantHighestHp=*/true);
    REQUIRE(highest != nullptr);
    REQUIRE(highest->id == 3); // highHp, not the Tower

    auto lowest = findHpExtremeEnemy(board, Vector2D{ 0.0f, 0.0f }, 0.0f, 0, /*wantHighestHp=*/false);
    REQUIRE(lowest != nullptr);
    REQUIRE(lowest->id == 1); // lowHp
}

TEST_CASE("findHpExtremeEnemy respects a bounded maxRadius", "[targeting_helpers]") {
    Board board;
    auto nearLowHp = std::make_shared<DummyEntity>(1, 1.0f, 0.0f, 100, 1);
    auto farHigherHp = std::make_shared<DummyEntity>(2, 10.0f, 0.0f, 900, 1); // outside a 3.0 radius
    spawn(board, nearLowHp);
    spawn(board, farHigherHp);

    auto highest = findHpExtremeEnemy(board, Vector2D{ 0.0f, 0.0f }, 3.0f, 0, /*wantHighestHp=*/true);
    REQUIRE(highest != nullptr);
    REQUIRE(highest->id == 1); // farHigherHp is out of range, so nearLowHp wins by default
}

// ---------------- Hero Giant (169) ----------------

TEST_CASE("Hero Giant (169) is registered with base Giant's stats and the Heroic Hurl ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(169);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 3968); // base Giant's own hp, copied verbatim -- see CardRegistry.h id 2
    REQUIRE(hero->abilityElixirCost == Catch::Approx(2.0f));
    REQUIRE(hero->abilityCooldownTicks == 140);
}

TEST_CASE("HeroGiantHurlEffect throws the highest-HP enemy in range across the lane and stuns it",
        "[hero_giant]") {
    Board board;
    auto giant = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 3968, 0, 1.2f, 253, 15);
    auto victim = std::make_shared<StationaryCombatant>(2, 6.0f, 5.0f, 1000, 1, 1.0f, 100, 10); // dist 1.0, within 3.0 grab range
    spawn(board, giant);
    spawn(board, victim);

    HeroGiantHurlEffect effect(3.0f, 20);
    effect.apply(board, *giant);

    // Board width isn't asserted directly here (board geometry is a
    // separate concern) -- only that the victim's X actually flipped
    // (mirrored) and it's now stunned (fully frozen).
    REQUIRE(victim->position.x != Catch::Approx(6.0f));
    REQUIRE(victim->freezeTicks == 20);
    REQUIRE(victim->freezeSlow == Catch::Approx(0.0f));
}

// ---------------- Hero Mega Minion (173) ----------------

TEST_CASE("Hero Mega Minion (173) is registered with base Mega Minion's stats, a one-use ability, and a post-spawn lockout",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(173);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 837); // base Mega Minion's own hp, copied verbatim -- see CardRegistry.h id 43
    REQUIRE(hero->isFlying);
    REQUIRE(hero->abilityElixirCost == Catch::Approx(2.0f));
    REQUIRE(hero->abilityUsesRemaining == 1);
    REQUIRE(hero->abilityCooldownRemaining == 15); // post-spawn lockout, seeded via CardFactories::applyCardMetadata
}

TEST_CASE("HeroMegaMinionWarpEffect teleports to the lowest-HP enemy anywhere and deals bonus damage",
        "[hero_mega_minion]") {
    Board board;
    auto minion = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 837, 0, 1.6f, 312, 15);
    auto farLowHp = std::make_shared<DummyEntity>(2, 30.0f, 30.0f, 400, 1); // far away -- infinite range still finds it
    auto nearHigherHp = std::make_shared<DummyEntity>(3, 1.0f, 0.0f, 900, 1);
    spawn(board, minion);
    spawn(board, farLowHp);
    spawn(board, nearHigherHp);

    HeroMegaMinionWarpEffect effect(300);
    effect.apply(board, *minion);

    REQUIRE(minion->position.x == Catch::Approx(30.0f)); // teleported to the lowest-HP enemy, not the nearest one
    REQUIRE(minion->position.y == Catch::Approx(30.0f));
    REQUIRE(farLowHp->hp == 100); // 400 - 300 bonus damage
}

// ---------------- Hero Magic Archer (171) ----------------

TEST_CASE("Hero Magic Archer (171) is registered with base Magic Archer's stats and the Triple Threat ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(171);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 529); // base Magic Archer's own hp, copied verbatim -- see CardRegistry.h id 63
    REQUIRE(hero->abilityElixirCost == Catch::Approx(2.0f));
    REQUIRE(hero->abilityCooldownTicks == 250);
    REQUIRE(hero->maxSplitTargets == 1); // not yet activated
}

TEST_CASE("HeroMagicArcherTripleThreatEffect dashes back, spawns a decoy, and grants a temporary split-target window",
        "[hero_magic_archer]") {
    Board board;
    auto archer = std::make_shared<StationaryCombatant>(1, 5.0f, 20.0f, 529, 0, 7.0f, 143, 11); // team 0
    spawn(board, archer);

    CardStats decoyStats;
    decoyStats.name = "Decoy";
    decoyStats.archetype = Archetype::MeleeSquad;
    decoyStats.hp = 100; decoyStats.speed = 0.0f; decoyStats.attackRange = 1.0f;
    decoyStats.damage = 0; decoyStats.attackCooldown = 100; decoyStats.symbol = 'd';
    HeroMagicArcherTripleThreatEffect effect(5.0f, decoyStats, 70);

    effect.apply(board, *archer);
    board.commitPendingEntities();

    REQUIRE(archer->position.y == Catch::Approx(15.0f)); // 20 - 5, toward team 0's own side (lower y)
    REQUIRE(archer->maxSplitTargets == 3);
    REQUIRE(archer->temporarySplitTargetsTicksRemaining == 70);

    int decoyCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Decoy") {
            decoyCount++;
            REQUIRE(e->position.x == Catch::Approx(5.0f)); // spawned at the archer's OLD position
            REQUIRE(e->position.y == Catch::Approx(20.0f));
        }
    }
    REQUIRE(decoyCount == 1);
}

TEST_CASE("The temporary split-target window restores maxSplitTargets once it elapses", "[hero_magic_archer]") {
    Board board;
    auto archer = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 529, 0, 7.0f, 143, 11);
    spawn(board, archer);

    CardStats decoyStats;
    decoyStats.name = "Decoy";
    decoyStats.archetype = Archetype::MeleeSquad;
    decoyStats.hp = 100; decoyStats.speed = 0.0f; decoyStats.attackRange = 1.0f;
    decoyStats.damage = 0; decoyStats.attackCooldown = 100; decoyStats.symbol = 'd';
    HeroMagicArcherTripleThreatEffect effect(5.0f, decoyStats, 3); // short 3-tick window for a fast test

    effect.apply(board, *archer);
    REQUIRE(archer->maxSplitTargets == 3);

    archer->update(board);
    archer->update(board);
    REQUIRE(archer->maxSplitTargets == 3); // still active after 2 of 3 ticks

    archer->update(board); // the 3rd tick: window elapses
    REQUIRE(archer->maxSplitTargets == 1);
}

// ---------------- Hero Ice Golem (175) ----------------

TEST_CASE("Hero Ice Golem (175) is registered with base Ice Golem's stats and the Snowstorm ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(175);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hero = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->hp == 1315); // base Ice Golem's own hp, copied verbatim -- see CardRegistry.h id 40
    REQUIRE(hero->abilityElixirCost == Catch::Approx(2.0f));
    REQUIRE(hero->abilityCooldownTicks == 170);
}

TEST_CASE("HeroIceGolemSnowstormEffect's first blast damages/slows, discounting Tower damage",
        "[hero_ice_golem]") {
    Board board;
    auto golem = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1315, 0, 0.75f, 84, 25);
    auto enemyTroop = std::make_shared<StationaryCombatant>(2, 5.5f, 5.0f, 10000, 1, 1.0f, 100, 10);
    auto enemyTower = std::make_shared<Tower>(3, 4.5f, 5.0f, 5000, 1, 7.0f, 90, 10, 'R');
    spawn(board, golem);
    spawn(board, enemyTroop);
    spawn(board, enemyTower);

    HeroIceGolemSnowstormEffect effect(4.0f, 80, 1.0f, 20, 0.6f, 15);
    effect.apply(board, *golem);
    board.commitPendingEntities();

    std::vector<std::shared_ptr<AreaSpell>> blasts;
    for (const auto& e : board.getEntities()) {
        if (auto spell = std::dynamic_pointer_cast<AreaSpell>(e)) blasts.push_back(spell);
    }
    REQUIRE(blasts.size() == 3);

    blasts[0]->update(board); // delayTicks=0: fires immediately
    REQUIRE(enemyTroop->hp == 10000 - 80);
    REQUIRE(enemyTower->hp == 5000 - 4); // 80 * 0.05 spellTowerDamageMultiplier, rounded down
    REQUIRE(enemyTroop->freezeTicks == 20);
    REQUIRE(enemyTroop->freezeSlow == Catch::Approx(0.6f)); // partial slow, not a full freeze
}

TEST_CASE("HeroIceGolemSnowstormEffect's third blast fully freezes instead of slowing", "[hero_ice_golem]") {
    Board board;
    auto golem = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1315, 0, 0.75f, 84, 25);
    auto enemyTroop = std::make_shared<StationaryCombatant>(2, 5.5f, 5.0f, 10000, 1, 1.0f, 100, 10);
    spawn(board, golem);
    spawn(board, enemyTroop);

    HeroIceGolemSnowstormEffect effect(4.0f, 80, 1.0f, 20, 0.6f, 15);
    effect.apply(board, *golem);
    board.commitPendingEntities();

    std::vector<std::shared_ptr<AreaSpell>> blasts;
    for (const auto& e : board.getEntities()) {
        if (auto spell = std::dynamic_pointer_cast<AreaSpell>(e)) blasts.push_back(spell);
    }
    REQUIRE(blasts.size() == 3);

    for (int i = 0; i < 11; ++i) blasts[2]->update(board); // delayTicks=10: needs 11 calls to fire
    REQUIRE(enemyTroop->freezeTicks == 15);
    REQUIRE(enemyTroop->freezeSlow == Catch::Approx(0.0f)); // full freeze, not a partial slow
}

TEST_CASE("AreaSpell's spellTowerDamageMultiplier only discounts Tower targets, not ordinary troops",
        "[targeting_helpers][area_spell]") {
    Board board;
    auto troop1 = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 1000, 1, 1.0f, 100, 10);
    spawn(board, troop1);

    auto spell = std::make_shared<AreaSpell>(
        board.allocateId(), 5.0f, 5.0f, 0, 4.0f, 100, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 0, false, 0, 0, 0, 0.05f);
    board.addEntity(spell);
    board.commitPendingEntities();

    spell->update(board);
    REQUIRE(troop1->hp == 1000 - 100); // full damage -- the discount only ever applies to isTower() targets
}

// ---------------- Hero Barbarian Barrel (174) ----------------

TEST_CASE("Hero Barbarian Barrel (174) is registered as isHero, and spawns a Hero-flagged Barbarian with the Rowdy Reroll ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(174);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero); // deck-slot-legality flag, even though the spell entity itself carries no ability
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    // The spell itself has delayTicks=8 (see the base Barbarian Barrel's
    // own registration) -- its spawnOnDetonate (the Barbarian) only fires
    // partway through AreaSpell::update(), not at construction, so this
    // needs to be driven forward before the Barbarian exists at all.
    std::shared_ptr<AreaSpell> spell;
    for (const auto& e : board.getEntities()) {
        spell = std::dynamic_pointer_cast<AreaSpell>(e);
        if (spell) break;
    }
    REQUIRE(spell != nullptr);
    for (int i = 0; i < 9; ++i) spell->update(board);
    board.commitPendingEntities();

    std::shared_ptr<CombatEntity> barbarian;
    for (const auto& e : board.getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce) { barbarian = ce; break; }
    }
    REQUIRE(barbarian != nullptr);
    REQUIRE(barbarian->isHero);
    REQUIRE(barbarian->hp == 691); // base Barbarian's own hp, copied verbatim -- see barbarianBarrelBarbarianStats
    REQUIRE(barbarian->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(barbarian->abilityUsesRemaining == 1);
}

// ---------------- Hero Goblins (172) -- post-death squad reactivation ----------------
// Unlike every other Hero, Hero Goblins has no alive-path ability at all --
// "Banner Brigade" only becomes activatable once the whole 4-unit squad is
// dead, within a window, via GameManager::isPostDeathAbilityReady/
// activateChampionAbility's post-death branch (see PlayerState::
// ChampionSlotState::lastSquadWipeTick, GameManager::syncChampionCooldowns).

TEST_CASE("Hero Goblins (172) is registered with base Goblins' stats and no alive-path ability",
        "[card_registry][hero]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(172);
    REQUIRE(def != nullptr);
    REQUIRE(def->isHero);
    REQUIRE_FALSE(def->isChampion);
    REQUIRE(def->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(def->abilityUsableAfterDeathTicks == 70);
    REQUIRE(def->postDeathAbilityEffect != nullptr);

    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    int squadCount = 0;
    for (const auto& e : board.getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!ce) continue;
        squadCount++;
        REQUIRE(ce->hp == 202); // base Goblins' own hp, copied verbatim -- see CardRegistry.h id 4
        REQUIRE(ce->isHero); // deck-slot-legality flag
        REQUIRE(ce->abilityEffect == nullptr); // no alive-path ability -- reactivation only, post-death
    }
    REQUIRE(squadCount == 4); // 4-unit squad, same offsets as base Goblins
}

TEST_CASE("Hero Goblins' Banner Brigade is unavailable while any squad member is still alive",
        "[game_manager][hero][goblins]") {
    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step(); // commits the pending squad and runs syncChampionCooldowns once

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));

    // Kill 3 of the 4 -- one survivor is enough to keep the ability locked.
    int killed = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0 && killed < 3) {
            ce->takeDamage(ce->hp);
            killed++;
        }
    }
    game.step();

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1));
}

TEST_CASE("Hero Goblins' Banner Brigade activates within the window after the last goblin dies, spawning a fresh plain squad",
        "[game_manager][hero][goblins]") {
    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step();

    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) ce->takeDamage(ce->hp);
    }
    game.step(); // detects the full wipe -- lastSquadWipeTick set this tick

    game.playerAI.elixir = 10.0f;
    REQUIRE(game.isChampionAbilityReady(0, 1));
    float elixirBefore = game.getElixirAI();

    REQUIRE(game.activateChampionAbility(0, 1));
    REQUIRE(game.getElixirAI() == Catch::Approx(elixirBefore - 1.0f));
    game.step(); // commits the freshly-spawned squad

    // The reactivated squad spawns via a sentinel internal cardId (-48, see
    // CardRegistry.h's Hero Goblins registration), never the Hero's own
    // deck id (172) -- distinguishing it from a (nonexistent, here) live
    // remnant of the original squad.
    int freshCount = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!ce || ce->cardId != -48) continue;
        REQUIRE(ce->isAlive());
        REQUIRE(ce->hp == 202);
        // Reactivated squad is plain (non-Hero) Goblins -- no ability at
        // all, so a second Banner Brigade can never chain off of it.
        REQUIRE_FALSE(ce->isHero);
        freshCount++;
    }
    REQUIRE(freshCount == 4);
}

TEST_CASE("Hero Goblins' Banner Brigade reactivates the squad at the last-known death position, not the original deploy point",
        "[game_manager][hero][goblins]") {
    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step(); // one tick of movement away from the exact deploy point (9.0, 10.0)

    // Death position capture uses whichever squad member the internal scan
    // happens to observe last (see syncChampionCooldowns), which isn't
    // necessarily any particular one of the 4 -- each carries its own
    // +-0.5-tile squad offset from the group's shared center, so this
    // compares against the CENTROID of all 4 (the group's actual shared
    // position) rather than any single member's own offset position.
    Vector2D centroidBefore{ 0.0f, 0.0f };
    int countBefore = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) {
            centroidBefore.x += ce->position.x;
            centroidBefore.y += ce->position.y;
            countBefore++;
            ce->takeDamage(ce->hp);
        }
    }
    REQUIRE(countBefore == 4);
    centroidBefore.x /= countBefore;
    centroidBefore.y /= countBefore;
    // Confirms this scenario actually exercises movement, not a no-op --
    // otherwise this test couldn't distinguish death position from deploy.
    REQUIRE_FALSE((centroidBefore.x == Catch::Approx(9.0f) && centroidBefore.y == Catch::Approx(10.0f)));
    game.step(); // wipe detected this tick, position captured off this same array

    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    game.step(); // commits the freshly-spawned squad

    Vector2D centroidAfter{ 0.0f, 0.0f };
    int countAfter = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!ce || ce->cardId != -48) continue;
        centroidAfter.x += ce->position.x;
        centroidAfter.y += ce->position.y;
        countAfter++;
    }
    REQUIRE(countAfter == 4);
    centroidAfter.x /= countAfter;
    centroidAfter.y /= countAfter;
    // Reactivated near the death position, not the original (9.0, 10.0)
    // deploy point -- syncChampionCooldowns' scan intentionally captures
    // whichever single squad member it happens to observe last, not a true
    // centroid (a documented approximation, same category as this file's
    // other engine-internal constants), so this tolerance covers that
    // member's own +-0.5-tile squad offset in both axes (up to the full
    // diagonal, ~1.41 tiles) rather than asserting exact centroid equality.
    REQUIRE(std::abs(centroidAfter.x - centroidBefore.x) < 2.0f);
    REQUIRE(std::abs(centroidAfter.y - centroidBefore.y) < 2.0f);
}

TEST_CASE("Hero Goblins' Banner Brigade is unavailable once the reactivation window elapses",
        "[game_manager][hero][goblins]") {
    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step();

    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) ce->takeDamage(ce->hp);
    }
    game.step(); // wipe detected this tick

    game.playerAI.elixir = 10.0f;
    for (int i = 0; i < 71; ++i) { // window is 70 ticks -- this overshoots it
        game.step();
        game.playerAI.elixir = 10.0f;
    }

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
}

TEST_CASE("Hero Goblins' Banner Brigade cannot chain a second reactivation off the reactivated squad's own death",
        "[game_manager][hero][goblins]") {
    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step();

    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) ce->takeDamage(ce->hp);
    }
    game.step();
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    game.step(); // commits the reactivated (plain, non-Hero) squad

    // Kill the reactivated squad too -- its cardId (-48, the internal
    // sentinel, see CardRegistry.h) doesn't match deck slot 1's own id
    // (172), so syncChampionCooldowns' squad-wipe scan (keyed on
    // deckConfig[slot]) never observes this death at all, and
    // lastSquadWipeTick (already consumed to -1 by the first activation)
    // stays exactly there.
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isAlive() && ce->cardId == -48) ce->takeDamage(ce->hp);
    }
    game.step();
    game.playerAI.elixir = 10.0f;

    REQUIRE_FALSE(game.isChampionAbilityReady(0, 1));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
}

TEST_CASE("HeroBarbarianBarrelRerollEffect rolls forward, damages enemies in the line, and halves damage against Towers",
        "[hero_barbarian_barrel]") {
    Board board;
    auto barbarian = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 691, 0, 0.7f, 192, 14); // team 0
    auto enemyTroop = std::make_shared<StationaryCombatant>(2, 5.0f, 7.0f, 1000, 1, 1.0f, 100, 10); // in the roll's path
    auto enemyTower = std::make_shared<Tower>(3, 5.0f, 8.0f, 5000, 1, 7.0f, 90, 10, 'R'); // also in the roll's path
    spawn(board, barbarian);
    spawn(board, enemyTroop);
    spawn(board, enemyTower);

    HeroBarbarianBarrelRerollEffect effect(3.0f, 0.7f, 233);
    effect.apply(board, *barbarian);

    REQUIRE(barbarian->position.y == Catch::Approx(8.0f)); // 5 + 3, forward toward the enemy half
    REQUIRE(enemyTroop->hp == 1000 - 233); // full roll damage
    REQUIRE(enemyTower->hp == 5000 - 116); // 233 / 2 = 116 (integer division), halved vs Towers
}
