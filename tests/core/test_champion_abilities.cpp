#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardRegistry.h"
#include "GoldenKnightDashEffect.h"
#include "SkeletonKingSoulSummonEffect.h"
#include "ArcherQueenCloakEffect.h"
#include "MonkDeflectEffect.h"
#include "SpawnOnAbility.h"
#include "GoblinsteinLightningLinkEffect.h"
#include "BossBanditGetawayGrenadeEffect.h"
#include "AreaSpell.h"
#include "MeleeTroop.h"
#include "Tower.h"

// ---------------- registry wiring: one sanity check per Champion ----------------

TEST_CASE("Golden Knight (116) is registered with the right stats and ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(116);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto knight = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(knight != nullptr);
    REQUIRE(knight->hp == 1799);
    REQUIRE(knight->isChampion);
    REQUIRE(knight->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(knight->abilityCooldownTicks == 120);
}

TEST_CASE("Skeleton King (117) is registered with soul collection and the right ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(117);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto king = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(king != nullptr);
    REQUIRE(king->hp == 2298);
    REQUIRE(king->soulCollectionRadius == Catch::Approx(5.0f));
    REQUIRE(king->maxSouls == 10);
    REQUIRE(king->abilityElixirCost == Catch::Approx(2.0f));
}

TEST_CASE("Archer Queen (118) is registered as air+ground with the right ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(118);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto queen = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(queen != nullptr);
    REQUIRE(queen->hp == 1000);
    REQUIRE(queen->targetsAir);
    REQUIRE(queen->abilityCooldownTicks == 170);
}

TEST_CASE("Monk (119) is registered with the right stats and ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(119);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto monk = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(monk != nullptr);
    REQUIRE(monk->hp == 2214);
    REQUIRE(monk->abilityElixirCost == Catch::Approx(1.0f));
}

TEST_CASE("Little Prince (120) is registered with a hit-speed ramp and the Guardian ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(120);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto prince = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(prince != nullptr);
    REQUIRE(prince->hp == 698);
    REQUIRE(prince->hitSpeedRampFullTick == 60);
    REQUIRE(prince->abilityCooldownTicks == 300);
}

TEST_CASE("Goblinstein (121) spawns both the Monster (champion) and the Doctor", "[card_registry][champion]") {
    Board board;
    CardRegistry::getInstance().getCard(121)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 2);
    std::shared_ptr<CombatEntity> monster, doctor;
    for (const auto& e : board.getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) monster = ce; else if (ce) doctor = ce;
    }
    REQUIRE(monster != nullptr);
    REQUIRE(doctor != nullptr);
    REQUIRE(monster->hp == 2385);
    REQUIRE(doctor->hp == 721);
    REQUIRE(doctor->targetsAir);
}

TEST_CASE("Boss Bandit (122) is registered with charge and a limited-use ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(122);
    REQUIRE(def != nullptr);
    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto bandit = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(bandit != nullptr);
    REQUIRE(bandit->hp == 2624);
    REQUIRE(bandit->chargeGrantsInvulnerability);
    REQUIRE(bandit->abilityUsesRemaining == 2);
}

// ---------------- GoldenKnightDashEffect ----------------

TEST_CASE("GoldenKnightDashEffect chains to the next-closest enemy once the first one dies", "[golden_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1799, 0, 1.2f, 161, 9);
    auto enemyA = std::make_shared<DummyEntity>(2, 3.0f, 0.0f, 300, 1);  // dies to one 335-damage dash
    auto enemyB = std::make_shared<DummyEntity>(3, 3.0f, 3.0f, 1000, 1); // survives to confirm the chain continued
    spawn(board, knight);
    spawn(board, enemyA);
    spawn(board, enemyB);

    GoldenKnightDashEffect effect(335, 5.5f, 2); // capped at exactly 2 dashes
    effect.apply(board, *knight);

    REQUIRE_FALSE(enemyA->isAlive());
    REQUIRE(enemyB->hp == 1000 - 335); // the chain moved on once enemyA died
}

TEST_CASE("GoldenKnightDashEffect keeps re-hitting the same enemy if it's still the closest and survives", "[golden_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1799, 0, 1.2f, 161, 9);
    auto enemy = std::make_shared<DummyEntity>(2, 3.0f, 0.0f, 100000, 1); // survives every dash
    spawn(board, knight);
    spawn(board, enemy);

    GoldenKnightDashEffect effect(335, 5.5f, 3); // capped at 3 dashes
    effect.apply(board, *knight);

    REQUIRE(enemy->hp == 100000 - 335 * 3); // hit on all 3 dashes -- real behavior, per research
}

TEST_CASE("GoldenKnightDashEffect stops the chain after hitting a Tower", "[golden_knight]") {
    Board board;
    auto knight = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1799, 0, 1.2f, 161, 9);
    auto tower = std::make_shared<Tower>(2, 2.0f, 0.0f, 4008, 1, 7.0f, 90, 10, 'R');
    auto behindTower = std::make_shared<DummyEntity>(3, 2.0f, 2.0f, 1000, 1); // would be next in range
    spawn(board, knight);
    spawn(board, tower);
    spawn(board, behindTower);

    GoldenKnightDashEffect effect(335, 5.5f, 10);
    effect.apply(board, *knight);

    REQUIRE(tower->hp == 4008 - 335);
    REQUIRE(behindTower->hp == 1000); // chain stopped at the Tower
}

// ---------------- SkeletonKingSoulSummonEffect ----------------

TEST_CASE("SkeletonKingSoulSummonEffect spawns base+souls skeletons and consumes the souls", "[skeleton_king]") {
    Board board;
    auto king = std::make_shared<StationaryCombatant>(1, 10.0f, 10.0f, 2298, 0, 1.2f, 180, 16);
    king->soulCount = 4;
    spawn(board, king);

    // Built directly rather than reaching into CardRegistry's private
    // child-stats helper -- only the spawn count/soul-consumption below
    // are under test, not the skeleton's own stats.
    CardStats skeletonStats;
    skeletonStats.name = "Skeletons";
    skeletonStats.archetype = Archetype::MeleeSquad;
    skeletonStats.hp = 81; skeletonStats.speed = 0.7f; skeletonStats.attackRange = 0.5f;
    skeletonStats.damage = 81; skeletonStats.attackCooldown = 11; skeletonStats.symbol = 'k';
    SkeletonKingSoulSummonEffect realEffect(skeletonStats, 6, 3.5f);

    realEffect.apply(board, *king);
    board.commitPendingEntities();

    int skeletonCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Skeletons") skeletonCount++;
    }
    REQUIRE(skeletonCount == 10); // 6 base + 4 souls
    REQUIRE(king->soulCount == 0); // consumed
}

// ---------------- ArcherQueenCloakEffect ----------------

TEST_CASE("ArcherQueenCloakEffect grants temporary invisibility and haste", "[archer_queen]") {
    Board board;
    auto queen = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 1000, 0, 5.0f, 225, 12);

    ArcherQueenCloakEffect effect(35, 1.0f / 2.8f);
    effect.apply(board, *queen);

    REQUIRE(queen->temporaryInvisibilityTicksRemaining == 35);
    REQUIRE_FALSE(queen->isTargetable());
    REQUIRE(queen->temporaryHitSpeedMultiplier == Catch::Approx(1.0f / 2.8f));
}

// ---------------- MonkDeflectEffect ----------------

TEST_CASE("MonkDeflectEffect reduces incoming damage for its duration", "[monk]") {
    Board board;
    auto monk = std::make_shared<StationaryCombatant>(1, 0.0f, 0.0f, 2214, 0, 1.2f, 140, 8);

    MonkDeflectEffect effect(0.35f, 40);
    effect.apply(board, *monk);
    monk->takeDamage(100);

    REQUIRE(monk->hp == 2214 - 35); // 65% reduction
}

// ---------------- SpawnOnAbility (Little Prince's Guardian) ----------------

TEST_CASE("SpawnOnAbility spawns its child at the caster's position and team", "[little_prince]") {
    Board board;
    auto prince = std::make_shared<StationaryCombatant>(1, 7.0f, 8.0f, 698, 1, 5.5f, 104, 12);

    CardStats guardStats;
    guardStats.name = "Guardienne";
    guardStats.archetype = Archetype::MeleeSquad;
    guardStats.hp = 1600; guardStats.speed = 0.5f; guardStats.attackRange = 1.2f;
    guardStats.damage = 217; guardStats.attackCooldown = 12; guardStats.symbol = 'u';
    SpawnOnAbility effect(guardStats);

    effect.apply(board, *prince);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1);
    auto guard = board.getEntities()[0];
    REQUIRE(guard->name == "Guardienne");
    REQUIRE(guard->team == 1);
    REQUIRE(guard->position.x == Catch::Approx(7.0f));
    REQUIRE(guard->position.y == Catch::Approx(8.0f));
}

// ---------------- GoblinsteinLightningLinkEffect ----------------

TEST_CASE("GoblinsteinLightningLinkEffect anchors a repeating shock at the Monster's position", "[goblinstein]") {
    Board board;
    auto monster = std::make_shared<StationaryCombatant>(1, 5.0f, 5.0f, 2385, 0, 1.2f, 128, 15);
    auto enemy = std::make_shared<DummyEntity>(2, 5.0f, 6.0f, 10000, 1); // dist 1.0, within radius 2.0
    spawn(board, monster);
    spawn(board, enemy);

    GoblinsteinLightningLinkEffect effect(2.0f, 107, 5, 40); // 8 hits total
    effect.apply(board, *monster);
    board.commitPendingEntities();

    std::shared_ptr<AreaSpell> shock;
    for (const auto& e : board.getEntities()) {
        shock = std::dynamic_pointer_cast<AreaSpell>(e);
        if (shock) break;
    }
    REQUIRE(shock != nullptr);

    shock->update(board); // no delay: first shock lands immediately
    REQUIRE(enemy->hp == 10000 - 107);
}

// ---------------- BossBanditGetawayGrenadeEffect ----------------

TEST_CASE("BossBanditGetawayGrenadeEffect cloaks and teleports back toward the caster's own side", "[boss_bandit]") {
    Board board;
    auto bandit = std::make_shared<StationaryCombatant>(1, 9.0f, 20.0f, 2624, 0, 0.8f, 245, 11); // team 0

    BossBanditGetawayGrenadeEffect effect(10, 6.0f);
    effect.apply(board, *bandit);

    REQUIRE(bandit->temporaryInvisibilityTicksRemaining == 10);
    REQUIRE(bandit->position.y == Catch::Approx(14.0f)); // 20 - 6, toward team 0's own side (lower y)
    REQUIRE(bandit->position.x == Catch::Approx(9.0f));
}

TEST_CASE("BossBanditGetawayGrenadeEffect teleports the opposite direction for team 1", "[boss_bandit]") {
    Board board;
    auto bandit = std::make_shared<StationaryCombatant>(1, 9.0f, 20.0f, 2624, 1, 0.8f, 245, 11); // team 1

    BossBanditGetawayGrenadeEffect effect(10, 6.0f);
    effect.apply(board, *bandit);

    REQUIRE(bandit->position.y == Catch::Approx(26.0f)); // 20 + 6, toward team 1's own side (higher y)
}
