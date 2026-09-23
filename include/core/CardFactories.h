#pragma once
#include "CardStats.h"
#include "Board.h"
#include "MeleeTroop.h"
#include "RangedTroop.h"
#include "BuildingTargeter.h"
#include "RangedBuildingTargeter.h"
#include "Building.h"
#include "AreaSpell.h"
#include <stdexcept>

// One factory per Archetype (CardStats.h). Nothing here is specific to a card.
namespace CardFactories {

inline void applyOnHit(const std::shared_ptr<CombatEntity>& entity, const CardStats& stats) {
    if (stats.onHit) {
        entity->addOnHitEffect(stats.onHit);
    }
}

// Copies a card's per-entity configuration onto each entity it produces.
inline void applyCardMetadata(const std::shared_ptr<CombatEntity>& entity, const CardStats& stats) {
    entity->name = stats.name;
    entity->cardId = stats.id;
    // Deploy time (CardStats.h, DEPLOY_TIME_TICKS). Set here because every
    // troop and building passes through; spells and deploy effects do not,
    // since a spell has its own delay.
    entity->deployTicksRemaining = DEPLOY_TIME_TICKS;
    entity->isFlying = stats.isFlying;
    entity->targetsAir = stats.targetsAir;
    entity->deathEffect = stats.deathEffect;
    entity->rampMidTick = stats.rampMidTick;
    entity->rampFullTick = stats.rampFullTick;
    entity->rampStartFraction = stats.rampStartFraction;
    entity->rampMidFraction = stats.rampMidFraction;
    entity->rampStage4Tick = stats.rampStage4Tick;
    entity->rampStage4Fraction = stats.rampStage4Fraction;
    entity->rampGracePeriodTicks = stats.rampGracePeriodTicks;
    entity->maxSplitTargets = stats.maxSplitTargets;
    entity->splitTargetsFullDamage = stats.splitTargetsFullDamage;
    entity->splashRadius = stats.splashRadius;
    entity->shieldHp = stats.shieldHp;
    entity->chargeThreshold = stats.chargeThreshold;
    entity->chargeMultiplier = stats.chargeMultiplier;
    entity->chargeIsSticky = stats.chargeIsSticky;
    entity->enrageMaxHp = stats.enrageMaxHp;
    entity->enrageHealPerHit = stats.enrageHealPerHit;
    entity->parryIntervalTicks = stats.parryIntervalTicks;
    entity->hookRange = stats.hookRange;
    entity->startsInvisible = stats.startsInvisible;
    entity->revealTicksAfterAttack = stats.revealTicksAfterAttack;
    entity->periodicEffect = stats.periodicEffect;
    entity->periodicIntervalTicks = stats.periodicIntervalTicks;
    entity->periodicTicksUntilNext = stats.periodicIntervalTicks; // first fire after one full interval
    entity->auraRadius = stats.auraRadius;
    entity->auraMaxTargets = stats.auraMaxTargets;
    entity->auraEveryNAttacks = stats.auraEveryNAttacks;
    entity->auraBuffMultiplier = stats.auraBuffMultiplier;
    entity->auraBuffDurationTicks = stats.auraBuffDurationTicks;
    entity->healAllyAmount = stats.healAllyAmount;
    entity->dieAfterFirstHit = stats.dieAfterFirstHit;
    entity->chargeGrantsInvulnerability = stats.chargeGrantsInvulnerability;
    entity->resetCooldownOnFreeze = stats.resetCooldownOnFreeze;
    entity->recoilDistance = stats.recoilDistance;
    entity->minAttackRange = stats.minAttackRange;
    entity->sightRange = stats.sightRange;
    entity->transformAtHpFraction = stats.transformAtHpFraction;
    entity->transformCheckMaxHp = stats.hp;
    entity->transformLifetimeTicks = stats.transformLifetimeTicks;
    entity->transformBecomesStationary = stats.transformBecomesStationary;
    entity->transformKillsSelf = stats.transformKillsSelf;
    entity->transformDeathEffect = stats.transformDeathEffect;
    entity->jumpMinRange = stats.jumpMinRange;
    entity->jumpMaxRange = stats.jumpMaxRange;
    entity->jumpDamageMultiplier = stats.jumpDamageMultiplier;
    entity->jumpSplashRadius = stats.jumpSplashRadius;
    entity->lineSplash = stats.lineSplash;
    entity->lineSplashRange = stats.lineSplashRange;
    entity->rangeFalloff = stats.rangeFalloff;
    entity->rangeFalloffMinFraction = stats.rangeFalloffMinFraction;
    entity->rangeBandMinDist = stats.rangeBandMinDist;
    entity->rangeBandMaxDist = stats.rangeBandMaxDist;
    entity->rangeBandDamageMultiplier = stats.rangeBandDamageMultiplier;
    entity->isChampion = stats.isChampion;
    entity->isHero = stats.isHero;
    entity->abilityElixirCost = stats.abilityElixirCost;
    entity->abilityCooldownTicks = stats.abilityCooldownTicks;
    entity->abilityEffect = stats.abilityEffect;
    entity->abilityUsesRemaining = stats.abilityUsesLimit;
    // A post-spawn ability lockout (Hero Mega Minion) seeds the cooldown
    // directly.
    if (stats.initialAbilityCooldownTicks > 0) entity->abilityCooldownRemaining = stats.initialAbilityCooldownTicks;
    entity->soulCollectionRadius = stats.soulCollectionRadius;
    entity->maxSouls = stats.maxSouls;
    entity->hitSpeedRampMidTick = stats.hitSpeedRampMidTick;
    entity->hitSpeedRampFullTick = stats.hitSpeedRampFullTick;
    entity->hitSpeedRampMidFraction = stats.hitSpeedRampMidFraction;
    entity->hitSpeedRampFullFraction = stats.hitSpeedRampFullFraction;
    entity->burstEveryNAttacks = stats.burstEveryNAttacks;
    entity->burstDamageMultiplier = stats.burstDamageMultiplier;
    entity->onDamageTakenEffect = stats.onDamageTaken;
    entity->healOnHitAmount = stats.healOnHitAmount;
    entity->healOnHitMaxHp = stats.healOnHitMaxHp;
    entity->onHitSpawnEffect = stats.onHitSpawnEffect;
    entity->onHitPullRadius = stats.onHitPullRadius;
    entity->onHitPullDistance = stats.onHitPullDistance;
    entity->onHitPullDamage = stats.onHitPullDamage;
    entity->selfHasteDurationTicks = stats.selfHasteDurationTicks;
    entity->selfHasteCooldownMultiplier = stats.selfHasteCooldownMultiplier;
    // Runtime state (ability cooldown, souls, invisibility, hit-speed
    // multiplier) is not copied from stats.
    applyOnHit(entity, stats);
    if (stats.initialCooldownTicks > 0) entity->seedCooldown(stats.initialCooldownTicks);
    if (stats.passiveDamageReduction < 1.0f) entity->applyCurse(stats.passiveDamageReduction, 999999);
}

// Flying implies ignoring the river; ignoresRiver alone also covers ground
// units that jump it (Hog Rider).
inline bool shouldIgnoreRiver(const CardStats& stats) {
    return stats.ignoresRiver || stats.isFlying;
}

// The footprint isValidPlacement keeps clear of existing buildings: Building's
// collision radius for a building, the implicit troop radius otherwise. The
// real game forbids placing on a building outright.
inline float placementRadius(Archetype archetype) {
    return archetype == Archetype::DefensiveBuilding
        ? Building::COLLISION_RADIUS
        : Entity::IMPLICIT_TROOP_RADIUS;
}

// A card's one-time deploy burst (e.g. Electro Wizard's zap) at the placement
// point, once per card rather than per squad member. Most cards set none.
inline void spawnDeployEffect(const CardStats& stats, float x, float y, int team, Board& board) {
    if (stats.spawnEffectRadius <= 0.0f) return;
    auto effect = std::make_shared<AreaSpell>(
        board.allocateId(), x, y, team, stats.spawnEffectRadius, stats.spawnEffectDamage,
        0, stats.symbol, stats.spawnEffectOnHit);
    effect->name = stats.name;
    effect->cardId = stats.id;
    board.addEntity(effect);
}

inline void spawnMeleeSquad(const CardStats& stats, float x, float y, int team, Board& board) {
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<MeleeTroop>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        if (shouldIgnoreRiver(stats)) troop->setIgnoresRiver(true);
        applyCardMetadata(troop, stats);
        board.addEntity(troop);
    }
    spawnDeployEffect(stats, x, y, team, board);
}

inline void spawnRangedSquad(const CardStats& stats, float x, float y, int team, Board& board) {
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<RangedTroop>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        if (shouldIgnoreRiver(stats)) troop->setIgnoresRiver(true);
        troop->boomerang = stats.boomerang;
        troop->boomerangReturnDelayTicks = stats.boomerangReturnDelayTicks;
        applyCardMetadata(troop, stats);
        board.addEntity(troop);
    }
    spawnDeployEffect(stats, x, y, team, board);
}

inline void spawnMeleeBuildingTargeter(const CardStats& stats, float x, float y, int team, Board& board) {
    // Loops over spawnOffsets for multi-unit children such as Golemites.
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<BuildingTargeter>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        if (shouldIgnoreRiver(stats)) troop->setIgnoresRiver(true);
        applyCardMetadata(troop, stats);
        board.addEntity(troop);
    }
    spawnDeployEffect(stats, x, y, team, board);
}

inline void spawnRangedBuildingTargeter(const CardStats& stats, float x, float y, int team, Board& board) {
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<RangedBuildingTargeter>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        if (shouldIgnoreRiver(stats)) troop->setIgnoresRiver(true);
        applyCardMetadata(troop, stats);
        board.addEntity(troop);
    }
    spawnDeployEffect(stats, x, y, team, board);
}

inline void spawnDefensiveBuilding(const CardStats& stats, float x, float y, int team, Board& board) {
    auto building = std::make_shared<Building>(
        board.allocateId(), x, y, stats.hp, team, stats.symbol,
        stats.attackRange, stats.damage, stats.attackCooldown);
    applyCardMetadata(building, stats);
    board.addEntity(building);
}

inline void spawnSpell(const CardStats& stats, float x, float y, int team, Board& board) {
    auto spell = std::make_shared<AreaSpell>(
        board.allocateId(), x, y, team, stats.spellRadius, stats.damage, stats.spellDelayTicks, stats.symbol,
        stats.spellOnHit, stats.spellGroundOnly, stats.spellRemainingHits, stats.spellTickInterval,
        stats.spellBuffsAllies, stats.spellBuffMultiplier, stats.spellBuffDurationTicks, stats.spellKnockback,
        stats.spellSpawnEffect, stats.spellClonesAllies, stats.spellTargetTopHpCount,
        stats.spellTieredDamage, stats.spellTierSingleDamage, stats.spellTierFewDamage, stats.spellTierManyDamage);
    spell->name = stats.name;
    spell->cardId = stats.id;
    // Rolling spells (The Log, Barbarian Barrel) are configured after
    // construction: configureRoll latches the roll origin from the spawn
    // position, so it must run before the first update.
    if (stats.spellRollRange > 0.0f) {
        spell->configureRoll(stats.spellRollRange, stats.spellRollWidth,
                             stats.spellRollSpeed, stats.spellRollKnockback);
    }
    board.addEntity(spell);
}

inline void spawn(const CardStats& stats, float x, float y, int team, Board& board) {
    switch (stats.archetype) {
        case Archetype::MeleeSquad:             spawnMeleeSquad(stats, x, y, team, board); break;
        case Archetype::RangedSquad:             spawnRangedSquad(stats, x, y, team, board); break;
        case Archetype::MeleeBuildingTargeter:   spawnMeleeBuildingTargeter(stats, x, y, team, board); break;
        case Archetype::RangedBuildingTargeter:  spawnRangedBuildingTargeter(stats, x, y, team, board); break;
        case Archetype::DefensiveBuilding:       spawnDefensiveBuilding(stats, x, y, team, board); break;
        case Archetype::Spell:                   spawnSpell(stats, x, y, team, board); break;
        default: throw std::logic_error("CardFactories::spawn: unhandled archetype");
    }
    // Compound cards (Goblin Giant, Ram Rider, Goblin Machine, Goblin Gang,
    // Rascals): the second, independently targeting unit spawns at the same
    // point.
    if (stats.secondaryUnit) {
        spawn(*stats.secondaryUnit, x, y, team, board);
    }
}

} // namespace CardFactories
