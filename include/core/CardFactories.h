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

// One factory function per Archetype (see CardStats.h). Each knows how to
// build exactly one shape of entity/entities from data -- nothing here is
// specific to any single card.
namespace CardFactories {

inline void applyOnHit(const std::shared_ptr<CombatEntity>& entity, const CardStats& stats) {
    if (stats.onHit) {
        entity->addOnHitEffect(stats.onHit);
    }
}

// Every entity a card produces carries that card's display name (e.g. all
// three Barbarians are each named "Barbarians"), whatever on-hit/death
// effect the card carries, its ground/air properties, and its damage-ramp /
// target-split configuration (both no-ops unless a card opts in).
inline void applyCardMetadata(const std::shared_ptr<CombatEntity>& entity, const CardStats& stats) {
    entity->name = stats.name;
    entity->cardId = stats.id;
    entity->isFlying = stats.isFlying;
    entity->targetsAir = stats.targetsAir;
    entity->deathEffect = stats.deathEffect;
    entity->rampMidTick = stats.rampMidTick;
    entity->rampFullTick = stats.rampFullTick;
    entity->rampStartFraction = stats.rampStartFraction;
    entity->rampMidFraction = stats.rampMidFraction;
    entity->maxSplitTargets = stats.maxSplitTargets;
    entity->splashRadius = stats.splashRadius;
    applyOnHit(entity, stats);
}

// A flying card ignores the river as a consequence of being airborne, even
// if its data entry never explicitly set ignoresRiver -- the two flags stay
// independently settable (Hog Rider ignores the river via a move ability,
// not by flying) but flying always implies it.
inline bool shouldIgnoreRiver(const CardStats& stats) {
    return stats.ignoresRiver || stats.isFlying;
}

// The footprint GameManager::isValidPlacement should keep clear of an
// existing building, matched to what the archetype will actually spawn
// with: a DefensiveBuilding gets Building's own fixed collision radius,
// every troop-shaped archetype gets the same implicit radius Board uses
// once it's on the field (resolvePositionAgainstBuildings). Real Clash
// Royale forbids placing anything on top of a building outright; this is
// this engine's continuous-space approximation of that rule.
inline float placementRadius(Archetype archetype) {
    return archetype == Archetype::DefensiveBuilding
        ? Building::COLLISION_RADIUS
        : Entity::IMPLICIT_TROOP_RADIUS;
}

// Fires a card's one-time deploy burst (e.g. Electro Wizard's spawn zap) at
// the card's placement point -- once per card played, not once per squad
// member, so squads call this after their spawn loop rather than inside it.
// A no-op for the overwhelming majority of cards, which never set it.
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
    // Loops over spawnOffsets like the squad factories, even though every
    // currently-registered card here is a single unit (the default offset
    // is just {0,0}) -- needed for Golemites, which come in twos, without
    // this archetype needing its own bespoke multi-unit variant.
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
        stats.spellOnHit, stats.spellGroundOnly, stats.spellRemainingHits, stats.spellTickInterval);
    spell->name = stats.name;
    spell->cardId = stats.id;
    board.addEntity(spell);
}

inline void spawn(const CardStats& stats, float x, float y, int team, Board& board) {
    switch (stats.archetype) {
        case Archetype::MeleeSquad:             spawnMeleeSquad(stats, x, y, team, board); return;
        case Archetype::RangedSquad:             spawnRangedSquad(stats, x, y, team, board); return;
        case Archetype::MeleeBuildingTargeter:   spawnMeleeBuildingTargeter(stats, x, y, team, board); return;
        case Archetype::RangedBuildingTargeter:  spawnRangedBuildingTargeter(stats, x, y, team, board); return;
        case Archetype::DefensiveBuilding:       spawnDefensiveBuilding(stats, x, y, team, board); return;
        case Archetype::Spell:                   spawnSpell(stats, x, y, team, board); return;
    }
    throw std::logic_error("CardFactories::spawn: unhandled archetype");
}

} // namespace CardFactories
