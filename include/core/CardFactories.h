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

inline void spawnMeleeSquad(const CardStats& stats, float x, float y, int team, Board& board) {
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<MeleeTroop>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        applyOnHit(troop, stats);
        board.addEntity(troop);
    }
}

inline void spawnRangedSquad(const CardStats& stats, float x, float y, int team, Board& board) {
    for (const auto& offset : stats.spawnOffsets) {
        auto troop = std::make_shared<RangedTroop>(
            board.allocateId(), x + offset.x, y + offset.y, stats.hp, team,
            stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
        applyOnHit(troop, stats);
        board.addEntity(troop);
    }
}

inline void spawnMeleeBuildingTargeter(const CardStats& stats, float x, float y, int team, Board& board) {
    auto troop = std::make_shared<BuildingTargeter>(
        board.allocateId(), x, y, stats.hp, team,
        stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
    if (stats.ignoresRiver) troop->setIgnoresRiver(true);
    applyOnHit(troop, stats);
    board.addEntity(troop);
}

inline void spawnRangedBuildingTargeter(const CardStats& stats, float x, float y, int team, Board& board) {
    auto troop = std::make_shared<RangedBuildingTargeter>(
        board.allocateId(), x, y, stats.hp, team,
        stats.speed, stats.attackRange, stats.damage, stats.attackCooldown, stats.symbol);
    if (stats.ignoresRiver) troop->setIgnoresRiver(true);
    applyOnHit(troop, stats);
    board.addEntity(troop);
}

inline void spawnDefensiveBuilding(const CardStats& stats, float x, float y, int team, Board& board) {
    auto building = std::make_shared<Building>(
        board.allocateId(), x, y, stats.hp, team, stats.symbol,
        stats.attackRange, stats.damage, stats.attackCooldown);
    applyOnHit(building, stats);
    board.addEntity(building);
}

inline void spawnSpell(const CardStats& stats, float x, float y, int team, Board& board) {
    board.addEntity(std::make_shared<AreaSpell>(
        board.allocateId(), x, y, team, stats.spellRadius, stats.damage, stats.spellDelayTicks, stats.symbol));
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
