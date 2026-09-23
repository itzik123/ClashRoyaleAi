#pragma once
#include "CardStats.h"
#include "RoyalChefBuffEffect.h"
#include <memory>

// The four selectable Tower Troops; one replaces both Princess Towers for the
// match (GameManager::addTower, reset). None is the default and keeps the
// engine's original stats (2534 hp / 90 dmg / 8-tick cooldown / 7.5 range).
enum class TowerTroopType {
    None,
    TowerPrincess,
    Cannoneer,
    DaggerDuchess,
    RoyalChef
};

// Symbol stays 'P' for every variant (see GameManager::addTower). Fields are
// set directly because towers are built by GameManager, not registered cards.
inline CardStats towerTroopStats(TowerTroopType type) {
    CardStats stats;
    stats.spawnOffsets = { {0.0f, 0.0f} };
    // The Princess Tower's sourced sight range, applied to every variant.
    stats.sightRange = 7.5f;
    // Must be set explicitly: applyCardMetadata runs after the Tower
    // constructor and overwrites targetsAir with this struct's value.
    stats.targetsAir = true;

    switch (type) {
        case TowerTroopType::TowerPrincess:
            // Level 11: 3204 hp / 153 dmg / 437 dps. A whole-tick cooldown
            // cannot hit 437 dps at 153 damage; 4 ticks (~382 dps) is closest.
            stats.hp = 3204; stats.attackRange = 7.5f; stats.damage = 153; stats.attackCooldown = 4;
            break;
        case TowerTroopType::Cannoneer:
            // Level 11: 3052 hp / 109 dmg / 0.8 s, no splash.
            stats.hp = 3052; stats.attackRange = 7.5f; stats.damage = 109; stats.attackCooldown = 8;
            break;
        case TowerTroopType::DaggerDuchess:
            // Liquipedia: 2298 hp, range 7.5, 0.5 s, 89 per dagger, 8 daggers.
            // Modelled as a burst on the 8th attack; the 8x multiplier is an
            // approximation, since sources disagree on the split.
            stats.hp = 2298; stats.attackRange = 7.5f; stats.damage = 89; stats.attackCooldown = 5;
            stats.withBurstAttack(8, 8.0f);
            break;
        case TowerTroopType::RoyalChef:
            // Combat stats unsourced, so Tower Princess's are reused. Feeds an
            // ally every ~21-35 s (28 s here); see RoyalChefBuffEffect.
            stats.hp = 3204; stats.attackRange = 7.5f; stats.damage = 153; stats.attackCooldown = 4;
            stats.withPeriodicEffect(280, std::make_shared<RoyalChefBuffEffect>(6.0f, 1.10f));
            break;
        case TowerTroopType::None:
        default:
            stats.hp = 2534; stats.attackRange = 7.5f; stats.damage = 90; stats.attackCooldown = 8;
            break;
    }
    return stats;
}
