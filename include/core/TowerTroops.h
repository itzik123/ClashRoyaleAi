#pragma once
#include "CardStats.h"
#include "RoyalChefBuffEffect.h"
#include <memory>

// The 4 selectable Tower Troops (real game: pick one, it replaces BOTH of
// your Princess Towers for the whole match) -- see GameManager's
// addTower overload and reset(). None reproduces this engine's original
// hardcoded stats exactly (2534hp/90dmg/8-tick cooldown/7.5 range) and
// stays the default, so no existing deck/test/training run is affected
// unless a tower troop is explicitly selected -- the sourced "Tower
// Princess" troop specifically is a real, separate, stronger option
// (~26% more hp), not a silent buff to today's baseline.
enum class TowerTroopType {
    None,
    TowerPrincess,
    Cannoneer,
    DaggerDuchess,
    RoyalChef
};

// Symbol stays 'P' for every variant (see GameManager::addTower's own
// comment on why) -- only name/combat stats vary. CardStats's core combat
// fields are set directly (public members) rather than through troop()/
// building() -- those helpers are private to CardRegistry, and a Tower
// isn't a CardRegistry-registered card anyway (built directly by
// GameManager, same as King/Princess Towers always have been).
inline CardStats towerTroopStats(TowerTroopType type) {
    CardStats stats;
    stats.spawnOffsets = { {0.0f, 0.0f} };

    switch (type) {
        case TowerTroopType::TowerPrincess:
            // Confirmed (level 11): 3204hp/153dmg/437dps. This engine's
            // integer-tick cooldown can't hit 437 dps exactly at 153
            // damage (would need a fractional-tick hit speed) -- 4 ticks
            // (0.4s, ~382dps) is the closest whole-tick approximation.
            stats.hp = 3204; stats.attackRange = 7.5f; stats.damage = 153; stats.attackCooldown = 4;
            break;
        case TowerTroopType::Cannoneer:
            // Confirmed (level 11): 3052hp/109dmg/0.8s hit speed, no splash.
            stats.hp = 3052; stats.attackRange = 7.5f; stats.damage = 109; stats.attackCooldown = 8;
            break;
        case TowerTroopType::DaggerDuchess:
            // Confirmed (Liquipedia): 2298hp, 7.5 range, 0.5s hit speed,
            // 89 dmg/dagger, 8 max daggers. "Throws low damage daggers
            // until fully charged, then throws all of her daggers at one
            // target for high damage" maps onto the burst-on-Nth-attack
            // primitive (added for Musketeer/Dagger Duchess-shape
            // evolutions) rather than a bespoke magazine-and-reload
            // system -- exact damage-per-dagger-vs-burst split isn't
            // fully resolved by sourced data (secondary sources
            // disagreed), so the burst multiplier (8x) is an engine-
            // internal approximation, not an independently sourced number.
            stats.hp = 2298; stats.attackRange = 7.5f; stats.damage = 89; stats.attackCooldown = 5;
            stats.withBurstAttack(8, 8.0f);
            break;
        case TowerTroopType::RoyalChef:
            // Own combat stats not confirmed by sourced data -- reuses
            // Tower Princess's combat numbers as a reasonable baseline
            // (every Tower Troop still has SOME basic attack; only his
            // support ability is distinctly documented). "Feeds" an ally
            // every ~21-35s (28s used here, the midpoint) -- see
            // RoyalChefBuffEffect for the "+1 level" approximation.
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
