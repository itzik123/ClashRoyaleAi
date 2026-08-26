#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "SpawnOnDeathForEnemyTeam.h"
#include "CompositeDeathEffect.h"
#include <vector>
#include <memory>

// Mother Witch's curse: applies the damage-taken debuff (same as
// entities/CurseOnHit) AND arms a one-time spawn -- if the cursed unit
// dies while still cursed, a Cursed Hog spawns fighting for Mother
// Witch's side (the team opposite whoever just died), via
// SpawnOnDeathForEnemyTeam. Lives in core/ (not entities/, unlike plain
// CurseOnHit) because it needs CardStats/CardFactories to know what to
// spawn, same reasoning as every other core/ effect implementing an
// entities/ interface.
//
// Composes with the target's own existing deathEffect (if any) via
// CompositeDeathEffect instead of overwriting it -- cursing a unit that
// already has its own on-death behavior (e.g. a Golemite mid-split-chain)
// shouldn't silently cancel that behavior just because it also got cursed.
class CursedHogOnHit : public IOnHitEffect {
    float damageTakenMultiplier;
    int ticks;
    CardStats cursedHogStats;

public:
    CursedHogOnHit(float damageTakenMultiplier, int ticks, CardStats cursedHogStats)
        : damageTakenMultiplier(damageTakenMultiplier), ticks(ticks), cursedHogStats(std::move(cursedHogStats)) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        // The curse itself REFRESHES on every hit -- that is the real card.
        target->applyCurse(damageTakenMultiplier, ticks);

        // The hog spawn is armed ONCE. Re-arming wrapped the previous chain in
        // a fresh CompositeDeathEffect every hit, so a unit that took N hits
        // died into N hogs (and carried N levels of nesting). Measured before
        // the latch: three applications produced three hogs. See
        // CombatEntity::curseDeathSpawnAttached.
        if (target->curseDeathSpawnAttached) return;
        target->curseDeathSpawnAttached = true;

        auto spawnEffect = std::make_shared<SpawnOnDeathForEnemyTeam>(cursedHogStats);
        if (target->deathEffect) {
            target->deathEffect = std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{ target->deathEffect, spawnEffect });
        } else {
            target->deathEffect = spawnEffect;
        }
    }
};
