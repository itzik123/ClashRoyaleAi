#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "SpawnOnDeathForEnemyTeam.h"
#include "CompositeDeathEffect.h"
#include <vector>
#include <memory>

// Mother Witch's curse: the damage-taken debuff, plus a one-time arming so that
// a unit dying while cursed spawns a Cursed Hog for the opposite team. Composes
// with the target's existing deathEffect rather than replacing it.
class CursedHogOnHit : public IOnHitEffect {
    float damageTakenMultiplier;
    int ticks;
    CardStats cursedHogStats;

public:
    CursedHogOnHit(float damageTakenMultiplier, int ticks, CardStats cursedHogStats)
        : damageTakenMultiplier(damageTakenMultiplier), ticks(ticks), cursedHogStats(std::move(cursedHogStats)) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        // The curse refreshes on every hit.
        target->applyCurse(damageTakenMultiplier, ticks);

        // The hog spawn is armed once; re-arming nested a new composite per hit
        // and spawned one hog per hit. See
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
