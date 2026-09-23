#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// A damage-over-time mark on whatever gets hit (Dart Goblin / Firecracker
// Evolutions); see CombatEntity::applyDot. Split out for the same include-cycle
// reason as FreezeOnHit.
class PoisonOnHit : public IOnHitEffect {
    int damagePerTick;
    int totalTicks;
    int tickInterval;

public:
    PoisonOnHit(int damagePerTick, int totalTicks, int tickInterval)
        : damagePerTick(damagePerTick), totalTicks(totalTicks), tickInterval(tickInterval) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        target->applyDot(damagePerTick, totalTicks, tickInterval);
    }
};
