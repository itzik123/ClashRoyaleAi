#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// Applies a poison-style damage-over-time mark to whatever gets hit (Dart
// Goblin/Firecracker Evolutions) -- see CombatEntity::applyDot. Split from
// OnHitEffect.h for the same reason as FreezeOnHit: calling
// target->applyDot(...) needs CombatEntity's complete definition.
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
