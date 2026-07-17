#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// Separated from OnHitEffect.h because calling target->applyFreeze(...)
// needs CombatEntity's complete definition -- and CombatEntity.h itself
// includes OnHitEffect.h for the IOnHitEffect interface. Splitting the
// interface from this concrete effect breaks that cycle.
class FreezeOnHit : public IOnHitEffect {
    int ticks;
    float slowFactor;

public:
    FreezeOnHit(int ticks, float slowFactor) : ticks(ticks), slowFactor(slowFactor) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        target->applyFreeze(ticks, slowFactor);
    }
};
