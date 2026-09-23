#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// Kept out of OnHitEffect.h: calling applyFreeze needs CombatEntity's full
// definition, and CombatEntity.h includes OnHitEffect.h.
class FreezeOnHit : public IOnHitEffect {
    int ticks;
    float slowFactor;

public:
    FreezeOnHit(int ticks, float slowFactor) : ticks(ticks), slowFactor(slowFactor) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        target->applyFreeze(ticks, slowFactor);
    }
};
