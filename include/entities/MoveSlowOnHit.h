#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// A movement-only slow on troops (Poison: enemy troops 15% slower while inside
// it). Unlike FreezeOnHit, attack speed is untouched and buildings are skipped:
// the real card slows troop movement only. Kept out of OnHitEffect.h for the
// same include reason as FreezeOnHit.
class MoveSlowOnHit : public IOnHitEffect {
    int ticks;
    float factor;

public:
    MoveSlowOnHit(int ticks, float factor) : ticks(ticks), factor(factor) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        if (target->isBuilding()) return;
        target->applyMoveSlow(ticks, factor);
    }
};
