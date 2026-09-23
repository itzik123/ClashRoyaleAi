#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// Mother Witch's curse: a temporary damage-taken debuff. The real card also
// turns a unit that dies cursed into a friendly Goblin; not modelled.
class CurseOnHit : public IOnHitEffect {
    float damageTakenMultiplier;
    int ticks;

public:
    CurseOnHit(float damageTakenMultiplier, int ticks)
        : damageTakenMultiplier(damageTakenMultiplier), ticks(ticks) {}

    void apply(std::shared_ptr<CombatEntity> target) const override {
        target->applyCurse(damageTakenMultiplier, ticks);
    }
};
