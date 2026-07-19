#pragma once
#include "OnHitEffect.h"
#include "CombatEntity.h"

// Mother Witch's curse: a temporary damage-taken debuff, applied the same
// way FreezeOnHit applies its slow. The real card also transforms a
// cursed enemy into a friendly Goblin if it dies while cursed -- not
// modeled (no on-cursed-death hook exists).
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
