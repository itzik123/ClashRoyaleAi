#pragma once
#include "OnDamageTakenEffect.h"
#include "CombatEntity.h"

// Minion Horde Evolution's "Dark Guard": a member hit by a troop or spell turns
// untargetable for a fixed duration. Unlike the Cloaking Cape, no hit-speed
// change.
class DarkGuardOnDamageEffect : public IOnDamageTakenEffect {
    int durationTicks;

public:
    explicit DarkGuardOnDamageEffect(int durationTicks) : durationTicks(durationTicks) {}

    void apply(CombatEntity& self) const override {
        self.temporaryInvisibilityTicksRemaining = durationTicks;
    }
};
