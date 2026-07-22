#pragma once
#include "OnDamageTakenEffect.h"
#include "CombatEntity.h"

// Minion Horde Evolution's "Dark Guard": taking damage from a troop or
// spell turns the hit member invisible (untargetable) for a fixed
// duration -- reuses temporaryInvisibilityTicksRemaining, the same field
// Archer Queen's Cloaking Cape/Boss Bandit's Getaway Grenade set, just
// triggered from taking damage instead of an activated ability. No hit-
// speed change accompanies this (temporaryHitSpeedMultiplier is left at
// its default 1.0), unlike those two.
class DarkGuardOnDamageEffect : public IOnDamageTakenEffect {
    int durationTicks;

public:
    explicit DarkGuardOnDamageEffect(int durationTicks) : durationTicks(durationTicks) {}

    void apply(CombatEntity& self) const override {
        self.temporaryInvisibilityTicksRemaining = durationTicks;
    }
};
