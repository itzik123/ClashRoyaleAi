#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Archer Queen's "Cloaking Cape": untargetable plus an attack-speed boost for a
// fixed duration. The real card's movement slowdown is not modelled (speed
// lives on Troop).
class ArcherQueenCloakEffect : public IAbilityEffect {
    int durationTicks;
    float hitSpeedMultiplier;

public:
    ArcherQueenCloakEffect(int durationTicks, float hitSpeedMultiplier)
        : durationTicks(durationTicks), hitSpeedMultiplier(hitSpeedMultiplier) {}

    void apply(Board&, CombatEntity& self) const override {
        self.temporaryInvisibilityTicksRemaining = durationTicks;
        self.temporaryHitSpeedMultiplier = hitSpeedMultiplier;
    }
};
