#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Archer Queen's "Cloaking Cape": untargetable + a big attack-speed boost
// for a fixed duration -- see CombatEntity::temporaryInvisibilityTicksRemaining/
// temporaryHitSpeedMultiplier. The real card's accompanying movement-speed
// drop isn't modeled -- speed lives on Troop, a layer above CombatEntity,
// same documented gap as Rage's own movement-speed component elsewhere in
// this codebase.
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
