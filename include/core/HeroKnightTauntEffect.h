#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Knight's "Triumphant Taunt": gains a shield (expiring after a fixed
// duration even if never fully depleted by damage -- see
// CombatEntity::shieldExpiresTicksRemaining) and forces every enemy within
// tauntRadius to attack him for the same duration -- see applyTauntNearby.
class HeroKnightTauntEffect : public IAbilityEffect {
    int shieldAmount;
    int durationTicks;
    float tauntRadius;

public:
    HeroKnightTauntEffect(int shieldAmount, int durationTicks, float tauntRadius)
        : shieldAmount(shieldAmount), durationTicks(durationTicks), tauntRadius(tauntRadius) {}

    void apply(Board& board, CombatEntity& self) const override {
        self.shieldHp += shieldAmount;
        self.shieldExpiresTicksRemaining = durationTicks;
        applyTauntNearby(board, self.position, tauntRadius, durationTicks, self.id, self.team);
    }
};
