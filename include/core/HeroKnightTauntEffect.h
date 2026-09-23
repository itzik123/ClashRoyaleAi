#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Knight's "Triumphant Taunt": a shield that expires after the duration,
// and every enemy within tauntRadius is forced to attack him for as long.
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
