#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Monk's "Pensive Protection": reduced damage taken for a fixed duration, via
// applyCurse with a multiplier below 1. Not modelled: reflecting projectiles
// (takeDamage carries no attacker) and knockback immunity (pull/push are shared
// primitives with no opt-out).
class MonkDeflectEffect : public IAbilityEffect {
    float damageTakenMultiplier;
    int durationTicks;

public:
    MonkDeflectEffect(float damageTakenMultiplier, int durationTicks)
        : damageTakenMultiplier(damageTakenMultiplier), durationTicks(durationTicks) {}

    void apply(Board&, CombatEntity& self) const override {
        self.applyCurse(damageTakenMultiplier, durationTicks);
    }
};
