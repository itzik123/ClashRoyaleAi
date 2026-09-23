#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Mini P.E.K.K.A's "Breakfast Boost". The real card charges a meter by
// attacking; here it is a flat one-time hp and damage boost, once per deploy
// via usesLimit=1.
class HeroMiniPekkaBoostEffect : public IAbilityEffect {
    int hpBonus;
    float damageMultiplier;

public:
    HeroMiniPekkaBoostEffect(int hpBonus, float damageMultiplier)
        : hpBonus(hpBonus), damageMultiplier(damageMultiplier) {}

    void apply(Board&, CombatEntity& self) const override {
        self.hp += hpBonus;
        // 999999 ticks: lasts the rest of the deployment.
        self.applyBuff(damageMultiplier, 999999);
    }
};
