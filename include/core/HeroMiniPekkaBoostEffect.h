#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Mini P.E.K.K.A's "Breakfast Boost" -- the real card fills a meter by
// attacking, up to +2 levels once fully charged; per this feature's own
// plan, simplified (no meter simulation) to a flat, one-time hp+damage
// boost the player triggers directly, gated to fire only once per
// deployment via the registration's own usesLimit=1 (see CardStats::
// withHeroAbility), not a repeating cooldown.
class HeroMiniPekkaBoostEffect : public IAbilityEffect {
    int hpBonus;
    float damageMultiplier;

public:
    HeroMiniPekkaBoostEffect(int hpBonus, float damageMultiplier)
        : hpBonus(hpBonus), damageMultiplier(damageMultiplier) {}

    void apply(Board&, CombatEntity& self) const override {
        self.hp += hpBonus;
        // 999999 ticks: same "effectively-infinite duration" idiom as
        // CardStats::withPassiveDamageReduction's own comment -- a
        // permanent-for-the-rest-of-this-deployment boost, not a real timed buff.
        self.applyBuff(damageMultiplier, 999999);
    }
};
