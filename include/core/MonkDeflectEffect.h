#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Monk's "Pensive Protection": reduces incoming damage for a fixed
// duration. Reuses CombatEntity::applyCurse -- despite the name, that
// method is just "multiply incoming damage taken by X for N ticks" with
// no directional assumption baked into takeDamage() itself, so a
// multiplier below 1.0 IS a damage-reduction shield, not just a debuff;
// no new field needed. The real ability's other two components --
// reflecting incoming projectiles back at their shooter, and knockback/
// Tornado-pull immunity -- aren't modeled: takeDamage() carries no
// attacker identity to reflect at (the same architectural gap already
// documented for Ronin's parry, which would need a signature change
// touching every takeDamage call site in the engine), and pullToward/
// pushAway are free functions shared by every knockback/hook/jump/pull
// source in this codebase, not something a single card can opt out of
// without touching that shared primitive.
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
