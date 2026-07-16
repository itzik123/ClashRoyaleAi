#pragma once
#include "Entity.h"
#include <memory>

// Composable extra behavior triggered whenever a CombatEntity lands an
// attack, decoupled from *how* the attack itself deals damage (melee vs
// projectile). Lets cards like Ice Wizard/Ice Golem be expressed as "a
// normal troop plus an effect" instead of a bespoke entity subclass.
class IOnHitEffect {
public:
    virtual ~IOnHitEffect() = default;
    virtual void apply(std::shared_ptr<Entity> target) const = 0;
};

class FreezeOnHit : public IOnHitEffect {
    int ticks;
    float slowFactor;

public:
    FreezeOnHit(int ticks, float slowFactor) : ticks(ticks), slowFactor(slowFactor) {}

    void apply(std::shared_ptr<Entity> target) const override {
        target->applyFreeze(ticks, slowFactor);
    }
};
