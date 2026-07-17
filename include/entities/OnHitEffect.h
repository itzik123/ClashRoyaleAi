#pragma once
#include <memory>

class CombatEntity;

// Composable extra behavior triggered whenever a CombatEntity lands an
// attack, decoupled from *how* the attack itself deals damage (melee vs
// projectile). Lets cards like Ice Wizard/Ice Golem be expressed as "a
// normal troop plus an effect" instead of a bespoke entity subclass.
//
// Pure interface only -- a forward declaration of CombatEntity is enough
// here since nothing in this file calls a method on it. Concrete effects
// that do (e.g. FreezeOnHit) live in their own headers that can safely
// include CombatEntity.h, which itself includes this file: CombatEntity.h
// only needs to know the *shape* of an effect, not any particular one.
class IOnHitEffect {
public:
    virtual ~IOnHitEffect() = default;
    virtual void apply(std::shared_ptr<CombatEntity> target) const = 0;
};
