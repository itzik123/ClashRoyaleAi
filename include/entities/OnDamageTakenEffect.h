#pragma once
#include <memory>

class CombatEntity;

// Fires from CombatEntity::takeDamage for damage that gets past shield, parry
// and curse and reaches hp; the receiving-end counterpart of IOnHitEffect.
class IOnDamageTakenEffect {
public:
    virtual ~IOnDamageTakenEffect() = default;
    virtual void apply(CombatEntity& self) const = 0;
};
