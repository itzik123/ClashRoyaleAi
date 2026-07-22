#pragma once
#include <memory>

class CombatEntity;

// Composable extra behavior triggered whenever a CombatEntity actually
// TAKES damage -- mirrors IOnHitEffect (which fires for the attacker that
// LANDS a hit), just from the receiving end instead. Fires from
// CombatEntity::takeDamage(), only for damage that survives shield/parry/
// curse absorption and actually reaches hp (matches takeDamage's own
// "only real damage matters" framing) -- e.g. an evolution that buffs
// itself when struck. Takes a plain reference, not a shared_ptr, since it
// always fires on `*this` from inside takeDamage() itself, never on some
// other entity found via a board scan.
class IOnDamageTakenEffect {
public:
    virtual ~IOnDamageTakenEffect() = default;
    virtual void apply(CombatEntity& self) const = 0;
};
