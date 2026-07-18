#pragma once
#include "Entity.h"

// Composable extra behavior triggered when a CombatEntity dies (e.g. Golem
// spawning two Golemites), decoupled from *what* gets spawned or *how*.
// Lets death-triggered cards be expressed as "a normal troop plus an
// effect" instead of a bespoke entity subclass, mirroring IOnHitEffect.
//
// Takes position/team by value rather than the dying entity itself: nothing
// a death effect does needs the entity's other state, and concrete effects
// (e.g. SpawnOnDeath, in core/ since it needs CardFactories) shouldn't need
// to know the caller's concrete type either.
class IDeathEffect {
public:
    virtual ~IDeathEffect() = default;
    virtual void apply(Board& board, const Vector2D& position, int team) const = 0;
};
