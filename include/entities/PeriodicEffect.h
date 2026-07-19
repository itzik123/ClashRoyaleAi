#pragma once
#include "Entity.h"

// Composable behavior fired periodically while a CombatEntity is alive
// (e.g. Witch spawning Skeletons every few seconds), decoupled from *what*
// gets spawned -- same shape and reasoning as IDeathEffect, just triggered
// on a repeating timer instead of once on death. Concrete effects that
// need to spawn something live in core/ (e.g. PeriodicSpawnEffect, since
// it needs CardFactories), same as SpawnOnDeath.
class IPeriodicEffect {
public:
    virtual ~IPeriodicEffect() = default;
    virtual void apply(Board& board, const Vector2D& position, int team) const = 0;
};
