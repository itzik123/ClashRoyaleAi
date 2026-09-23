#pragma once
#include "Entity.h"

// Behaviour triggered when a CombatEntity dies (e.g. Golem spawning Golemites),
// so such cards are a normal troop plus an effect rather than a subclass. Takes
// position and team, not the entity: no death effect needs more.
class IDeathEffect {
public:
    virtual ~IDeathEffect() = default;
    virtual void apply(Board& board, const Vector2D& position, int team) const = 0;
};
