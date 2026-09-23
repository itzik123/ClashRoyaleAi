#pragma once
#include "Entity.h"

// Behaviour fired on a repeating timer while a CombatEntity lives; the timer
// counterpart of IDeathEffect. Effects that spawn live in core/ because they
// need CardFactories.
class IPeriodicEffect {
public:
    virtual ~IPeriodicEffect() = default;
    virtual void apply(Board& board, const Vector2D& position, int team) const = 0;
};
