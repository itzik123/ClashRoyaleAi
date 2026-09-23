#pragma once
#include "Entity.h"

// Marker base for what a card places directly (troops, buildings, spells), as
// opposed to Projectile, which only exists as a side effect of an attack.
class CardEntity : public Entity {
public:
    using Entity::Entity;
};
