#pragma once
#include "Entity.h"

// Marker base for anything a card can place directly on the board --
// troops, buildings, and spells (via CombatEntity and AreaSpell). Formalizes
// the one real split among Entity's descendants: "a player chose to put
// this here" versus Projectile, which only ever appears as a side effect of
// another entity's attack and is never itself selected from a hand.
class CardEntity : public Entity {
public:
    using Entity::Entity;
};
