#pragma once
#include "DeathEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Spawns a data-driven child troop for the OPPOSING team when this entity
// dies (Mother Witch's Cursed Hog: a unit she cursed spawns a Hog fighting
// FOR Mother Witch, i.e. against the cursed unit's own team). Otherwise
// identical to SpawnOnDeath, which spawns on the dying entity's own team.
class SpawnOnDeathForEnemyTeam : public IDeathEffect {
    CardStats childStats;

public:
    explicit SpawnOnDeathForEnemyTeam(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, 1 - team, board);
    }
};
