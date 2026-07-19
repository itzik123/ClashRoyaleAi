#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Spawns a data-driven child troop at this entity's own position every
// time its periodic timer fires (e.g. Witch's Skeletons, Furnace's Fire
// Spirits) -- same reasoning and reuse of CardFactories::spawn as
// SpawnOnDeath, just triggered on a repeating timer instead of once on
// death.
class PeriodicSpawnEffect : public IPeriodicEffect {
    CardStats childStats;

public:
    explicit PeriodicSpawnEffect(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, team, board);
    }
};
