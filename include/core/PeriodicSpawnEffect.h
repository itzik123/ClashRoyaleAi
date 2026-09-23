#pragma once
#include "PeriodicEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Spawns a child troop at this entity's position on each timer fire (Witch's
// Skeletons, Furnace's Fire Spirits).
class PeriodicSpawnEffect : public IPeriodicEffect {
    CardStats childStats;

public:
    explicit PeriodicSpawnEffect(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, team, board);
    }
};
