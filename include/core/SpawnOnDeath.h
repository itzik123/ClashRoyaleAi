#pragma once
#include "DeathEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Spawns a child troop (e.g. Golem's Golemites) where this entity died, on its
// own team.
class SpawnOnDeath : public IDeathEffect {
    CardStats childStats;

public:
    explicit SpawnOnDeath(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, team, board);
    }
};
