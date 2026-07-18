#pragma once
#include "DeathEffect.h"
#include "CardStats.h"
#include "CardFactories.h"

// Spawns a data-driven child troop (e.g. Golem's two Golemites) at the
// position of whatever just died carrying this effect, on the same team.
// Reuses CardFactories::spawn so "what shape of entity to create" stays
// defined in exactly one place instead of duplicated here.
class SpawnOnDeath : public IDeathEffect {
    CardStats childStats;

public:
    explicit SpawnOnDeath(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        CardFactories::spawn(childStats, position.x, position.y, team, board);
    }
};
