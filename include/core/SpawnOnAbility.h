#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// Generic "summon a fixed troop" activated ability (Little Prince's Royal
// Rescue, summoning the Guardienne) -- same reuse-CardFactories::spawn
// shape as SpawnOnDeath/PeriodicSpawnEffect, just fired by an explicit
// player action instead of a death or a repeating timer. Spawns at the
// Champion's own current position/team.
class SpawnOnAbility : public IAbilityEffect {
    CardStats childStats;

public:
    explicit SpawnOnAbility(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, CombatEntity& self) const override {
        CardFactories::spawn(childStats, self.position.x, self.position.y, self.team, board);
    }
};
