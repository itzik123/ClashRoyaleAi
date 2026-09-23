#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// A summon ability (Little Prince's Royal Rescue): spawns a fixed troop at the
// Champion's position.
class SpawnOnAbility : public IAbilityEffect {
    CardStats childStats;

public:
    explicit SpawnOnAbility(CardStats childStats) : childStats(std::move(childStats)) {}

    void apply(Board& board, CombatEntity& self) const override {
        CardFactories::spawn(childStats, self.position.x, self.position.y, self.team, board);
    }
};
