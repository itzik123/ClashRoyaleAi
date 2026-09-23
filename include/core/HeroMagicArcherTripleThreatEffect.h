#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// Hero Magic Archer's "Triple Threat": dashes back toward his own side, leaves
// a stationary decoy where he stood, and gains a temporary multi-shot window
// (approximated with the Electro Wizard split-target machinery). The wind-up
// resolves instantly.
class HeroMagicArcherTripleThreatEffect : public IAbilityEffect {
    float dashDistance;
    CardStats decoyStats;
    int splitWindowTicks;

public:
    HeroMagicArcherTripleThreatEffect(float dashDistance, CardStats decoyStats, int splitWindowTicks)
        : dashDistance(dashDistance), decoyStats(std::move(decoyStats)), splitWindowTicks(splitWindowTicks) {}

    void apply(Board& board, CombatEntity& self) const override {
        Vector2D oldPosition = self.position;

        Vector2D newPos = self.position;
        newPos.y += (self.team == 0) ? -dashDistance : dashDistance;
        self.position = board.clampToBoard(newPos, false);

        CardFactories::spawn(decoyStats, oldPosition.x, oldPosition.y, self.team, board);

        self.baseMaxSplitTargets = self.maxSplitTargets;
        self.maxSplitTargets = 3;
        self.temporarySplitTargetsTicksRemaining = splitWindowTicks;
    }
};
