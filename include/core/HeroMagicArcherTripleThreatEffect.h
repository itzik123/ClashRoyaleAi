#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"

// Hero Magic Archer's "Triple Threat": dashes back toward his own side
// (same team-relative direction idiom as BossBanditGetawayGrenadeEffect's
// own teleport), spawns a stationary decoy at the position he just left,
// and gains a temporary multi-shot window -- see
// CombatEntity::temporarySplitTargetsTicksRemaining/maxSplitTargets for how
// that's modeled (approximated via the existing Electro Wizard split-target
// machinery, not genuinely independent projectiles). The real ~1s wind-up
// delay before the dash is collapsed into an instant resolution, same
// documented simplification as GoldenKnightDashEffect/HeroGiantHurlEffect.
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
