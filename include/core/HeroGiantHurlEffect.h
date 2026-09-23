#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "TargetingHelpers.h"
#include "Board.h"

// Hero Giant's "Heroic Hurl": throws the highest-HP enemy troop within
// grabRange to the mirrored lane and stuns it on landing. The real ~1 s wind-up
// resolves instantly, since nothing else acts in between.
class HeroGiantHurlEffect : public IAbilityEffect {
    float grabRange;
    int stunTicks;

public:
    HeroGiantHurlEffect(float grabRange, int stunTicks)
        : grabRange(grabRange), stunTicks(stunTicks) {}

    void apply(Board& board, CombatEntity& self) const override {
        auto victim = findHpExtremeEnemy(board, self.position, grabRange, self.team, /*wantHighestHp=*/true);
        if (!victim) return;
        mirrorToOppositeLane(*victim, board.getWidth());
        if (auto ce = std::dynamic_pointer_cast<CombatEntity>(victim)) {
            ce->applyFreeze(stunTicks, 0.0f);
        }
    }
};
