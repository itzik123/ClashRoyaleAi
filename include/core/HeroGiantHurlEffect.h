#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "TargetingHelpers.h"
#include "Board.h"

// Hero Giant's "Heroic Hurl": grabs the highest-HP enemy troop within
// short range and throws it across the arena to the opposite lane
// (horizontally-mirrored X, same lane-swap convention as
// MightyMinerEscapeEffect's own teleport), stunning it on landing. The
// real card's ~1s wind-up delay before the grab is collapsed into an
// instant resolution -- same documented simplification as
// GoldenKnightDashEffect's own comment (nothing else acts in between, so a
// simulated delay would change nothing observable).
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
