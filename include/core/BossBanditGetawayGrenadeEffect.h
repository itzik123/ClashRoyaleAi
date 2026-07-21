#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Boss Bandit's "Getaway Grenade": brief invisibility (see
// CombatEntity::temporaryInvisibilityTicksRemaining -- hitSpeedMultiplier
// left at 1.0, this ability doesn't also grant a haste buff) plus a
// teleport backward, toward her own side. The real card sequences these
// (invisible for 1s, THEN teleports), but this engine applies both at
// once -- the end state (briefly untargetable, now further back) is the
// same, and this engine has no sub-tick animation timing to sequence them
// within anyway. Clamped to the board via Board::clampToBoard, the same
// bounds/river rule normal movement already respects, since this bypasses
// the usual move-then-clamp pipeline.
class BossBanditGetawayGrenadeEffect : public IAbilityEffect {
    int invisibilityTicks;
    float teleportDistance;

public:
    BossBanditGetawayGrenadeEffect(int invisibilityTicks, float teleportDistance)
        : invisibilityTicks(invisibilityTicks), teleportDistance(teleportDistance) {}

    void apply(Board& board, CombatEntity& self) const override {
        self.temporaryInvisibilityTicksRemaining = invisibilityTicks;

        Vector2D newPos = self.position;
        newPos.y += (self.team == 0) ? -teleportDistance : teleportDistance;
        self.position = board.clampToBoard(newPos, false);
    }
};
