#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Boss Bandit's "Getaway Grenade": brief invisibility and a teleport back
// toward her own side. The real card sequences them; here both apply at once.
// Clamped with ignoresRiver=true, since the retreat usually has to recross the
// river (her riverIgnores lives on Troop, not CombatEntity).
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
        self.position = board.clampToBoard(newPos, true);
    }
};
