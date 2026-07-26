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
// within anyway. Clamped to the board via Board::clampToBoard with
// ignoresRiver=true -- a "getaway" retreating from enemy territory back to
// her own side routinely needs to cross back over the river, and clamping
// her to the near bank instead would frequently defeat the entire point of
// the escape. Matches Boss Bandit's own CardStats::withIgnoresRiver
// (same river-crossing approximation as the regular Bandit), but this
// bypasses the usual move-then-clamp pipeline (CombatEntity has no
// riverIgnores field of its own to read -- that lives on Troop), so it's
// hardcoded here rather than derived from `self`.
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
