#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "Board.h"

// Mighty Miner's "Explosive Escape": teleports him to the horizontally-
// mirrored position across the board's center vertical line -- same Y, only
// X flips (a lane swap, not a top/bottom flip) -- via
// newX = (board.getWidth() - 1) - x, the same board-mirroring convention
// already used by ClashEnv::extractObservationForTeam and
// GameManager::reset()'s King Tower comment, just applied to X here instead
// of Y. Leaves a bomb behind at his ORIGINAL (pre-teleport) position: a
// plain AreaSpell spawned directly (not through CardFactories::spawnSpell,
// since this fires mid-battle from an ability, not a card play), reusing
// AreaSpell's own delayTicks/knockback support instead of a new delayed-
// damage primitive. groundOnly stays false: hits ground and air alike, per
// research.
//
// bombRadius/bombKnockback below aren't part of the sourced research data
// (a "medium area" and "knocks back" with no exact published numbers) --
// reasonable engine-internal geometry constants, same caveat as
// CombatEntity::splashRadius/shieldHp elsewhere in this codebase.
class MightyMinerEscapeEffect : public IAbilityEffect {
    float bombRadius;
    int bombDamage;
    int bombDelayTicks;
    float bombKnockback;

public:
    MightyMinerEscapeEffect(float bombRadius, int bombDamage, int bombDelayTicks, float bombKnockback)
        : bombRadius(bombRadius), bombDamage(bombDamage),
          bombDelayTicks(bombDelayTicks), bombKnockback(bombKnockback) {}

    void apply(Board& board, CombatEntity& self) const override {
        Vector2D originalPosition = self.position;

        float mirroredX = static_cast<float>(board.getWidth() - 1) - self.position.x;
        self.position.x = mirroredX;
        // Y intentionally unchanged -- a lane swap, not a top/bottom flip.

        auto bomb = std::make_shared<AreaSpell>(
            board.allocateId(), originalPosition.x, originalPosition.y, self.team,
            bombRadius, bombDamage, bombDelayTicks, /*symbol=*/'*', /*onHit=*/nullptr,
            /*groundOnly=*/false, /*remainingHits=*/1, /*tickInterval=*/0,
            /*buffsAllies=*/false, /*buffMultiplier=*/1.0f, /*buffDurationTicks=*/0,
            /*knockback=*/bombKnockback);
        board.addEntity(bomb);
    }
};
