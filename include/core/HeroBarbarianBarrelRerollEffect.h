#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Tower.h"
#include "Board.h"
#include <cmath>

// Hero Barbarian Barrel's "Rowdy Reroll": the Barbarian re-enters his
// barrel and rolls forward again, dealing the same roll damage a second
// time along the path -- halved against any Crown Tower hit (same
// dynamic_cast<Tower*> idiom GoldenKnightDashEffect already uses). Reuses
// applyLineSplashDamage's own line-hit-test geometry (project onto the
// line, clamp to [0,range], measure perpendicular distance) inline rather
// than modifying that shared free function for one card's Tower-specific
// discount. Gated to fire ONCE per deployment via the registration's own
// usesLimit=1.
class HeroBarbarianBarrelRerollEffect : public IAbilityEffect {
    float rollDistance;
    float halfWidth;
    int rollDamage;

public:
    HeroBarbarianBarrelRerollEffect(float rollDistance, float halfWidth, int rollDamage)
        : rollDistance(rollDistance), halfWidth(halfWidth), rollDamage(rollDamage) {}

    void apply(Board& board, CombatEntity& self) const override {
        Vector2D oldPos = self.position;
        Vector2D newPos = oldPos;
        newPos.y += (self.team == 0) ? rollDistance : -rollDistance; // forward, toward the enemy half
        self.position = board.clampToBoard(newPos, false);

        float dx = newPos.x - oldPos.x;
        float dy = newPos.y - oldPos.y;
        float lineLen = std::sqrt(dx * dx + dy * dy);
        if (lineLen <= 0.01f) return;
        float ux = dx / lineLen, uy = dy / lineLen;

        for (const auto& entity : board.getEntities()) {
            if (entity->id == self.id) continue;
            if (entity->team == self.team || !entity->isAlive() || !entity->isTargetable()) continue;

            float proj = (entity->position.x - oldPos.x) * ux + (entity->position.y - oldPos.y) * uy;
            if (proj < 0.0f) proj = 0.0f;
            if (proj > lineLen) proj = lineLen;
            float closestX = oldPos.x + ux * proj;
            float closestY = oldPos.y + uy * proj;
            float ddx = entity->position.x - closestX;
            float ddy = entity->position.y - closestY;
            if (std::sqrt(ddx * ddx + ddy * ddy) > halfWidth) continue;

            bool isTower = dynamic_cast<Tower*>(entity.get()) != nullptr;
            int dealt = isTower ? rollDamage / 2 : rollDamage;
            entity->takeDamage(dealt);
            board.statsEvents.notifyDamageDealt(
                { self.id, self.team, self.cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
        }
    }
};
