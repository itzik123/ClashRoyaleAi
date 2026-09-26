#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Tower.h"
#include "Board.h"
#include <algorithm>
#include <cmath>

// Hero Barbarian Barrel's "Rowdy Reroll": the Barbarian rolls forward again,
// hitting every GROUND enemy the barrel's corridor touches (halved against
// Crown Towers), and heals for half the damage it dealt, up to `maxHp`. The
// corridor is the barrel's own: `halfWidth` either side of the line, tested
// against each target's surface, as AreaSpell's rolling sweep does. Once per
// deploy via usesLimit=1. The line test is inlined rather than adding a tower
// discount to applyLineSplashDamage.
class HeroBarbarianBarrelRerollEffect : public IAbilityEffect {
    float rollDistance;
    float halfWidth;
    int rollDamage;
    int maxHp;

public:
    HeroBarbarianBarrelRerollEffect(float rollDistance, float halfWidth, int rollDamage, int maxHp)
        : rollDistance(rollDistance), halfWidth(halfWidth), rollDamage(rollDamage), maxHp(maxHp) {}

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

        int totalDealt = 0;
        for (const auto& entity : board.getEntities()) {
            if (entity->id == self.id) continue;
            if (entity->team == self.team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (entity->isFlying) continue; // the barrel rolls along the ground

            float proj = (entity->position.x - oldPos.x) * ux + (entity->position.y - oldPos.y) * uy;
            if (proj < 0.0f) proj = 0.0f;
            if (proj > lineLen) proj = lineLen;
            float closestX = oldPos.x + ux * proj;
            float closestY = oldPos.y + uy * proj;
            float ddx = entity->position.x - closestX;
            float ddy = entity->position.y - closestY;
            if (std::sqrt(ddx * ddx + ddy * ddy) > halfWidth + CombatEntity::effectiveRadiusOf(*entity)) continue;

            bool isTower = dynamic_cast<Tower*>(entity.get()) != nullptr;
            int dealt = isTower ? rollDamage / 2 : rollDamage;
            entity->takeDamage(dealt);
            totalDealt += dealt;
            board.statsEvents.notifyDamageDealt(
                { self.id, self.team, self.cardId, entity->id, entity->cardId, entity->team, dealt, board.currentTick, entity->isTower() });
        }
        if (totalDealt > 0 && self.hp < maxHp) {
            self.hp = std::min(self.hp + totalDealt / 2, maxHp);
        }
    }
};
