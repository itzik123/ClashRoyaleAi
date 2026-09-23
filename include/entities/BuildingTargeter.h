#pragma once
#include "Troop.h"
#include "Building.h"
#include "LanePath.h"
#include "StatsEvents.h"

class BuildingTargeter : public Troop {
public:
    BuildingTargeter(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

    std::shared_ptr<Entity> clone(int newId) const override {
        auto copy = std::make_shared<BuildingTargeter>(*this);
        copy->id = newId;
        copy->hp = 1; // Clone: full damage, 1 hp
        return copy;
    }

    // For Board::deepCopy: id and hp preserved, unlike clone().
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<BuildingTargeter>(*this);
    }

protected:
    // Considers buildings only (towers included): the nearest building in
    // sight, otherwise the lane objective, otherwise the nearest tower.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        std::shared_ptr<Entity> closestInSight = nullptr;
        float minSightDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Entity> closestTower = nullptr;
        float minTowerDistance = std::numeric_limits<float>::max();

        const float myRadius = ownEffectiveRadius();   // loop invariant
        for (const auto& entity : board.getEntities()) {
            if (entity->team == this->team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (!entity->isBuilding()) continue;
            float dist = position.distanceTo(entity->position);

            // The nearest tower is tracked only for the out-of-sight fallback.
            if (entity->isTower() && dist < minTowerDistance) {
                minTowerDistance = dist;
                closestTower = entity;
            }

            // The nearest building in sight wins, and towers compete on equal
            // terms, as in the real game. Ranked by footprint distance
            // (Entity::getTargetingRadius); the sight gate uses
            // effectiveSightWith on centre distance, as in
            // CombatEntity::findTarget.
            const float rank = dist - entity->getTargetingRadius();
            if (rank <= effectiveSightWith(myRadius, *entity) && rank < minSightDistance) {
                minSightDistance = rank;
                closestInSight = entity;
            }
        }
        if (closestInSight) return closestInSight;

        // The same lane rule as CombatEntity::findTarget, so a
        // building-targeter and its escort walk to the same tower.
        auto laneTarget = LanePath::laneObjective(board, team, position);
        if (laneTarget && laneTarget->isBuilding()) return laneTarget;
        return closestTower;
    }

    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        int dealt = getCurrentDamage();
        target->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { id, team, cardId, target->id, target->cardId, target->team, dealt, board.currentTick, target->isTower() });
        applyOnHitEffects(target);
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};