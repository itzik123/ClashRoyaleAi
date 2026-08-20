#pragma once
#include "Troop.h"
#include "Building.h"
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

    // Board::deepCopy -- id and hp preserved exactly, unlike clone() above.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<BuildingTargeter>(*this);
    }

protected:
    // Same two-tier sight-range/tower-fallback shape as
    // CombatEntity::findTarget's own (see its comment), just filtered to
    // buildings only -- a building-targeter never considers troops at
    // all. Since every enemy Tower is itself a Building, this is really
    // "closest non-tower defensive building within sightRange, else the
    // closest tower regardless of distance," matching a building-
    // targeter's actual real-game behavior: engage a visible defensive
    // building, otherwise beeline for the enemy tower. isBuilding()
    // (Entity.h) replaces this method's own dynamic_pointer_cast<Building>
    // that used to live here.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        std::shared_ptr<Entity> closestInSight = nullptr;
        float minSightDistance = std::numeric_limits<float>::max();
        std::shared_ptr<Entity> closestTower = nullptr;
        float minTowerDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (entity->team == this->team || !entity->isAlive() || !entity->isTargetable()) continue;
            if (!entity->isBuilding()) continue;
            float dist = position.distanceTo(entity->position);
            if (entity->isTower()) {
                if (dist < minTowerDistance) {
                    minTowerDistance = dist;
                    closestTower = entity;
                }
            // effectiveSightTo, not raw sightRange -- see CombatEntity's own
            // findTarget and the comment on effectiveSightTo for the measured
            // free-siege bug the mismatch caused.
            } else if (dist <= effectiveSightTo(entity) && dist < minSightDistance) {
                minSightDistance = dist;
                closestInSight = entity;
            }
        }
        return closestInSight ? closestInSight : closestTower;
    }

    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        int dealt = getCurrentDamage();
        target->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { id, team, cardId, target->id, target->cardId, target->team, dealt, board.currentTick });
        applyOnHitEffects(target);
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};