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

protected:
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        std::shared_ptr<Entity> closestBuilding = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (entity->team != this->team && entity->isAlive() && entity->isTargetable()) {
                auto buildingPtr = std::dynamic_pointer_cast<Building>(entity);
                if (buildingPtr) {
                    float dist = position.distanceTo(entity->position);
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestBuilding = entity;
                    }
                }
            }
        }
        return closestBuilding;
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