#pragma once
#include "Troop.h"
#include "Building.h"
#include "Projectile.h"

class RangedBuildingTargeter : public Troop {
public:
    RangedBuildingTargeter(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

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
        auto proj = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, damage);
        board.addEntity(proj);
    }
};
