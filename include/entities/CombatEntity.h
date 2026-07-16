#pragma once
#include "Entity.h"
#include "Board.h"
#include <memory>
#include <limits>

class CombatEntity : public Entity {
protected:
    float attackRange;
    int damage;
    int attackCooldown;
    float currentCooldown;

public:
    CombatEntity(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown)
        : Entity(id, x, y, hp, team, symbol),
        attackRange(attackRange), damage(damage),
        attackCooldown(attackCooldown), currentCooldown(0.0f) {}

    void update(Board& board) override {
        if (freezeTicks > 0) {
            freezeTicks--;
            if (currentCooldown > 0.0f) {
                currentCooldown -= freezeSlow;
            }
        } else {
            if (currentCooldown > 0.0f) {
                currentCooldown -= 1.0f;
            }
        }
        
        if (currentCooldown < 0.0f) currentCooldown = 0.0f;

        auto target = findTarget(board);
        if (target) {
            float dist = position.distanceTo(target->position);
            
            float targetRadius = target->getCollisionRadius();
            if (targetRadius <= 0.0f) targetRadius = 0.4f; // Implicit radius for troops
            
            float myRadius = this->getCollisionRadius();
            if (myRadius <= 0.0f) myRadius = 0.4f; // Implicit radius for troops
            
            float effectiveAttackRange = attackRange + myRadius + targetRadius;

            if (dist <= effectiveAttackRange) {
                if (currentCooldown == 0.0f) {
                    performAttack(board, target);
                    currentCooldown = static_cast<float>(attackCooldown);
                }
            } else {
                moveTowards(board, target->position);
            }
        }
        
        clampPosition();
    }

protected:
    virtual std::shared_ptr<Entity> findTarget(Board& board) const {
        std::shared_ptr<Entity> closestTarget = nullptr;
        float minDistance = std::numeric_limits<float>::max();

        for (const auto& entity : board.getEntities()) {
            if (entity->team != this->team && entity->isAlive() && entity->isTargetable() && entity->id != this->id) {
                float dist = position.distanceTo(entity->position);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestTarget = entity;
                }
            }
        }
        return closestTarget;
    }

    virtual void performAttack(Board& board, std::shared_ptr<Entity> target) = 0;

    virtual void moveTowards(Board& board, const Vector2D& dest) {
        // Default: stationary entities don't move
    }

    virtual void clampPosition() {
        // Default: no-op
    }
};
