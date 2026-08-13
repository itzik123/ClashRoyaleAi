#pragma once
#include "Building.h"
#include "Projectile.h"

class Tower : public Building {
public:
    Tower(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol)
        : Building(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown, -1) {
        targetsAir = true; // every tower in Clash Royale defends against air
        // Default sightRange to attackRange -- never less than what it can
        // already hit, regardless of construction path. GameManager sets
        // the exact sourced values (King 7.0, Princess Tower variants 7.5
        // via towerTroopStats/applyCardMetadata) explicitly after this;
        // this is just a sane fallback for a Tower built any other way
        // (tests, etc).
        sightRange = attackRange;
    }

    float getCollisionRadius() const override {
        return (symbol == 'R') ? 2.0f : 1.5f;
    }

    bool isTower() const override { return true; }

    // Board::deepCopy. Its own override, not Building's inherited one, which
    // would slice a Tower down to a plain Building -- losing isTower() and
    // with it the findTarget fallback that makes towers always-visible
    // destinations, so troops in a rollout would wander instead of pushing.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Tower>(*this);
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 2.0f, getCurrentDamage(), onHitEffects,
            false, 0, id, cardId, splashRadius);
        board.addEntity(arrow);
    }
};