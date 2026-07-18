#pragma once
#include "Troop.h"
#include "Projectile.h"

class RangedTroop : public Troop {
public:
    // Boomerang (Executioner): the projectile hits its target once on
    // arrival, then again after boomerangReturnDelayTicks once it "returns",
    // instead of dying after a single hit. false (the default) is every
    // other ranged troop's normal single-hit projectile.
    bool boomerang = false;
    int boomerangReturnDelayTicks = 0;

    RangedTroop(int id, float x, float y, int hp, int team,
        float speed, float attackRange, int damage, int attackCooldown, char symbol = 'A')
        : Troop(id, x, y, hp, team, symbol, speed, attackRange, damage, attackCooldown) {}

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        // On-hit effects ride along with the shot and land when it does,
        // instead of applying instantly at the moment of firing.
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 1.5f, getCurrentDamage(), onHitEffects,
            boomerang, boomerangReturnDelayTicks, id, cardId);
        board.addEntity(arrow);
    }
};