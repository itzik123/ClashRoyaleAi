#pragma once
#include "Building.h"
#include "Projectile.h"

class Tower : public Building {
public:
    Tower(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol)
        : Building(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown, -1) {
        targetsAir = true; // every tower defends against air
        // Fallback for towers built outside GameManager (tests): sight never
        // below attack range. GameManager sets the sourced values afterwards.
        sightRange = attackRange;
    }

    float getCollisionRadius() const override {
        return (symbol == 'R') ? 2.0f : 1.5f;
    }

    // Target-selection footprint, larger than the collision radius and fitted
    // to real play: a Hog with a Cannon 7-8 tiles off-lane keeps walking at the
    // Princess Tower and diverts only at 6. Against a Cannon's 1.0 that needs
    // 2.92 < r < 3.65; 3.3 is mid-band. The King's is scaled by the same 2.0 /
    // 1.5 ratio as its collision radius.
    float getTargetingRadius() const override {
        return (symbol == 'R') ? 4.4f : 3.3f;
    }

    bool isTower() const override { return true; }

    // Its own override: Building's would slice a Tower to a plain Building,
    // losing isTower() and the tower fallback in findTarget. The copy carries
    // `awake`, so a snapshotted King stays dormant.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Tower>(*this);
    }

    // --- activation ---
    // The King Tower starts dormant: it cannot acquire or fire until woken,
    // permanently, by taking any damage or by losing a friendly Princess Tower.
    // Only the King is built asleep (GameManager::addTower).
    bool isAwake() const { return awake; }
    void wake() { awake = true; }
    void sleep() { awake = false; }

    void update(Board& board) override {
        if (!awake) {
            // Trigger 1: any damage. Latched on hp < maxHp rather than hooked
            // into one of the several damage entry points; the latch only sets,
            // so a heal cannot re-sleep it.
            if (hp < maxHp) awake = true;

            // Trigger 2: a friendly Princess Tower destroyed. The count is
            // recorded on the first update rather than compared with 2, since a
            // King on a bare board (every unit test) starts with none.
            int living = 0;
            for (const auto& e : board.getEntities()) {
                if (e->isAlive() && e->isTower() && e->team == team && e->symbol != 'R') ++living;
            }
            if (initialFriendlyPrincesses < 0) initialFriendlyPrincesses = living;
            else if (living < initialFriendlyPrincesses) awake = true;
        }
        Building::update(board);
    }

protected:
    // The single choke point for dormancy: with no acquisition nothing is ever
    // attacked. The tower stays targetable, which is what lets trigger 1 fire.
    std::shared_ptr<Entity> findTarget(Board& board) const override {
        if (!awake) return nullptr;
        return CombatEntity::findTarget(board);
    }

    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        // lineSplash/lineSplashRange forwarded: a Tower Troop can carry them.
        auto arrow = std::make_shared<Projectile>(
            board.allocateId(), position.x, position.y, team, target, 2.0f, getCurrentDamage(), onHitEffects,
            false, 0, id, cardId, splashRadius, lineSplash, lineSplashRange);
        board.addEntity(arrow);
    }

private:
    // Princess Towers; only the King is put to sleep.
    bool awake = true;
    // -1 until the first update() records the starting count.
    int initialFriendlyPrincesses = -1;
};