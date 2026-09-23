#pragma once
#include "CombatEntity.h"
#include "StatsEvents.h"

class Building : public CombatEntity {
protected:
    int maxHp;
    int lifetimeTicks;
    int ticksAlive = 0;

public:
    // Every Building has the same footprint; GameManager::isValidPlacement uses
    // this too.
    static constexpr float COLLISION_RADIUS = 1.0f;

    Building(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown, int lifetime = 300)
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown),
        maxHp(hp), lifetimeTicks(lifetime) {}

    // Full health as constructed: the ceiling for callers that write hp from
    // outside.
    int getMaxHp() const { return maxHp; }

    float getCollisionRadius() const override { return COLLISION_RADIUS; }

    bool isBuilding() const override { return true; }

    // For Board::deepCopy. Copies ticksAlive, so a snapshot keeps its place in
    // the decay schedule.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Building>(*this);
    }

    void update(Board& board) override {
        CombatEntity::update(board);
        if (lifetimeTicks > 0) {
            ticksAlive++;
            // Expiry is checked on the clock, independent of the decay
            // schedule: integer decay does not finish the building exactly at
            // its lifetime unless maxHp divides evenly.
            //
            // hp = 0 rather than takeDamage(): expiry is not damage, so no
            // shield absorbs it and no OnDamageTakenEffect fires.
            if (ticksAlive >= lifetimeTicks) {
                hp = 0;
                board.statsEvents.notifyAttributionCleared({ id });
            } else if (ticksAlive % 10 == 0) { // every 10 ticks
                int decayIntervals = lifetimeTicks / 10;
                if (decayIntervals <= 0) decayIntervals = 1; // lifetimes under 10 ticks
                int decayAmount = maxHp / decayIntervals;
                if (decayAmount <= 0) decayAmount = 1;
                takeDamage(decayAmount);
                // Decay has no attacker, so no DamageDealtEvent; but it clears
                // any earlier combat attribution, on every decay tick, so a
                // stale hit cannot be credited with a later decay death.
                board.statsEvents.notifyAttributionCleared({ id });
            }
        }
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        int dealt = getCurrentDamage();
        target->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { id, team, cardId, target->id, target->cardId, target->team, dealt, board.currentTick, target->isTower() });
        applyOnHitEffects(target);
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};