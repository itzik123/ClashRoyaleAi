#pragma once
#include "CombatEntity.h"
#include "StatsEvents.h"

class Building : public CombatEntity {
protected:
    int maxHp;
    int lifetimeTicks;
    int ticksAlive = 0;

public:
    // Every Building, regardless of card, occupies the same fixed footprint.
    // Named so GameManager::isValidPlacement can require this same distance
    // at placement time instead of re-guessing it as a separate constant.
    static constexpr float COLLISION_RADIUS = 1.0f;

    Building(int id, float x, float y, int hp, int team, char symbol,
        float attackRange, int damage, int attackCooldown, int lifetime = 300)
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown),
        maxHp(hp), lifetimeTicks(lifetime) {}

    float getCollisionRadius() const override { return COLLISION_RADIUS; }

    bool isBuilding() const override { return true; }

    void update(Board& board) override {
        CombatEntity::update(board);
        if (lifetimeTicks > 0) {
            ticksAlive++;
            if (ticksAlive % 10 == 0) { // Decay every 1 second (10 ticks)
                int decayIntervals = lifetimeTicks / 10;
                if (decayIntervals <= 0) decayIntervals = 1; // avoid div-by-zero for lifetimes under 10 ticks
                int decayAmount = maxHp / decayIntervals;
                if (decayAmount <= 0) decayAmount = 1;
                takeDamage(decayAmount);
                // Not a DamageDealtEvent -- decay has no attacker, it's not a
                // combat event. But it DOES mean any earlier combat hit this
                // building took is no longer what's actually killing it, so
                // tell KillStatsCollector to drop that stale attribution
                // (every decay tick, not just the lethal one, since staying
                // silent on the non-lethal ticks would leave a stale entry
                // to wrongly resurface if this building dies on a later
                // decay tick without having been re-hit in between).
                board.statsEvents.notifyAttributionCleared({ id });
            }
        }
    }

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        int dealt = getCurrentDamage();
        target->takeDamage(dealt);
        board.statsEvents.notifyDamageDealt(
            { id, team, cardId, target->id, target->cardId, target->team, dealt, board.currentTick });
        applyOnHitEffects(target);
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};