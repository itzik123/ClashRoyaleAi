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

    // Full health as CONSTRUCTED, which is the only ceiling a caller
    // writing hp from outside can clamp against. Tower::update already
    // latches `awake` on the same `hp < maxHp` invariant.
    int getMaxHp() const { return maxHp; }

    float getCollisionRadius() const override { return COLLISION_RADIUS; }

    bool isBuilding() const override { return true; }

    // Board::deepCopy. Carries ticksAlive/maxHp with it via the implicit copy
    // constructor, so a snapshotted building keeps its exact position in its
    // own decay schedule -- a copy that restarted at ticksAlive = 0 would give
    // every rollout a building that outlives the real one.
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<Building>(*this);
    }

    void update(Board& board) override {
        CombatEntity::update(board);
        if (lifetimeTicks > 0) {
            ticksAlive++;
            // EXPIRY, checked before the decay schedule below and independent
            // of it. The decay amount is maxHp / (lifetimeTicks / 10) in
            // INTEGER arithmetic, so unless maxHp divides exactly the schedule
            // never quite finishes the building off and it outlives its own
            // lifetime by however long the truncated remainder takes -- a
            // Cannon (824 hp / 300 ticks) decayed 27/s, sat on 14 hp at 30.0s
            // and only died at 31.0s. Nothing anywhere consulted the clock; the
            // building simply died whenever subtraction happened to reach zero.
            //
            // It read as correct because the one test covering it used
            // hp = 3000 with lifetime 300, and 3000 / 30 = 100 exactly. That
            // test is named "fully decays to 0 exactly at its configured
            // lifetime" -- the contract was already written down, and only a
            // divisible hp made it look true.
            //
            // hp = 0 rather than takeDamage(): expiry is not damage. A shield
            // (Cannon Cart) must not absorb it, a parry must not negate it, and
            // no OnDamageTakenEffect should fire for a clock running out.
            if (ticksAlive >= lifetimeTicks) {
                hp = 0;
                board.statsEvents.notifyAttributionCleared({ id });
            } else if (ticksAlive % 10 == 0) { // Decay every 1 second (10 ticks)
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