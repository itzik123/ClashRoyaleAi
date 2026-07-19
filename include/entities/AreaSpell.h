#pragma once
#include "CardEntity.h"
#include "CombatEntity.h"
#include "OnHitEffect.h"
#include "PeriodicEffect.h"
#include "Board.h"
#include "StatsEvents.h"
#include <memory>

class AreaSpell : public CardEntity {
private:
    float radius;
    int damage;
    int delayTicks;
    std::shared_ptr<IOnHitEffect> onHit;
    // Most spells hit Air & Ground alike (Fireball, Zap, Poison, Rocket,
    // Lightning...); a handful (The Log, Barbarian Barrel) are ground-only
    // control spells that roll along the arena floor. Defaults to false
    // (hits everyone) to match the majority, with ground-only spells opting
    // in via CardStats::withGroundOnly.
    bool groundOnly;

    // Multi-tick spells (Poison's 8 ticks over 8s, Arrows' 3 rapid volleys):
    // remainingHits counts down one application at a time, re-evaluating who
    // is currently in radius on *each* application -- exactly like the real
    // game, a unit can walk out of a Poison cloud partway through and stop
    // taking damage. remainingHits == 1 (the default) is every other
    // spell's normal single-shot case; tickInterval is only relevant when
    // remainingHits > 1, and doubles delayTicks as the gap between hits.
    int remainingHits;
    int tickInterval;

    // Rage: instead of damaging enemies, buffs allies in radius. Flips the
    // team filter (== instead of !=) and applies a damage buff instead of
    // takeDamage()/onHit -- `damage` goes unused in this mode (should be 0
    // at the call site). false (the default) is every other spell here.
    bool buffsAllies;
    float buffMultiplier;
    int buffDurationTicks;

    // Knockback/pull (Fireball, Rocket, Giant Snowball push away; Tornado
    // pulls toward center): applied to everyone hit, alongside damage.
    // Positive pushes away from the spell's position, negative pulls
    // toward it -- see Entity.h's pushAway/pullToward. 0.0f (the default)
    // is every other spell here.
    float knockback;

    // Spell-spawns-troops (Goblin Barrel, Royal Delivery, Graveyard):
    // fires once per application (same cadence as remainingHits/
    // tickInterval above -- a single application for the first two,
    // repeated for Graveyard's rain-of-skeletons), spawning at this
    // spell's own position/team. Reuses IPeriodicEffect/
    // PeriodicSpawnEffect from core/ -- same interface as
    // CombatEntity's own periodic spawns, just fired from a spell
    // instead of a tick timer. nullptr (the default) is every spell
    // that doesn't spawn anything.
    std::shared_ptr<IPeriodicEffect> spawnOnDetonate;

    // Clone: duplicates every ally troop in radius instead of damaging
    // enemies (same ally-team filter as buffsAllies). Each clone gets a
    // fresh board id and 1 hp (see Entity::clone) but keeps every other
    // configured field of the original. false (the default) is every
    // other spell here.
    bool clonesAllies;

public:
    AreaSpell(int id, float x, float y, int team, float radius, int damage, int delayTicks, char symbol = '*',
        std::shared_ptr<IOnHitEffect> onHit = nullptr, bool groundOnly = false,
        int remainingHits = 1, int tickInterval = 0,
        bool buffsAllies = false, float buffMultiplier = 1.0f, int buffDurationTicks = 0,
        float knockback = 0.0f, std::shared_ptr<IPeriodicEffect> spawnOnDetonate = nullptr,
        bool clonesAllies = false)
        : CardEntity(id, x, y, 1, team, symbol), radius(radius), damage(damage), delayTicks(delayTicks),
        onHit(std::move(onHit)), groundOnly(groundOnly), remainingHits(remainingHits), tickInterval(tickInterval),
        buffsAllies(buffsAllies), buffMultiplier(buffMultiplier), buffDurationTicks(buffDurationTicks),
        knockback(knockback), spawnOnDetonate(std::move(spawnOnDetonate)), clonesAllies(clonesAllies) {}

    bool isTargetable() const override { return false; }

    void update(Board& board) override {
        if (delayTicks > 0) {
            delayTicks--;
            return;
        }

        bool alliesOnly = buffsAllies || clonesAllies;
        // Cloning while iterating board.getEntities() would add to the
        // same vector mid-loop -- collect targets first, clone after.
        std::vector<std::shared_ptr<Entity>> toClone;

        for (const auto& entity : board.getEntities()) {
            bool teamMatches = alliesOnly ? (entity->team == this->team) : (entity->team != this->team);
            if (entity->isAlive() && entity->isTargetable() && teamMatches && entity->id != this->id
                && (!groundOnly || !entity->isFlying)) {
                float dist = position.distanceTo(entity->position);
                if (dist <= radius) {
                    if (buffsAllies) {
                        if (auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity)) {
                            combatTarget->applyBuff(buffMultiplier, buffDurationTicks);
                        }
                        continue;
                    }
                    if (clonesAllies) {
                        toClone.push_back(entity);
                        continue;
                    }
                    entity->takeDamage(damage);
                    if (knockback > 0.0f) {
                        pushAway(*entity, position, knockback);
                    } else if (knockback < 0.0f) {
                        pullToward(*entity, position, -knockback);
                    }
                    // The spell entity itself is the "attacker" -- it never
                    // has a separate caster once cast (the troop/tower that
                    // played the card is already gone by the time this
                    // fires, for spawn-effect zaps like Electro Wizard's).
                    board.statsEvents.notifyDamageDealt(
                        { id, team, cardId, entity->id, entity->cardId, entity->team, damage, board.currentTick });
                    // Same surgical cast as CombatEntity::applyOnHitEffects and
                    // Projectile's arrival handler -- on-hit effects only ever
                    // mean something against a CombatEntity, so this is the one
                    // place AreaSpell needs to know that, not its own type.
                    if (onHit) {
                        auto combatTarget = std::dynamic_pointer_cast<CombatEntity>(entity);
                        if (combatTarget) onHit->apply(combatTarget);
                    }
                }
            }
        }

        for (const auto& original : toClone) {
            auto copy = original->clone(board.allocateId());
            if (copy) board.addEntity(copy);
        }

        if (spawnOnDetonate) spawnOnDetonate->apply(board, position, team);

        remainingHits--;
        if (remainingHits > 0) {
            delayTicks = tickInterval; // wait out the gap, then apply again
        } else {
            hp = 0;
        }
    }
};