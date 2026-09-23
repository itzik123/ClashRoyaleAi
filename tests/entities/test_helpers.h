#pragma once
// Minimal concrete stand-ins for the entity tests: the production hierarchy is
// abstract at several levels.
#include "Board.h"
#include "Entity.h"
#include "CombatEntity.h"
#include "AbilityEffect.h"
#include "CardStats.h"
#include <memory>

// Adds an entity and commits it immediately, as GameManager::step() does once
// per tick.
inline void spawn(Board& board, std::shared_ptr<Entity> entity) {
    board.addEntity(entity);
    board.commitPendingEntities();
}

// Advances `entity` past its deploy time, so a test about what a card does need
// not restate the delay. Only entities spawned through CardFactories have one;
// directly constructed test entities never do.
inline void advancePastDeploy(const std::shared_ptr<Entity>& entity, Board& board) {
    for (int i = 0; i < DEPLOY_TIME_TICKS; ++i) entity->update(board);
}

// A bare Entity with a no-op update(), for Entity's own state; a valid target,
// since targeting is Entity-typed. Use StationaryCombatant only where freeze or
// on-hit effects are needed.
class DummyEntity : public Entity {
public:
    bool targetable = true;

    DummyEntity(int id, float x, float y, int hp, int team, char symbol = '?')
        : Entity(id, x, y, hp, team, symbol) {}

    void update(Board&) override {}

    bool isTargetable() const override { return targetable; }
};

// A CombatEntity that never moves or decays, isolating CombatEntity::update()'s
// targeting, range, cooldown and freeze logic.
class StationaryCombatant : public CombatEntity {
public:
    int attackCount = 0;
    int lastTargetId = -1;

    StationaryCombatant(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol = '?')
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown) {}

protected:
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        attackCount++;
        lastTargetId = target->id;
        int dealt = getCurrentDamage(); // respects ramp and split, like the production leaf classes
        target->takeDamage(dealt);
        applyOnHitEffects(target); // direct-damage: effects land immediately, as in MeleeTroop
        applySplashDamage(board, target->position, splashRadius, target->id, id, team, cardId, dealt);
    }
};

// Records its last apply() call, so tests can check that and with what a death
// effect fired. `mutable` because apply() is const.
class RecordingDeathEffect : public IDeathEffect {
public:
    mutable bool applied = false;
    mutable Vector2D lastPosition{};
    mutable int lastTeam = -1;

    void apply(Board&, const Vector2D& position, int team) const override {
        applied = true;
        lastPosition = position;
        lastTeam = team;
    }
};

// Counts every call: a periodic effect fires repeatedly.
class RecordingPeriodicEffect : public IPeriodicEffect {
public:
    mutable int applyCount = 0;
    mutable Vector2D lastPosition{};
    mutable int lastTeam = -1;

    void apply(Board&, const Vector2D& position, int team) const override {
        applyCount++;
        lastPosition = position;
        lastTeam = team;
    }
};

// Counts IAbilityEffect calls.
class RecordingAbilityEffect : public IAbilityEffect {
public:
    mutable int applyCount = 0;

    void apply(Board&, CombatEntity&) const override {
        applyCount++;
    }
};
