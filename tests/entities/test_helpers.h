#pragma once
// Minimal concrete stand-ins used only by the entity test suite.
// The production hierarchy (Entity, CombatEntity, Troop, Building) is abstract
// at several levels, so tests need small concrete subclasses to exercise the
// shared base-class behavior in isolation from any single card's specifics.
#include "Board.h"
#include "Entity.h"
#include "CombatEntity.h"
#include <memory>

// Adds an entity to the board and makes it immediately visible to
// findTarget()/collision queries, mirroring what GameManager::step() does
// once per tick via commitPendingEntities().
inline void spawn(Board& board, std::shared_ptr<Entity> entity) {
    board.addEntity(entity);
    board.commitPendingEntities();
}

// Bare Entity with a no-op update(), for testing Entity's own state machine
// (hp/isAlive/name) without any combat or movement behavior attached. Valid
// as a findTarget()/performAttack() target (both stay Entity-typed) -- use
// StationaryCombatant below only where a test specifically needs freeze or
// on-hit effects, which really are CombatEntity-only concepts.
class DummyEntity : public Entity {
public:
    bool targetable = true;

    DummyEntity(int id, float x, float y, int hp, int team, char symbol = '?')
        : Entity(id, x, y, hp, team, symbol) {}

    void update(Board&) override {}

    bool isTargetable() const override { return targetable; }
};

// CombatEntity that never moves and never decays, isolating the
// find-target / attack-range / cooldown / freeze logic that lives in
// CombatEntity::update() from Troop's movement and Building's decay.
class StationaryCombatant : public CombatEntity {
public:
    int attackCount = 0;
    int lastTargetId = -1;

    StationaryCombatant(int id, float x, float y, int hp, int team,
        float attackRange, int damage, int attackCooldown, char symbol = '?')
        : CombatEntity(id, x, y, hp, team, symbol, attackRange, damage, attackCooldown) {}

protected:
    void performAttack(Board&, std::shared_ptr<Entity> target) override {
        attackCount++;
        lastTargetId = target->id;
        target->takeDamage(getCurrentDamage()); // respects ramp/split, like every production leaf class
        applyOnHitEffects(target); // direct-damage style: effects land immediately, like MeleeTroop
    }
};

// Records its last apply() call instead of doing anything real, so tests can
// assert *that* and *with what* a death effect fired without needing a real
// spawn (SpawnOnDeath) behind it. apply() is const on the interface, hence
// mutable here -- recording is the entire point of this double.
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
