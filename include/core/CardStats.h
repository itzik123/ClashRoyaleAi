#pragma once
#include "Entity.h"
#include "OnHitEffect.h"
#include <memory>
#include <string>
#include <vector>

// The handful of ways a card actually gets built onto the board. Adding a
// new card of an existing shape is a data row (see CardRegistry.h); adding a
// genuinely new mechanic is a new archetype + one new factory function in
// CardFactories.h.
enum class Archetype {
    MeleeSquad,             // one or more direct-damage troops (Knight, Goblins, Barbarians...)
    RangedSquad,             // one or more projectile troops (Musketeer, Archers, Spear Goblins...)
    MeleeBuildingTargeter,   // ignores troops, melee-hits buildings (Giant, Hog Rider, Golem)
    RangedBuildingTargeter,  // ignores troops, projectile-hits buildings (Royal Giant)
    DefensiveBuilding,       // stationary, direct damage (Cannon, Tesla, ...)
    Spell                    // one-shot area effect (Fireball, Zap, ...)
};

struct CardStats {
    int id = 0;
    std::string name;
    float cost = 0.0f;
    Archetype archetype = Archetype::MeleeSquad;

    // Combat stats, used by every archetype except Spell.
    int hp = 0;
    float speed = 0.0f;
    float attackRange = 0.0f;
    int damage = 0;
    int attackCooldown = 0;
    char symbol = '?';
    bool ignoresRiver = false;
    bool isFlying = false;
    bool targetsAir = false;

    // Relative spawn positions for each unit in the card. A single-unit
    // card is just one offset of {0, 0} -- "squad of one".
    std::vector<Vector2D> spawnOffsets{ Vector2D{ 0.0f, 0.0f } };

    // Optional extra behavior applied on every successful attack landed by
    // every unit this card spawns (e.g. Ice Wizard/Ice Golem's freeze).
    std::shared_ptr<IOnHitEffect> onHit;

    // Spell archetype only.
    float spellRadius = 0.0f;
    int spellDelayTicks = 0;

    // Small fluent setters so CardRegistry's data table can stay one card
    // per line/two, instead of spelling out every field for every card.
    CardStats& withOffsets(std::vector<Vector2D> offsets) {
        spawnOffsets = std::move(offsets);
        return *this;
    }
    CardStats& withIgnoresRiver(bool value = true) {
        ignoresRiver = value;
        return *this;
    }
    CardStats& withFlying(bool value = true) {
        isFlying = value;
        return *this;
    }
    CardStats& withTargetsAir(bool value = true) {
        targetsAir = value;
        return *this;
    }
    CardStats& withOnHit(std::shared_ptr<IOnHitEffect> effect) {
        onHit = std::move(effect);
        return *this;
    }
};
