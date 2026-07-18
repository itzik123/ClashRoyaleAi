#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include "Board.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "FreezeOnHit.h"
#include "SpawnOnDeath.h"

// External-facing shape is unchanged on purpose: GameManager, ClashEnv,
// GameLogger, TerminalRenderer and main.cpp all consume CardDefinition as
// before. Internally, spawnEntity is now always one of the small archetype
// factories in CardFactories.h bound to this card's CardStats, instead of a
// bespoke lambda per card.
struct CardDefinition {
    int id;
    std::string name;
    float cost;
    bool isSpell;
    // The footprint GameManager::isValidPlacement keeps clear of an existing
    // building -- see CardFactories::placementRadius. Unused for spells.
    float placementRadius;
    std::function<void(float x, float y, int team, Board& board)> spawnEntity;
};

class CardRegistry {
private:
    std::unordered_map<int, CardDefinition> cards;

    static CardStats troop(int id, std::string name, float cost, Archetype archetype,
        int hp, float speed, float attackRange, int damage, int attackCooldown, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = archetype;
        s.hp = hp; s.speed = speed; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown; s.symbol = symbol;
        return s;
    }

    static CardStats building(int id, std::string name, float cost,
        int hp, char symbol, float attackRange, int damage, int attackCooldown) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::DefensiveBuilding;
        s.hp = hp; s.symbol = symbol; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown;
        return s;
    }

    static CardStats spell(int id, std::string name, float cost,
        float radius, int damage, int delayTicks, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::Spell;
        s.spellRadius = radius; s.damage = damage; s.spellDelayTicks = delayTicks; s.symbol = symbol;
        return s;
    }

    // 4-wide, 0.6-spaced grid layout for Skeleton Army's 15 skeletons.
    static std::vector<Vector2D> skeletonArmyOffsets() {
        std::vector<Vector2D> offsets;
        for (int i = 0; i < 15; ++i) {
            float ox = (static_cast<float>(i % 4) - 1.5f) * 0.6f;
            float oy = (static_cast<float>(i / 4) - 1.0f) * 0.6f;
            offsets.push_back({ ox, oy });
        }
        return offsets;
    }

    // Golemite: only ever spawned by a Golem's death, never itself a
    // playable card (no id in the registry, hence never add()-ed).
    static CardStats golemiteStats() {
        return troop(-1, "Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 1039, 0.2f, 0.25f, 84, 25, 'q')
            .withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} });
    }

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.placementRadius = CardFactories::placementRadius(stats.archetype);
        def.spawnEntity = [stats](float x, float y, int team, Board& board) {
            CardFactories::spawn(stats, x, y, team, board);
        };
        cards[stats.id] = std::move(def);
    }

    CardRegistry() {
        // Stats below are synced to current level-11 tournament-standard game
        // data (cost/hp/damage/hit-speed/range/count). Speed categories
        // aren't part of that data set, so existing speed values are left
        // as-is. Inferno Tower's ramp, Electro Wizard's split hit,
        // Executioner's boomerang return, and Poison/Arrows' multi-tick
        // damage are all modeled where they occur below (see each card's own
        // comment) -- not collapsed to a flat number. On-death burst damage
        // (Giant Skeleton, Lumberjack, Golem, Ice Golem, Balloon) isn't
        // modeled since there's no death-trigger system in this engine yet.
        // Electro Wizard's spawn zap is the one on-spawn effect that *is*
        // modeled, via CardStats::withSpawnEffect (see below).

        // === Melee Troops ===
        add(troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, 0.5f, 1.2f, 202, 12, 'K'));

        add(troop(4, "Goblins", 2.0f, Archetype::MeleeSquad, 202, 1.0f, 0.5f, 120, 11, 'g')
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        add(troop(5, "Mini PEKKA", 4.0f, Archetype::MeleeSquad, 1390, 0.8f, 0.8f, 755, 16, 'M'));

        add(troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 670, 0.5f, 0.7f, 191, 13, 'B')
            .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        add(troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, 0.5f, 1.2f, 266, 15, 'V'));

        add(troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 's')
            .withOffsets(skeletonArmyOffsets()));

        add(troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, 0.4f, 1.2f, 816, 18, 'E'));

        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1920, 0.6f, 1.6f, 391, 14, 'p'));

        add(troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, 0.7f, 1.2f, 384, 14, 'e')
            .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }));

        add(troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, 0.8f, 0.7f, 256, 8, 'l'));

        add(troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, 1.0f, 0.5f, 81, 11, 'k')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }));

        add(troop(39, "Giant Skeleton", 6.0f, Archetype::MeleeSquad, 3361, 0.4f, 0.8f, 276, 13, 'J'));

        // Electro Wizard: real attack is an instant zap with no projectile
        // travel time, so this is MeleeSquad-shaped (direct damage) despite
        // its 5.0 range -- archetype names describe *how* damage lands in
        // this engine, not the card's real-world range category. It splits
        // to the 2 closest enemies at once (each takes damage/2), or full
        // damage if only one is in range -- see
        // CombatEntity::findSplitTargets/getCurrentDamage. `damage` here is
        // the full single-target figure (230); getCurrentDamage() halves it
        // when a second target is found. Its 0.5s full stun (per hit *and*
        // on the spawn zap) is a freeze with slowFactor 0, same pattern as
        // Ice Wizard's slow. The spawn zap itself (radius 3, 192 damage,
        // same stun) rides withSpawnEffect, firing once at deploy via a
        // delay-0 AreaSpell.
        add(troop(35, "Electro Wizard", 4.0f, Archetype::MeleeSquad, 714, 0.5f, 5.0f, 230, 18, 'z')
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))
            .withSpawnEffect(3.0f, 192, std::make_shared<FreezeOnHit>(5, 0.0f)));

        // First flying cards -- ground/air targeting mechanism (isFlying/
        // targetsAir) exists purely to support these. Minions/Minion Horde
        // are the same unit at two squad sizes/costs, direct-damage like the
        // game's other small swarms (Goblins, Skeletons); Mega Minion is a
        // single heavier melee flier. All three hit air and ground alike.
        add(troop(41, "Minions", 3.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 11, 'm')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withFlying().withTargetsAir());

        add(troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, 0.8f, 2.5f, 107, 11, 'h')
            .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                           {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
            .withFlying().withTargetsAir());

        add(troop(43, "Mega Minion", 3.0f, Archetype::MeleeSquad, 837, 0.5f, 1.6f, 311, 15, 'F')
            .withFlying().withTargetsAir());

        // === Ranged Troops ===
        // Target: Air & Ground -- every ranged troop below needs its own
        // confirmed Target field before getting .withTargetsAir() (default
        // is ground-only); don't assume "ranged" implies it.
        add(troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, 0.5f, 5.0f, 112, 9, 'A')
            .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} })
            .withTargetsAir());

        add(troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, 0.5f, 6.0f, 217, 10, 'U'));
        add(troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, 0.5f, 4.5f, 225, 18, 'b'));
        add(troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 755, 0.5f, 5.5f, 281, 14, 'W'));
        add(troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, 0.8f, 6.5f, 156, 8, 'd')
            .withTargetsAir());
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 2081, 0.4f, 4.0f, 289, 25, 'w'));

        add(troop(23, "Spear Goblins", 2.0f, Archetype::RangedSquad, 133, 1.0f, 5.0f, 81, 10, 'S')
            .withOffsets({ {0.0f, 0.0f}, {0.7f, 0.0f}, {-0.7f, 0.0f} }));

        // Ice Wizard: genuinely ranged -- fires a projectile that applies the
        // freeze on arrival. (Originally this was mislabeled: it inherited
        // from RangedTroop but overrode performAttack to hit instantly,
        // bypassing the projectile entirely. Fixed now that on-hit effects
        // can ride along with a projectile instead of firing at launch.)
        add(troop(34, "Ice Wizard", 3.0f, Archetype::RangedSquad, 688, 0.5f, 5.5f, 89, 17, 'i')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        // Executioner's axe hits on arrival, then again 1.5s (15 ticks)
        // later on its "return" -- the real axe pierces everyone along its
        // path both ways; this engine has no line-collision primitive, so
        // it hits just the original target twice instead (see
        // Projectile's returnsToSender).
        add(troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, 0.4f, 4.5f, 168, 9, 'x')
            .withTargetsAir()
            .withBoomerang(15));

        // Baby Dragon: real splash isn't modeled (no area-of-effect on troop
        // attacks in this engine, same simplification as Wizard/Bowler
        // above) -- the per-hit damage figure is used as-is.
        add(troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, 0.8f, 3.5f, 168, 15, 'y')
            .withFlying().withTargetsAir());

        // === Building Targeters ===
        add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 4090, 0.3f, 1.2f, 253, 15, 'G'));

        add(troop(15, "Hog Rider", 4.0f, Archetype::MeleeBuildingTargeter, 1697, 0.8f, 0.8f, 317, 16, 'H')
            .withIgnoresRiver());

        // Golem splits into two Golemites on death (its separate death-damage
        // splash isn't modeled, same as every other card's on-death damage).
        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 5120, 0.2f, 0.75f, 312, 25, 'L')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(golemiteStats())));

        // Ice Golem: same story as Ice Wizard -- BuildingTargeter's own
        // performAttack is already direct damage, so this is behavior-exact.
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1315, 0.4f, 0.75f, 84, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        // Balloon: flying, buildings-only, real point-blank 0.1 attack range
        // (BuildingTargeter's own findTarget already never considers
        // isFlying/targetsAir, since Buildings never fly -- so no
        // .withTargetsAir() needed here, matching that Balloon can't hit air
        // in the real game either). Its on-death area damage isn't modeled
        // (no death-triggered effects exist in this engine yet).
        add(troop(45, "Balloon", 5.0f, Archetype::MeleeBuildingTargeter, 1679, 0.5f, 0.1f, 640, 20, 'a')
            .withFlying());

        // === Ranged Building Targeter ===
        add(troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, 0.3f, 5.0f, 180, 17, 'Y'));

        // === Defensive Structures ===
        add(building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 212, 10));
        add(building(26, "Tesla", 4.0f, 1152, 'T', 5.5f, 200, 11).withTargetsAir());
        add(building(27, "Bomb Tower", 4.0f, 1356, 'D', 6.0f, 222, 18));
        // Inferno Tower: damage is 5% of max for the first 2 seconds (20
        // ticks), 18.75% for the next 2 (tick 20-40), then full damage --
        // reset by a target switch or a stun (Zap/Freeze/etc, already
        // covered generically since ramp resets on any freeze application).
        // `damage` itself is the fully-ramped max (847); getCurrentDamage()
        // scales it down early in the ramp.
        add(building(28, "Inferno Tower", 5.0f, 1748, 'I', 6.0f, 847, 4)
            .withTargetsAir()
            .withDamageRamp(20, 40, 0.05f, 0.1875f));

        // === Spells ===
        // Arrows: 3 rapid volleys of 122 each (366 total), not one 366 lump.
        add(spell(3, "Arrows", 3.0f, 3.5f, 122, 10, '*').withRepeats(3, 2));
        add(spell(7, "Fireball", 4.0f, 2.5f, 688, 10, 'O'));
        add(spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z'));
        add(spell(30, "Rocket", 6.0f, 2.0f, 1484, 15, 'r'));
        add(spell(31, "Lightning", 6.0f, 3.5f, 1057, 5, 'j'));
        // Poison: a real damage-over-time cloud -- 92 damage once per second
        // (10 ticks) for 8 seconds, not 736 dumped in one lump. Whoever's
        // actually standing in it gets hit each tick; walking out mid-cloud
        // stops further damage, same as the real spell.
        add(spell(32, "Poison", 4.0f, 3.5f, 92, 0, 'n').withRepeats(8, 10));
        // The Log is a ground-control spell -- unlike every other spell here,
        // it rolls along the arena floor and can't hit flying units.
        add(spell(33, "The Log", 2.0f, 3.9f, 268, 8, 'o').withGroundOnly());
    }

public:
    static CardRegistry& getInstance() {
        static CardRegistry instance;
        return instance;
    }

    const CardDefinition* getCard(int id) const {
        auto it = cards.find(id);
        if (it != cards.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    const std::unordered_map<int, CardDefinition>& getAllCards() const {
        return cards;
    }
};
