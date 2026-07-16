#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include "Board.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "OnHitEffect.h"

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

    // Matches the original 4-wide, 0.6-spaced grid layout for Skeleton Army.
    static std::vector<Vector2D> skeletonArmyOffsets() {
        std::vector<Vector2D> offsets;
        for (int i = 0; i < 14; ++i) {
            float ox = (static_cast<float>(i % 4) - 1.5f) * 0.6f;
            float oy = (static_cast<float>(i / 4) - 1.0f) * 0.6f;
            offsets.push_back({ ox, oy });
        }
        return offsets;
    }

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.spawnEntity = [stats](float x, float y, int team, Board& board) {
            CardFactories::spawn(stats, x, y, team, board);
        };
        cards[stats.id] = std::move(def);
    }

    CardRegistry() {
        // === Melee Troops ===
        add(troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1399, 0.5f, 1.5f, 159, 11, 'K'));

        add(troop(4, "Goblins", 2.0f, Archetype::MeleeSquad, 169, 1.0f, 1.0f, 106, 11, 'g')
            .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f} }));

        add(troop(5, "Mini PEKKA", 4.0f, Archetype::MeleeSquad, 1056, 0.8f, 1.5f, 598, 18, 'M'));

        add(troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 636, 0.5f, 1.0f, 159, 15, 'B')
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        add(troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1548, 0.5f, 1.5f, 211, 15, 'V'));

        add(troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 67, 1.0f, 0.5f, 67, 10, 's')
            .withOffsets(skeletonArmyOffsets()));

        add(troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3458, 0.4f, 1.5f, 678, 18, 'E'));

        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1463, 0.6f, 1.5f, 325, 15, 'p'));

        add(troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 970, 0.7f, 1.0f, 254, 15, 'e')
            .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }));

        add(troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 990, 0.8f, 1.0f, 200, 7, 'l'));

        add(troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 67, 1.0f, 0.5f, 67, 10, 'k')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }));

        add(troop(39, "Giant Skeleton", 6.0f, Archetype::MeleeSquad, 2660, 0.4f, 1.5f, 172, 15, 'J'));

        // === Ranged Troops ===
        add(troop(1, "Archers", 3.0f, Archetype::RangedSquad, 254, 0.5f, 5.0f, 86, 12, 'A')
            .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} }));

        add(troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 598, 0.5f, 6.0f, 176, 11, 'U'));
        add(troop(9, "Bomber", 3.0f, Archetype::RangedSquad, 311, 0.5f, 4.5f, 271, 19, 'b'));
        add(troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 598, 0.5f, 5.5f, 228, 14, 'W'));
        add(troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 216, 0.8f, 6.5f, 93, 7, 'd'));
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 1596, 0.4f, 5.0f, 239, 25, 'w'));

        add(troop(23, "Spear Goblins", 2.0f, Archetype::RangedSquad, 110, 1.0f, 5.0f, 50, 13, 'S')
            .withOffsets({ {0.0f, 0.0f}, {0.7f, 0.0f}, {-0.7f, 0.0f} }));

        // Ice Wizard: genuinely ranged -- fires a projectile that applies the
        // freeze on arrival. (Originally this was mislabeled: it inherited
        // from RangedTroop but overrode performAttack to hit instantly,
        // bypassing the projectile entirely. Fixed now that on-hit effects
        // can ride along with a projectile instead of firing at launch.)
        add(troop(34, "Ice Wizard", 3.0f, Archetype::RangedSquad, 665, 0.5f, 5.5f, 69, 17, 'i')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        add(troop(35, "Electro Wizard", 4.0f, Archetype::RangedSquad, 590, 0.5f, 5.0f, 200, 18, 'z'));
        add(troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1010, 0.4f, 4.5f, 280, 24, 'x'));

        // === Building Targeters ===
        add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3344, 0.3f, 1.5f, 211, 15, 'G'));

        add(troop(15, "Hog Rider", 4.0f, Archetype::MeleeBuildingTargeter, 1408, 0.8f, 1.0f, 264, 15, 'H')
            .withIgnoresRiver());

        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 4256, 0.2f, 1.5f, 259, 25, 'L'));

        // Ice Golem: same story as Ice Wizard -- BuildingTargeter's own
        // performAttack is already direct damage, so this is behavior-exact.
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1047, 0.4f, 1.0f, 70, 25, 'c')
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.65f)));

        // === Ranged Building Targeter ===
        add(troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 2544, 0.3f, 6.5f, 159, 17, 'Y'));

        // === Defensive Structures ===
        add(building(25, "Cannon", 3.0f, 742, 'C', 5.5f, 127, 8));
        add(building(26, "Tesla", 4.0f, 954, 'T', 5.5f, 135, 8));
        add(building(27, "Bomb Tower", 5.0f, 1672, 'D', 6.0f, 176, 16));
        add(building(28, "Inferno Tower", 5.0f, 1408, 'I', 6.0f, 200, 4));

        // === Spells ===
        add(spell(3, "Arrows", 3.0f, 4.0f, 243, 10, '*'));
        add(spell(7, "Fireball", 4.0f, 2.5f, 572, 10, 'O'));
        add(spell(29, "Zap", 2.0f, 2.5f, 159, 3, 'Z'));
        add(spell(30, "Rocket", 6.0f, 2.0f, 1232, 15, 'r'));
        add(spell(31, "Lightning", 6.0f, 3.5f, 864, 5, 'j'));
        add(spell(32, "Poison", 4.0f, 3.5f, 600, 80, 'n'));
        add(spell(33, "The Log", 2.0f, 3.9f, 240, 8, 'o'));
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
