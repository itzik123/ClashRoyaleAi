#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include "Board.h"
#include "MeleeTroop.h"
#include "BuildingTargeter.h"
#include "RangedTroop.h"
#include "RangedBuildingTargeter.h"
#include "AreaSpell.h"

class IceTroop : public RangedTroop {
public:
    IceTroop(int id, float x, float y, int hp, int team, float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : RangedTroop(id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol) {}
        
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        target->takeDamage(damage);
        target->applyFreeze(30, 0.65f); // 3 seconds, 35% slow
    }
};

class IceBuildingTargeter : public BuildingTargeter {
public:
    IceBuildingTargeter(int id, float x, float y, int hp, int team, float speed, float attackRange, int damage, int attackCooldown, char symbol)
        : BuildingTargeter(id, x, y, hp, team, speed, attackRange, damage, attackCooldown, symbol) {}
        
    void performAttack(Board& board, std::shared_ptr<Entity> target) override {
        target->takeDamage(damage);
        target->applyFreeze(30, 0.65f);
    }
};

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

    CardRegistry() {
        // === Melee Troops ===

        cards[0] = { 0, "Knight", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 1399, team, 0.5f, 1.5f, 159, 11, 'K'));
        } };

        cards[4] = { 4, "Goblins", 2.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 169, team, 1.0f, 1.0f, 106, 11, 'g'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + 1.0f, y, 169, team, 1.0f, 1.0f, 106, 11, 'g'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x - 1.0f, y, 169, team, 1.0f, 1.0f, 106, 11, 'g'));
        } };

        cards[5] = { 5, "Mini PEKKA", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 1056, team, 0.8f, 1.5f, 598, 18, 'M'));
        } };

        cards[8] = { 8, "Barbarians", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x - 0.5f, y - 0.5f, 636, team, 0.5f, 1.0f, 159, 15, 'B'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + 0.5f, y - 0.5f, 636, team, 0.5f, 1.0f, 159, 15, 'B'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x - 0.5f, y + 0.5f, 636, team, 0.5f, 1.0f, 159, 15, 'B'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + 0.5f, y + 0.5f, 636, team, 0.5f, 1.0f, 159, 15, 'B'));
        } };

        cards[10] = { 10, "Valkyrie", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 1548, team, 0.5f, 1.5f, 211, 15, 'V'));
        } };

        cards[12] = { 12, "Skeleton Army", 3.0f, false, [](float x, float y, int team, Board& b) {
            for (int i = 0; i < 14; ++i) {
                float ox = (i % 4 - 1.5f) * 0.6f;
                float oy = (i / 4 - 1.0f) * 0.6f;
                b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + ox, y + oy, 67, team, 1.0f, 0.5f, 67, 10, 's'));
            }
        } };

        cards[13] = { 13, "P.E.K.K.A.", 7.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 3458, team, 0.4f, 1.5f, 678, 18, 'E'));
        } };

        cards[14] = { 14, "Prince", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 1463, team, 0.6f, 1.5f, 325, 15, 'p'));
        } };

        cards[17] = { 17, "Elite Barbarians", 6.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x - 0.5f, y, 970, team, 0.7f, 1.0f, 254, 15, 'e'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + 0.5f, y, 970, team, 0.7f, 1.0f, 254, 15, 'e'));
        } };

        cards[21] = { 21, "Lumberjack", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 990, team, 0.8f, 1.0f, 200, 7, 'l'));
        } };

        cards[24] = { 24, "Skeletons", 1.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 67, team, 1.0f, 0.5f, 67, 10, 'k'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x + 0.6f, y, 67, team, 1.0f, 0.5f, 67, 10, 'k'));
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x - 0.6f, y, 67, team, 1.0f, 0.5f, 67, 10, 'k'));
        } };

        cards[39] = { 39, "Giant Skeleton", 6.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<MeleeTroop>(b.allocateId(), x, y, 2660, team, 0.4f, 1.5f, 172, 15, 'J'));
        } };

        // === Ranged Troops ===

        cards[1] = { 1, "Archers", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 254, team, 0.5f, 5.0f, 86, 12, 'A'));
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x + 1.0f, y, 254, team, 0.5f, 5.0f, 86, 12, 'A'));
        } };

        cards[6] = { 6, "Musketeer", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 598, team, 0.5f, 6.0f, 176, 11, 'U'));
        } };

        cards[9] = { 9, "Bomber", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 311, team, 0.5f, 4.5f, 271, 19, 'b'));
        } };

        cards[11] = { 11, "Wizard", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 598, team, 0.5f, 5.5f, 228, 14, 'W'));
        } };

        cards[20] = { 20, "Dart Goblin", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 216, team, 0.8f, 6.5f, 93, 7, 'd'));
        } };

        cards[22] = { 22, "Bowler", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 1596, team, 0.4f, 5.0f, 239, 25, 'w'));
        } };

        cards[23] = { 23, "Spear Goblins", 2.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 110, team, 1.0f, 5.0f, 50, 13, 'S'));
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x + 0.7f, y, 110, team, 1.0f, 5.0f, 50, 13, 'S'));
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x - 0.7f, y, 110, team, 1.0f, 5.0f, 50, 13, 'S'));
        } };

        cards[34] = { 34, "Ice Wizard", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<IceTroop>(b.allocateId(), x, y, 665, team, 0.5f, 5.5f, 69, 17, 'i'));
        } };

        cards[35] = { 35, "Electro Wizard", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 590, team, 0.5f, 5.0f, 200, 18, 'z'));
        } };

        cards[36] = { 36, "Executioner", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedTroop>(b.allocateId(), x, y, 1010, team, 0.4f, 4.5f, 280, 24, 'x'));
        } };

        // === Building Targeters ===

        cards[2] = { 2, "Giant", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<BuildingTargeter>(b.allocateId(), x, y, 3344, team, 0.3f, 1.5f, 211, 15, 'G'));
        } };

        cards[15] = { 15, "Hog Rider", 4.0f, false, [](float x, float y, int team, Board& b) {
            auto hog = std::make_shared<BuildingTargeter>(b.allocateId(), x, y, 1408, team, 0.8f, 1.0f, 264, 15, 'H');
            hog->setIgnoresRiver(true);
            b.addEntity(hog);
        } };

        cards[19] = { 19, "Golem", 8.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<BuildingTargeter>(b.allocateId(), x, y, 4256, team, 0.2f, 1.5f, 259, 25, 'L'));
        } };

        cards[40] = { 40, "Ice Golem", 2.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<IceBuildingTargeter>(b.allocateId(), x, y, 1047, team, 0.4f, 1.0f, 70, 25, 'c'));
        } };

        // === Ranged Building Targeter ===

        cards[18] = { 18, "Royal Giant", 6.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<RangedBuildingTargeter>(b.allocateId(), x, y, 2544, team, 0.3f, 6.5f, 159, 17, 'Y'));
        } };

        // === Defensive Structures ===

        cards[25] = { 25, "Cannon", 3.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<Building>(b.allocateId(), x, y, 742, team, 'C', 5.5f, 127, 8));
        } };

        cards[26] = { 26, "Tesla", 4.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<Building>(b.allocateId(), x, y, 954, team, 'T', 5.5f, 135, 8));
        } };

        cards[27] = { 27, "Bomb Tower", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<Building>(b.allocateId(), x, y, 1672, team, 'D', 6.0f, 176, 16));
        } };

        cards[28] = { 28, "Inferno Tower", 5.0f, false, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<Building>(b.allocateId(), x, y, 1408, team, 'I', 6.0f, 200, 4));
        } };

        // === Spells ===

        cards[3] = { 3, "Arrows", 3.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 4.0f, 243, 10, '*'));
        } };

        cards[7] = { 7, "Fireball", 4.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 2.5f, 572, 10, 'O'));
        } };

        cards[29] = { 29, "Zap", 2.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 2.5f, 159, 3, 'Z'));
        } };

        cards[30] = { 30, "Rocket", 6.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 2.0f, 1232, 15, 'r'));
        } };

        cards[31] = { 31, "Lightning", 6.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 3.5f, 864, 5, 'j'));
        } };

        cards[32] = { 32, "Poison", 4.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 3.5f, 600, 80, 'n'));
        } };

        cards[33] = { 33, "The Log", 2.0f, true, [](float x, float y, int team, Board& b) {
            b.addEntity(std::make_shared<AreaSpell>(b.allocateId(), x, y, team, 3.9f, 240, 8, 'o'));
        } };
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