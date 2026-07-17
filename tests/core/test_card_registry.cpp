#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "CardRegistry.h"
#include "MeleeTroop.h"
#include "RangedTroop.h"
#include "BuildingTargeter.h"
#include "RangedBuildingTargeter.h"
#include "Building.h"
#include "AreaSpell.h"
#include "Projectile.h"
#include <vector>
#include <tuple>

TEST_CASE("Spawned entities carry the card's display name", "[card_registry][name]") {
    Board board;

    SECTION("single-unit card (Knight)") {
        CardRegistry::getInstance().getCard(0)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().back()->name == "Knight");
    }

    SECTION("squad card: every unit gets the same name (Goblins)") {
        CardRegistry::getInstance().getCard(4)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().size() == 3);
        for (const auto& e : board.getEntities()) {
            REQUIRE(e->name == "Goblins");
        }
    }

    SECTION("spell card (Fireball)") {
        CardRegistry::getInstance().getCard(7)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().back()->name == "Fireball");
    }

    SECTION("defensive building (Cannon)") {
        CardRegistry::getInstance().getCard(25)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().back()->name == "Cannon");
    }
}

TEST_CASE("Every currently-defined card resolves with the exact original stats", "[card_registry][data]") {
    // id, name, cost, isSpell -- transcribed from the pre-refactor CardRegistry.
    // Card ids 16, 37 and 38 were never defined before this refactor and stay
    // that way here on purpose (not something this refactor should silently add).
    const std::vector<std::tuple<int, std::string, float, bool>> expected = {
        {0, "Knight", 3.0f, false}, {1, "Archers", 3.0f, false}, {2, "Giant", 5.0f, false},
        {3, "Arrows", 3.0f, true}, {4, "Goblins", 2.0f, false}, {5, "Mini PEKKA", 4.0f, false},
        {6, "Musketeer", 4.0f, false}, {7, "Fireball", 4.0f, true}, {8, "Barbarians", 5.0f, false},
        {9, "Bomber", 3.0f, false}, {10, "Valkyrie", 4.0f, false}, {11, "Wizard", 5.0f, false},
        {12, "Skeleton Army", 3.0f, false}, {13, "P.E.K.K.A.", 7.0f, false}, {14, "Prince", 5.0f, false},
        {15, "Hog Rider", 4.0f, false}, {17, "Elite Barbarians", 6.0f, false}, {18, "Royal Giant", 6.0f, false},
        {19, "Golem", 8.0f, false}, {20, "Dart Goblin", 3.0f, false}, {21, "Lumberjack", 4.0f, false},
        {22, "Bowler", 5.0f, false}, {23, "Spear Goblins", 2.0f, false}, {24, "Skeletons", 1.0f, false},
        {25, "Cannon", 3.0f, false}, {26, "Tesla", 4.0f, false}, {27, "Bomb Tower", 5.0f, false},
        {28, "Inferno Tower", 5.0f, false}, {29, "Zap", 2.0f, true}, {30, "Rocket", 6.0f, true},
        {31, "Lightning", 6.0f, true}, {32, "Poison", 4.0f, true}, {33, "The Log", 2.0f, true},
        {34, "Ice Wizard", 3.0f, false}, {35, "Electro Wizard", 4.0f, false}, {36, "Executioner", 5.0f, false},
        {39, "Giant Skeleton", 6.0f, false}, {40, "Ice Golem", 2.0f, false},
    };

    const auto& registry = CardRegistry::getInstance();
    REQUIRE(registry.getAllCards().size() == expected.size());

    for (const auto& [id, name, cost, expectedIsSpell] : expected) {
        const CardDefinition* def = registry.getCard(id);
        INFO("card id " << id << " (" << name << ")");
        REQUIRE(def != nullptr);
        REQUIRE(def->id == id);
        REQUIRE(def->name == name);
        REQUIRE(def->cost == Catch::Approx(cost));
        REQUIRE(def->isSpell == expectedIsSpell);
    }
}

TEST_CASE("Card ids that were never defined stay undefined", "[card_registry][data]") {
    const auto& registry = CardRegistry::getInstance();
    REQUIRE(registry.getCard(16) == nullptr);
    REQUIRE(registry.getCard(37) == nullptr);
    REQUIRE(registry.getCard(38) == nullptr);
    REQUIRE(registry.getCard(9999) == nullptr);
}

TEST_CASE("MeleeSquad archetype: single unit (Knight)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* knight = CardRegistry::getInstance().getCard(0);
    REQUIRE(knight != nullptr);

    knight->spawnEntity(10.0f, 10.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1);
    auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities()[0]);
    REQUIRE(troop != nullptr);
    REQUIRE(troop->team == 0);
    REQUIRE(troop->symbol == 'K');
    REQUIRE(troop->position.x == Catch::Approx(10.0f));
    REQUIRE(troop->position.y == Catch::Approx(10.0f));
}

TEST_CASE("MeleeSquad archetype: multiple units at their offsets (Goblins)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* goblins = CardRegistry::getInstance().getCard(4);
    REQUIRE(goblins != nullptr);

    goblins->spawnEntity(10.0f, 10.0f, 1, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 3);
    REQUIRE(board.getEntities()[0]->position.x == Catch::Approx(10.0f));
    REQUIRE(board.getEntities()[1]->position.x == Catch::Approx(11.0f));
    REQUIRE(board.getEntities()[2]->position.x == Catch::Approx(9.0f));
    for (const auto& e : board.getEntities()) {
        REQUIRE(e->position.y == Catch::Approx(10.0f));
        REQUIRE(e->team == 1);
        REQUIRE(std::dynamic_pointer_cast<MeleeTroop>(e) != nullptr);
    }
}

TEST_CASE("RangedSquad archetype (Musketeer)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* musketeer = CardRegistry::getInstance().getCard(6);
    musketeer->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1);
    REQUIRE(std::dynamic_pointer_cast<RangedTroop>(board.getEntities()[0]) != nullptr);
}

TEST_CASE("MeleeBuildingTargeter archetype (Giant)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* giant = CardRegistry::getInstance().getCard(2);
    giant->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto targeter = std::dynamic_pointer_cast<BuildingTargeter>(board.getEntities()[0]);
    REQUIRE(targeter != nullptr);
    REQUIRE_FALSE(targeter->riverIgnores);
}

TEST_CASE("MeleeBuildingTargeter archetype wires ignoresRiver for Hog Rider only", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* hogRider = CardRegistry::getInstance().getCard(15);
    hogRider->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto targeter = std::dynamic_pointer_cast<BuildingTargeter>(board.getEntities()[0]);
    REQUIRE(targeter != nullptr);
    REQUIRE(targeter->riverIgnores);
}

TEST_CASE("RangedBuildingTargeter archetype (Royal Giant)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* royalGiant = CardRegistry::getInstance().getCard(18);
    royalGiant->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(std::dynamic_pointer_cast<RangedBuildingTargeter>(board.getEntities()[0]) != nullptr);
}

TEST_CASE("DefensiveBuilding archetype (Cannon)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* cannon = CardRegistry::getInstance().getCard(25);
    cannon->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto building = std::dynamic_pointer_cast<Building>(board.getEntities()[0]);
    REQUIRE(building != nullptr);
    REQUIRE(building->getCollisionRadius() == Catch::Approx(1.0f));
}

TEST_CASE("Spell archetype (Fireball)", "[card_registry][archetype]") {
    Board board;
    const CardDefinition* fireball = CardRegistry::getInstance().getCard(7);
    fireball->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities()[0]);
    REQUIRE(spell != nullptr);
    REQUIRE_FALSE(spell->isTargetable());
}

TEST_CASE("Ice Wizard is genuinely ranged: freeze lands with the arrow, not when it's fired", "[card_registry][on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10); // distance 1.0 from (5,5)
    spawn(board, enemy);

    const CardDefinition* iceWizard = CardRegistry::getInstance().getCard(34);
    iceWizard->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto iceWizardEntity = board.getEntities().back();
    REQUIRE(std::dynamic_pointer_cast<RangedTroop>(iceWizardEntity) != nullptr);

    iceWizardEntity->update(board); // fires: spawns a projectile, does not freeze yet
    board.commitPendingEntities();
    REQUIRE(enemy->freezeTicks == 0);

    auto projectile = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(projectile != nullptr);

    projectile->update(board); // projectile speed 1.5 >= distance 1.0: lands this tick
    REQUIRE(enemy->freezeTicks == 30);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.65f));
}

TEST_CASE("Ice Golem applies freeze on hit via the on-hit decorator", "[card_registry][on_hit]") {
    Board board;
    // Ice Golem is a MeleeBuildingTargeter: it only ever targets Buildings.
    auto enemy = std::make_shared<Building>(1, 5.0f, 5.5f, 1000, 1, 'C', 5.0f, 10, 10);
    spawn(board, enemy);

    const CardDefinition* iceGolem = CardRegistry::getInstance().getCard(40);
    iceGolem->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    board.getEntities().back()->update(board);

    REQUIRE(enemy->freezeTicks == 30);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.65f));
}

TEST_CASE("Ordinary melee troops are unaffected by the on-hit decorator (Knight)", "[card_registry][on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);

    const CardDefinition* knight = CardRegistry::getInstance().getCard(0);
    knight->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    board.getEntities().back()->update(board);

    REQUIRE(enemy->freezeTicks == 0);
}
