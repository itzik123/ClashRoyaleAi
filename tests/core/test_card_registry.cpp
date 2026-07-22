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
        REQUIRE(board.getEntities().size() == 4);
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

TEST_CASE("Every currently-defined card resolves with id/name/cost/isSpell", "[card_registry][data]") {
    // id, name, cost, isSpell. Cost reflects the same real-game data sync as
    // the rest of CardRegistry's stats (see the constructor's header comment).
    // Card ids 16, 37 and 38 were never defined and stay that way here on
    // purpose (not something a data sync should silently fill in).
    const std::vector<std::tuple<int, std::string, float, bool>> expected = {
        {0, "Knight", 3.0f, false}, {1, "Archers", 3.0f, false}, {2, "Giant", 5.0f, false},
        {3, "Arrows", 3.0f, true}, {4, "Goblins", 2.0f, false}, {5, "Mini PEKKA", 4.0f, false},
        {6, "Musketeer", 4.0f, false}, {7, "Fireball", 4.0f, true}, {8, "Barbarians", 5.0f, false},
        {9, "Bomber", 2.0f, false}, {10, "Valkyrie", 4.0f, false}, {11, "Wizard", 5.0f, false},
        {12, "Skeleton Army", 3.0f, false}, {13, "P.E.K.K.A.", 7.0f, false}, {14, "Prince", 5.0f, false},
        {15, "Hog Rider", 4.0f, false}, {17, "Elite Barbarians", 6.0f, false}, {18, "Royal Giant", 6.0f, false},
        {19, "Golem", 8.0f, false}, {20, "Dart Goblin", 3.0f, false}, {21, "Lumberjack", 4.0f, false},
        {22, "Bowler", 5.0f, false}, {23, "Spear Goblins", 2.0f, false}, {24, "Skeletons", 1.0f, false},
        {25, "Cannon", 3.0f, false}, {26, "Tesla", 4.0f, false}, {27, "Bomb Tower", 4.0f, false},
        {28, "Inferno Tower", 5.0f, false}, {29, "Zap", 2.0f, true}, {30, "Rocket", 6.0f, true},
        {31, "Lightning", 6.0f, true}, {32, "Poison", 4.0f, true}, {33, "The Log", 2.0f, true},
        {34, "Ice Wizard", 3.0f, false}, {35, "Electro Wizard", 4.0f, false}, {36, "Executioner", 5.0f, false},
        {39, "Giant Skeleton", 6.0f, false}, {40, "Ice Golem", 2.0f, false},
        {41, "Minions", 3.0f, false}, {42, "Minion Horde", 5.0f, false}, {43, "Mega Minion", 3.0f, false},
        {44, "Baby Dragon", 4.0f, false}, {45, "Balloon", 5.0f, false},
        // 2026 roster expansion (ids 46-107) -- see CardRegistry.h's own
        // constructor comment for the excluded-cards list this stops short of.
        {46, "Dark Prince", 4.0f, false}, {47, "Royal Ghost", 3.0f, false}, {48, "Mega Knight", 7.0f, false},
        {49, "Battle Healer", 4.0f, false}, {50, "Bandit", 3.0f, false}, {51, "Berserker", 2.0f, false},
        {52, "Miner", 3.0f, false}, {53, "Fisherman", 3.0f, false}, {54, "Ronin", 5.0f, false},
        {55, "Goblin Machine", 5.0f, false}, {56, "Inferno Dragon", 4.0f, false}, {57, "Electro Dragon", 5.0f, false},
        {58, "Night Witch", 4.0f, false}, {59, "Phoenix", 4.0f, false}, {60, "Sparky", 6.0f, false},
        {61, "Princess", 3.0f, false}, {62, "Hunter", 4.0f, false}, {63, "Magic Archer", 4.0f, false},
        {64, "Firecracker", 3.0f, false}, {65, "Skeleton Dragons", 4.0f, false}, {66, "Goblin Demolisher", 4.0f, false},
        {67, "Flying Machine", 4.0f, false}, {68, "Mother Witch", 4.0f, false}, {69, "Cannon Cart", 5.0f, false},
        {70, "Furnace", 4.0f, false}, {71, "Witch", 5.0f, false}, {72, "Ice Spirit", 1.0f, false},
        {73, "Fire Spirit", 1.0f, false}, {74, "Heal Spirit", 1.0f, false}, {75, "Electro Spirit", 1.0f, false},
        {76, "Guards", 3.0f, false}, {77, "Royal Recruits", 7.0f, false}, {78, "Bats", 2.0f, false},
        {79, "Zappies", 4.0f, false}, {80, "Three Musketeers", 9.0f, false}, {81, "Battle Ram", 4.0f, false},
        {82, "Royal Hogs", 5.0f, false}, {83, "Wall Breakers", 2.0f, false}, {84, "Electro Giant", 7.0f, false},
        {85, "Suspicious Bush", 2.0f, false}, {86, "Rune Giant", 4.0f, false}, {87, "Ram Rider", 5.0f, false},
        {88, "Goblin Giant", 6.0f, false}, {89, "Skeleton Barrel", 3.0f, false}, {90, "Elixir Golem", 3.0f, false},
        {91, "Lava Hound", 7.0f, false}, {92, "X-Bow", 6.0f, false}, {93, "Mortar", 4.0f, false},
        {94, "Barbarian Hut", 6.0f, false}, {95, "Goblin Hut", 4.0f, false}, {96, "Tombstone", 3.0f, false},
        {97, "Goblin Cage", 4.0f, false}, {98, "Goblin Drill", 4.0f, false}, {99, "Elixir Collector", 6.0f, false},
        {100, "Giant Snowball", 2.0f, true}, {101, "Barbarian Barrel", 2.0f, true}, {102, "Goblin Curse", 2.0f, true},
        {103, "Earthquake", 3.0f, true}, {104, "Void", 3.0f, true}, {105, "Vines", 3.0f, true},
        {106, "Tornado", 3.0f, true}, {107, "Freeze", 4.0f, true}, {108, "Rage", 2.0f, true},
        {109, "Goblin Barrel", 3.0f, true}, {110, "Graveyard", 5.0f, true}, {111, "Royal Delivery", 3.0f, true},
        {112, "Goblin Gang", 3.0f, false}, {113, "Rascals", 5.0f, false}, {114, "Clone", 3.0f, true},
        {115, "Mighty Miner", 4.0f, false}, {116, "Golden Knight", 4.0f, false}, {117, "Skeleton King", 4.0f, false},
        {118, "Archer Queen", 5.0f, false}, {119, "Monk", 4.0f, false}, {120, "Little Prince", 3.0f, false},
        {121, "Goblinstein", 5.0f, false}, {122, "Boss Bandit", 6.0f, false},
        {123, "Wall Breakers", 2.0f, false}, // Evolution slot -- same name/cost as base id 83
        {124, "Zap", 2.0f, true}, // Evolution slot -- same name/cost as base id 29
        {125, "Skeletons", 1.0f, false}, // Evolution slot -- same name/cost as base id 24
        {126, "Bats", 2.0f, false}, // Evolution slot -- same name/cost as base id 78
        {127, "Bomber", 2.0f, false}, // Evolution slot -- same name/cost as base id 9
        {128, "Archers", 3.0f, false}, // Evolution slot -- same name/cost as base id 1
        {129, "Cannon", 3.0f, false}, // Evolution slot -- same name/cost as base id 25
        {130, "Firecracker", 3.0f, false}, // Evolution slot -- same name/cost as base id 64
        {131, "Dart Goblin", 3.0f, false}, // Evolution slot -- same name/cost as base id 20
        {132, "Goblin Barrel", 3.0f, true}, // Evolution slot -- same name/cost as base id 109
        {133, "Skeleton Army", 3.0f, false}, // Evolution slot -- same name/cost as base id 12
        {134, "Skeleton Barrel", 3.0f, false}, // Evolution slot -- same name/cost as base id 89
        {135, "Knight", 3.0f, false}, // Evolution slot -- same name/cost as base id 0
        {136, "Royal Ghost", 3.0f, false}, // Evolution slot -- same name/cost as base id 47
        {137, "Baby Dragon", 4.0f, false}, // Evolution slot -- same name/cost as base id 44
        {138, "Furnace", 4.0f, false}, // Evolution slot -- same name/cost as base id 70
        {139, "Goblin Cage", 4.0f, false}, // Evolution slot -- same name/cost as base id 97
        {140, "Musketeer", 4.0f, false}, // Evolution slot -- same name/cost as base id 6
        {141, "Wizard", 5.0f, false}, // Evolution slot -- same name/cost as base id 11
        {142, "Witch", 5.0f, false}, // Evolution slot -- same name/cost as base id 71
        {143, "Royal Giant", 6.0f, false}, // Evolution slot -- same name/cost as base id 18
        {144, "Ice Spirit", 1.0f, false}, // Evolution slot -- same name/cost as base id 72
        {145, "Princess", 3.0f, false}, // Evolution slot -- same name/cost as base id 61
        {146, "Hunter", 4.0f, false}, // Evolution slot -- same name/cost as base id 62
        {147, "Valkyrie", 4.0f, false}, // Evolution slot -- same name/cost as base id 10
        {148, "P.E.K.K.A.", 7.0f, false}, // Evolution slot -- same name/cost as base id 13
        {149, "Minion Horde", 5.0f, false}, // Evolution slot -- same name/cost as base id 42
        {150, "Royal Recruits", 7.0f, false}, // Evolution slot -- same name/cost as base id 77
        {151, "Electro Dragon", 5.0f, false}, // Evolution slot -- same name/cost as base id 57
        {152, "Mortar", 4.0f, false}, // Evolution slot -- same name/cost as base id 93
        {153, "Goblin Drill", 4.0f, false}, // Evolution slot -- same name/cost as base id 98
        {154, "Tesla", 4.0f, false}, // Evolution slot -- same name/cost as base id 26
        {155, "Barbarians", 5.0f, false}, // Evolution slot -- same name/cost as base id 8
        {156, "Lumberjack", 4.0f, false}, // Evolution slot -- same name/cost as base id 21
        {157, "Executioner", 5.0f, false}, // Evolution slot -- same name/cost as base id 36
        {158, "Giant Snowball", 2.0f, true}, // Evolution slot -- same name/cost as base id 100
        {159, "Goblin Giant", 6.0f, false}, // Evolution slot -- same name/cost as base id 88
        {160, "Mega Knight", 7.0f, false}, // Evolution slot -- same name/cost as base id 48
        {161, "Battle Ram", 4.0f, false}, // Evolution slot -- same name/cost as base id 81
        {162, "Royal Hogs", 5.0f, false}, // Evolution slot -- same name/cost as base id 82
        {163, "Inferno Dragon", 4.0f, false}, // Evolution slot -- same name/cost as base id 56
        {164, "Mirror", 3.0f, true},
        {165, "Spirit Empress", 3.0f, false},
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

TEST_CASE("CardDefinition::isBuilding is true only for DefensiveBuilding archetype cards", "[card_registry][isBuilding]") {
    const auto& registry = CardRegistry::getInstance();

    SECTION("the four defensive buildings") {
        REQUIRE(registry.getCard(25)->isBuilding); // Cannon
        REQUIRE(registry.getCard(26)->isBuilding); // Tesla
        REQUIRE(registry.getCard(27)->isBuilding); // Bomb Tower
        REQUIRE(registry.getCard(28)->isBuilding); // Inferno Tower
    }

    SECTION("troops, building-targeters and spells are all false") {
        REQUIRE_FALSE(registry.getCard(0)->isBuilding);  // Knight (MeleeSquad)
        REQUIRE_FALSE(registry.getCard(6)->isBuilding);  // Musketeer (RangedSquad)
        REQUIRE_FALSE(registry.getCard(2)->isBuilding);  // Giant (MeleeBuildingTargeter)
        REQUIRE_FALSE(registry.getCard(18)->isBuilding); // Royal Giant (RangedBuildingTargeter)
        REQUIRE_FALSE(registry.getCard(7)->isBuilding);  // Fireball (Spell)
    }
}

TEST_CASE("CardDefinition::placementRadius matches what the archetype actually spawns with", "[card_registry][placement]") {
    const auto& registry = CardRegistry::getInstance();

    SECTION("troop-shaped archetypes get the implicit troop radius") {
        REQUIRE(registry.getCard(0)->placementRadius == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));  // Knight, MeleeSquad
        REQUIRE(registry.getCard(6)->placementRadius == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));  // Musketeer, RangedSquad
        REQUIRE(registry.getCard(2)->placementRadius == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS));  // Giant, MeleeBuildingTargeter
        REQUIRE(registry.getCard(18)->placementRadius == Catch::Approx(Entity::IMPLICIT_TROOP_RADIUS)); // Royal Giant, RangedBuildingTargeter
    }

    SECTION("DefensiveBuilding archetype gets Building's own collision radius") {
        REQUIRE(registry.getCard(25)->placementRadius == Catch::Approx(Building::COLLISION_RADIUS)); // Cannon
    }
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

    REQUIRE(board.getEntities().size() == 4);
    REQUIRE(board.getEntities()[0]->position.x == Catch::Approx(9.5f));
    REQUIRE(board.getEntities()[0]->position.y == Catch::Approx(9.5f));
    REQUIRE(board.getEntities()[1]->position.x == Catch::Approx(10.5f));
    REQUIRE(board.getEntities()[1]->position.y == Catch::Approx(9.5f));
    REQUIRE(board.getEntities()[2]->position.x == Catch::Approx(9.5f));
    REQUIRE(board.getEntities()[2]->position.y == Catch::Approx(10.5f));
    REQUIRE(board.getEntities()[3]->position.x == Catch::Approx(10.5f));
    REQUIRE(board.getEntities()[3]->position.y == Catch::Approx(10.5f));
    for (const auto& e : board.getEntities()) {
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

TEST_CASE("Executioner's axe hits its target twice: on arrival, then again on the return trip", "[card_registry][flying]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(36)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto executioner = std::dynamic_pointer_cast<RangedTroop>(board.getEntities().back());
    REQUIRE(executioner != nullptr);

    executioner->update(board); // fires: spawns the boomerang projectile
    board.commitPendingEntities();
    auto axe = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(axe != nullptr);

    axe->update(board); // projectile speed >= distance 1.0: outbound hit lands this tick
    REQUIRE(enemy->hp == 821); // 1000 - 179
    REQUIRE(axe->isAlive()); // still out on its return trip, not dead after one hit

    // Matches the real GameManager::step() contract (only ever calls
    // update() on entities still isAlive()) -- Projectile, like AreaSpell,
    // has no internal guard against being updated again after it dies.
    while (axe->isAlive()) axe->update(board);
    REQUIRE(enemy->hp == 642); // 1000 - 179*2
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

TEST_CASE("Golem splits into two Golemites on death", "[card_registry][death]") {
    Board board;
    const CardDefinition* golem = CardRegistry::getInstance().getCard(19);
    golem->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto golemEntity = board.getEntities()[0];
    golemEntity->takeDamage(golemEntity->hp); // dies
    board.cleanDeadEntities(); // fires the death effect, removes the Golem
    board.commitPendingEntities(); // the two Golemites become visible

    REQUIRE(board.getEntities().size() == 2);
    for (const auto& e : board.getEntities()) {
        auto golemite = std::dynamic_pointer_cast<BuildingTargeter>(e);
        REQUIRE(golemite != nullptr);
        REQUIRE(golemite->name == "Golemite");
        REQUIRE(golemite->team == 0);
        REQUIRE(golemite->position.y == Catch::Approx(5.0f));
    }
}

TEST_CASE("Golem also deals death-explosion damage alongside spawning Golemites (CompositeDeathEffect)", "[card_registry][death]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1); // dist 1.0 from (5,5)
    spawn(board, enemy);

    const CardDefinition* golem = CardRegistry::getInstance().getCard(19);
    golem->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto golemEntity = board.getEntities().back();
    golemEntity->takeDamage(golemEntity->hp);
    board.cleanDeadEntities();
    board.commitPendingEntities();

    REQUIRE(enemy->hp == 100000 - 200); // the explosion half of the composite effect
    int golemiteCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Golemite") golemiteCount++;
    }
    REQUIRE(golemiteCount == 2); // the spawn half still fires too
}

TEST_CASE("Giant Skeleton, Ice Golem and Balloon deal death-explosion damage", "[card_registry][death]") {
    SECTION("Giant Skeleton") {
        Board board;
        auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1);
        spawn(board, enemy);
        CardRegistry::getInstance().getCard(39)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto entity = board.getEntities().back();
        entity->takeDamage(entity->hp);
        board.cleanDeadEntities();
        REQUIRE(enemy->hp == 100000 - 300);
    }

    SECTION("Ice Golem") {
        Board board;
        auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1);
        spawn(board, enemy);
        CardRegistry::getInstance().getCard(40)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto entity = board.getEntities().back();
        entity->takeDamage(entity->hp);
        board.cleanDeadEntities();
        REQUIRE(enemy->hp == 100000 - 84);
    }

    SECTION("Balloon") {
        Board board;
        auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1);
        spawn(board, enemy);
        CardRegistry::getInstance().getCard(45)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto entity = board.getEntities().back();
        entity->takeDamage(entity->hp);
        board.cleanDeadEntities();
        REQUIRE(enemy->hp == 100000 - 240);
    }
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

TEST_CASE("DefensiveBuilding archetype: targetsAir matches real-game data per card", "[card_registry][archetype][flying]") {
    Board board;

    SECTION("Tesla hits air") {
        CardRegistry::getInstance().getCard(26)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board.getEntities().back());
        REQUIRE(building != nullptr);
        REQUIRE(building->targetsAir);
    }

    SECTION("Inferno Tower hits air") {
        CardRegistry::getInstance().getCard(28)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board.getEntities().back());
        REQUIRE(building != nullptr);
        REQUIRE(building->targetsAir);
    }

    SECTION("Cannon does not hit air") {
        CardRegistry::getInstance().getCard(25)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board.getEntities().back());
        REQUIRE(building != nullptr);
        REQUIRE_FALSE(building->targetsAir);
    }

    SECTION("Bomb Tower does not hit air") {
        CardRegistry::getInstance().getCard(27)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board.getEntities().back());
        REQUIRE(building != nullptr);
        REQUIRE_FALSE(building->targetsAir);
    }
}

TEST_CASE("Inferno Tower's damage ramps up the longer it stays locked onto the same target", "[card_registry][ramp]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(28)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto tower = std::dynamic_pointer_cast<Building>(board.getEntities().back());
    REQUIRE(tower != nullptr);

    tower->update(board); // ticksOnTarget == 0 on first lock: stage 1 (5% of 847)
    REQUIRE(enemy->hp == 1000000 - 42);

    for (int i = 0; i < 39; ++i) tower->update(board); // advance to just before ticksOnTarget == 40
    int hpBefore = enemy->hp;
    tower->update(board); // ticksOnTarget == 40 (an attack tick, cooldown 4 divides evenly): full damage
    REQUIRE(hpBefore - enemy->hp == 847);
}

// ---------------- flying cards ----------------

TEST_CASE("Archers can hit flying enemies (real-game Target: Air & Ground)", "[card_registry][flying]") {
    Board board;
    CardRegistry::getInstance().getCard(1)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    for (const auto& e : board.getEntities()) {
        auto troop = std::dynamic_pointer_cast<RangedTroop>(e);
        REQUIRE(troop != nullptr);
        REQUIRE(troop->targetsAir);
        REQUIRE_FALSE(troop->isFlying); // ground troop, just capable of hitting air
    }
}

TEST_CASE("Minions and Minion Horde spawn as flying, air-targeting melee squads", "[card_registry][flying]") {
    Board board;

    SECTION("Minions: 3 units") {
        CardRegistry::getInstance().getCard(41)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().size() == 3);
        for (const auto& e : board.getEntities()) {
            auto troop = std::dynamic_pointer_cast<MeleeTroop>(e);
            REQUIRE(troop != nullptr);
            REQUIRE(troop->isFlying);
            REQUIRE(troop->targetsAir);
        }
    }

    SECTION("Minion Horde: 6 units") {
        CardRegistry::getInstance().getCard(42)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().size() == 6);
        for (const auto& e : board.getEntities()) {
            auto troop = std::dynamic_pointer_cast<MeleeTroop>(e);
            REQUIRE(troop != nullptr);
            REQUIRE(troop->isFlying);
            REQUIRE(troop->targetsAir);
        }
    }
}

TEST_CASE("Mega Minion spawns as a flying, air-targeting melee unit", "[card_registry][flying]") {
    Board board;
    CardRegistry::getInstance().getCard(43)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto troop = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities()[0]);
    REQUIRE(troop != nullptr);
    REQUIRE(troop->isFlying);
    REQUIRE(troop->targetsAir);
}

TEST_CASE("Baby Dragon spawns as a flying, air-targeting ranged unit", "[card_registry][flying]") {
    Board board;
    CardRegistry::getInstance().getCard(44)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto troop = std::dynamic_pointer_cast<RangedTroop>(board.getEntities()[0]);
    REQUIRE(troop != nullptr);
    REQUIRE(troop->isFlying);
    REQUIRE(troop->targetsAir);
}

TEST_CASE("Balloon spawns flying but never sets targetsAir (buildings-only, matches the real card)", "[card_registry][flying]") {
    Board board;
    CardRegistry::getInstance().getCard(45)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto targeter = std::dynamic_pointer_cast<BuildingTargeter>(board.getEntities()[0]);
    REQUIRE(targeter != nullptr);
    REQUIRE(targeter->isFlying);
    REQUIRE_FALSE(targeter->targetsAir);
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

TEST_CASE("Poison deals damage every second for 8 seconds, not a lump sum", "[card_registry][repeat]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(32)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    spell->update(board); // first tick lands immediately (no initial delay)
    REQUIRE(enemy->hp == 100000 - 92);
    REQUIRE(spell->isAlive()); // 7 more ticks left over the following ~7 seconds

    while (spell->isAlive()) spell->update(board);
    REQUIRE(enemy->hp == 100000 - 92 * 8); // all 8 ticks landed in total
}

TEST_CASE("Poison stops damaging a unit that walks out of the cloud mid-duration", "[card_registry][repeat]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(32)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    spell->update(board); // 1st tick lands
    REQUIRE(enemy->hp == 100000 - 92);

    enemy->position = { 50.0f, 50.0f }; // walks far outside the 3.5 radius
    while (spell->isAlive()) spell->update(board);
    REQUIRE(enemy->hp == 100000 - 92); // no further ticks landed after it left
}

TEST_CASE("Arrows deals damage in 3 rapid volleys, not one lump sum", "[card_registry][repeat]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(3)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);
    REQUIRE(enemy->hp == 100000 - 123 * 3); // all 3 volleys landed
}

TEST_CASE("The Log is ground-only and does not hit flying enemies, unlike Fireball", "[card_registry][flying]") {
    SECTION("The Log") {
        Board board;
        auto flyingEnemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 1000, 1); // dist 1.0
        flyingEnemy->isFlying = true;
        spawn(board, flyingEnemy);

        CardRegistry::getInstance().getCard(33)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto spell = board.getEntities().back();
        while (spell->isAlive()) spell->update(board); // wait out its delay until it detonates

        REQUIRE(flyingEnemy->hp == 1000); // untouched
    }

    SECTION("Fireball") {
        Board board;
        auto flyingEnemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 1000, 1); // dist 1.0
        flyingEnemy->isFlying = true;
        spawn(board, flyingEnemy);

        CardRegistry::getInstance().getCard(7)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto spell = board.getEntities().back();
        while (spell->isAlive()) spell->update(board);

        REQUIRE(flyingEnemy->hp < 1000); // hit, same as any ground enemy
    }
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

TEST_CASE("Electro Wizard stuns on hit via the on-hit decorator (freeze with slowFactor 0)", "[card_registry][on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10); // distance 1.0 from (5,5)
    spawn(board, enemy);

    const CardDefinition* electroWizard = CardRegistry::getInstance().getCard(35);
    electroWizard->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    // Electro Wizard also spawns a deploy-zap AreaSpell alongside itself now,
    // so the troop isn't necessarily board.getEntities().back() anymore.
    std::shared_ptr<MeleeTroop> electroWizardEntity;
    for (const auto& e : board.getEntities()) {
        auto troop = std::dynamic_pointer_cast<MeleeTroop>(e);
        if (troop) electroWizardEntity = troop;
    }
    REQUIRE(electroWizardEntity != nullptr);

    // Direct-damage attack (no projectile -- the real card is an instant
    // zap): damage and stun both land the same tick. Only one enemy in
    // range, so it takes the full 230, not the split half.
    electroWizardEntity->update(board);

    REQUIRE(enemy->hp == 882); // 1000 - 118
    REQUIRE(enemy->freezeTicks == 5);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.0f));
}

TEST_CASE("Electro Wizard splits its attack across the 2 closest enemies at half damage each", "[card_registry][flying]") {
    Board board;
    auto near = std::make_shared<StationaryCombatant>(1, 5.0f, 5.5f, 1000, 1, 5.0f, 10, 10);  // dist 0.5
    auto far = std::make_shared<StationaryCombatant>(2, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);    // dist 1.0
    spawn(board, near);
    spawn(board, far);

    const CardDefinition* electroWizard = CardRegistry::getInstance().getCard(35);
    electroWizard->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    std::shared_ptr<MeleeTroop> electroWizardEntity;
    for (const auto& e : board.getEntities()) {
        auto troop = std::dynamic_pointer_cast<MeleeTroop>(e);
        if (troop) electroWizardEntity = troop;
    }
    REQUIRE(electroWizardEntity != nullptr);

    electroWizardEntity->update(board);

    REQUIRE(near->hp == 941);  // 1000 - 118/2
    REQUIRE(far->hp == 941);
    REQUIRE(near->freezeTicks == 5); // both stunned
    REQUIRE(far->freezeTicks == 5);
}

TEST_CASE("Electro Wizard's deploy zap damages and stuns enemies in radius the instant it's played", "[card_registry][on_hit][spawn_effect]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 6.0f, 5.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    const CardDefinition* electroWizard = CardRegistry::getInstance().getCard(35);
    electroWizard->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    std::shared_ptr<AreaSpell> deployZap;
    for (const auto& e : board.getEntities()) {
        auto spell = std::dynamic_pointer_cast<AreaSpell>(e);
        if (spell) deployZap = spell;
    }
    REQUIRE(deployZap != nullptr);

    deployZap->update(board); // zero delay: detonates immediately

    REQUIRE(enemy->hp == 882); // 1000 - 118
    REQUIRE(enemy->freezeTicks == 5);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.0f));
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

// ---------------- 2026 roster expansion ----------------

TEST_CASE("Freeze deals no direct damage but fully stuns everyone in radius via the new spellOnHit hook", "[card_registry][spell_on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(107)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);

    REQUIRE(enemy->hp == 1000); // no damage component
    REQUIRE(enemy->freezeTicks == 40); // 4s full stun
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.0f));
}

TEST_CASE("Giant Snowball deals damage and applies a partial slow via spellOnHit", "[card_registry][spell_on_hit]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(100)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);

    REQUIRE(enemy->hp == 821); // 1000 - 179
    REQUIRE(enemy->freezeTicks == 15);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.5f));
}

TEST_CASE("Battle Ram releases 2 Barbarians on death, reusing the Barbarians card's own sourced stats", "[card_registry][death]") {
    Board board;
    const CardDefinition* battleRam = CardRegistry::getInstance().getCard(81);
    REQUIRE(battleRam != nullptr);
    battleRam->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto ram = board.getEntities()[0];
    ram->takeDamage(ram->hp); // dies
    board.cleanDeadEntities();
    board.commitPendingEntities(); // the two released Barbarians become visible

    REQUIRE(board.getEntities().size() == 2);
    for (const auto& e : board.getEntities()) {
        auto barbarian = std::dynamic_pointer_cast<MeleeTroop>(e);
        REQUIRE(barbarian != nullptr);
        REQUIRE(barbarian->name == "Barbarians");
        REQUIRE(barbarian->hp == 691); // matches the standalone Barbarians card's own hp
        REQUIRE(barbarian->team == 0);
    }
}

TEST_CASE("Inferno Dragon's beam damage ramps up like Inferno Tower's, while flying and air-targeting", "[card_registry][ramp][flying]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000000, 1, 5.0f, 10, 10); // dist 1.0 from (5,5)
    spawn(board, enemy);

    const CardDefinition* infernoDragon = CardRegistry::getInstance().getCard(56);
    infernoDragon->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto dragon = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities().back());
    REQUIRE(dragon != nullptr);
    REQUIRE(dragon->isFlying);
    REQUIRE(dragon->targetsAir);

    dragon->update(board); // ticksOnTarget == 0 on first lock: stage 1 (~8.3% of 422)
    REQUIRE(enemy->hp == 1000000 - 35);
}

TEST_CASE("Elixir Collector and the other spawner buildings deal no damage (spawn mechanics not modeled)", "[card_registry][data]") {
    Board board;

    SECTION("Elixir Collector") {
        CardRegistry::getInstance().getCard(99)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        REQUIRE(board.getEntities().back()->hp == 1070);
    }

    SECTION("Tombstone") {
        Board board2;
        CardRegistry::getInstance().getCard(96)->spawnEntity(5.0f, 5.0f, 0, board2);
        board2.commitPendingEntities();
        auto building = std::dynamic_pointer_cast<Building>(board2.getEntities().back());
        REQUIRE(building != nullptr);
        // Never attacks: a nearby enemy takes no damage over several updates.
        auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
        spawn(board2, enemy);
        for (int i = 0; i < 10; ++i) building->update(board2);
        REQUIRE(enemy->hp == 1000);
    }
}

TEST_CASE("Valkyrie's splash hits a second enemy standing near her primary target", "[card_registry][splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 6.0f, 5.0f, 1000, 1, 'P'); // dist 1.0 from (5,5)
    auto nearby = std::make_shared<DummyEntity>(2, 6.0f, 5.5f, 1000, 1, 'N');  // 0.5 from primary, within her 1.5 splash
    spawn(board, primary);
    spawn(board, nearby);

    const CardDefinition* valkyrie = CardRegistry::getInstance().getCard(10);
    REQUIRE(valkyrie != nullptr);
    valkyrie->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    board.getEntities().back()->update(board);

    REQUIRE(primary->hp == 734);  // 1000 - 266
    REQUIRE(nearby->hp == 734);   // caught in the splash, same damage
}

TEST_CASE("Tombstone periodically spawns Skeletons while alive, not just on death", "[card_registry][periodic]") {
    Board board;
    const CardDefinition* tombstone = CardRegistry::getInstance().getCard(96);
    REQUIRE(tombstone != nullptr);
    tombstone->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto building = board.getEntities().back();
    REQUIRE(board.getEntities().size() == 1); // just the Tombstone itself so far

    for (int i = 0; i < 35; ++i) building->update(board); // its periodic interval
    board.commitPendingEntities(); // the spawned Skeletons become visible

    int skeletonCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Skeletons") skeletonCount++;
    }
    REQUIRE(skeletonCount == 2);
}

TEST_CASE("Rage buffs an ally's damage in radius instead of damaging enemies", "[card_registry][spell][buff]") {
    Board board;
    auto ally = std::make_shared<StationaryCombatant>(1, 6.0f, 5.0f, 1000, 0, 1.0f, 10, 10); // dist 1.0, same team
    auto enemy = std::make_shared<StationaryCombatant>(2, 5.0f, 6.0f, 1000, 1, 1.0f, 10, 10); // dist 1.0, other team
    spawn(board, ally);
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(108)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);

    REQUIRE(ally->buffTicksRemaining == 45);
    REQUIRE(ally->hp == 1000);   // buffed, not damaged
    REQUIRE(enemy->hp == 1000);  // Rage never touches enemies at all
}

TEST_CASE("Phoenix revives exactly once on death, then stays dead the second time", "[card_registry][death]") {
    Board board;
    const CardDefinition* phoenix = CardRegistry::getInstance().getCard(59);
    REQUIRE(phoenix != nullptr);
    phoenix->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    REQUIRE(board.getEntities().size() == 1);
    auto firstLife = board.getEntities()[0];
    firstLife->takeDamage(firstLife->hp); // first death
    board.cleanDeadEntities();
    board.commitPendingEntities(); // the revived Phoenix becomes visible

    REQUIRE(board.getEntities().size() == 1);
    auto secondLife = board.getEntities()[0];
    REQUIRE(secondLife->name == "Phoenix");
    REQUIRE(secondLife->id != firstLife->id); // a genuinely new entity, not the same one
    REQUIRE(secondLife->hp == 1052); // full hp, not a fraction

    secondLife->takeDamage(secondLife->hp); // second death
    board.cleanDeadEntities();
    board.commitPendingEntities();

    REQUIRE(board.getEntities().empty()); // no third life -- revive only ever fires once
}

TEST_CASE("Goblin Barrel drops 3 Goblins directly at the target, dealing no separate area damage", "[card_registry][spell][spawn]") {
    Board board;
    CardRegistry::getInstance().getCard(109)->spawnEntity(9.0f, 9.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);
    board.commitPendingEntities();

    int goblinCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Goblins") goblinCount++;
    }
    REQUIRE(goblinCount == 3);
}

TEST_CASE("Graveyard rains Skeletons over its duration, one small batch per tick", "[card_registry][spell][spawn]") {
    Board board;
    CardRegistry::getInstance().getCard(110)->spawnEntity(9.0f, 9.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);
    board.commitPendingEntities();

    int skeletonCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Skeletons") skeletonCount++;
    }
    REQUIRE(skeletonCount == 9); // 9 applications, one Skeleton each
}

TEST_CASE("Royal Delivery deals landing damage and drops a single shielded defender", "[card_registry][spell][spawn]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 10.0f, 9.0f, 1000, 1, 5.0f, 10, 10); // dist 1.0 from (9,9)
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(111)->spawnEntity(9.0f, 9.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);

    while (spell->isAlive()) spell->update(board);
    board.commitPendingEntities();

    REQUIRE(enemy->hp == 562); // 1000 - 438 landing damage

    std::shared_ptr<Entity> defender;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Royal Recruits") defender = e;
    }
    REQUIRE(defender != nullptr);
}

TEST_CASE("Compound cards spawn a primary and a secondary unit that target independently", "[card_registry][compound]") {
    Board board;

    SECTION("Goblin Machine: melee body + ranged turret") {
        CardRegistry::getInstance().getCard(55)->spawnEntity(9.0f, 9.0f, 0, board);
        board.commitPendingEntities();

        REQUIRE(board.getEntities().size() == 2);
        auto body = std::dynamic_pointer_cast<MeleeTroop>(board.getEntities()[0]);
        auto turret = std::dynamic_pointer_cast<RangedTroop>(board.getEntities()[1]);
        REQUIRE(body != nullptr);
        REQUIRE(turret != nullptr);
    }

    SECTION("Goblin Gang: 3 melee Goblins + 3 ranged Spear Goblins") {
        CardRegistry::getInstance().getCard(112)->spawnEntity(9.0f, 9.0f, 0, board);
        board.commitPendingEntities();

        int meleeCount = 0, rangedCount = 0;
        for (const auto& e : board.getEntities()) {
            if (std::dynamic_pointer_cast<MeleeTroop>(e)) meleeCount++;
            if (std::dynamic_pointer_cast<RangedTroop>(e)) rangedCount++;
        }
        REQUIRE(meleeCount == 3);
        REQUIRE(rangedCount == 3);
    }
}

TEST_CASE("Mighty Miner (115) is a Champion with the correct stats, ramp, and ability", "[card_registry][champion]") {
    Board board;
    const CardDefinition* def = CardRegistry::getInstance().getCard(115);
    REQUIRE(def != nullptr);
    REQUIRE_FALSE(def->deployAnywhere); // unlike the regular Miner (52): confirmed no deploy-anywhere

    def->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto miner = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(miner != nullptr);
    REQUIRE(miner->hp == 2250);
    REQUIRE(miner->isChampion);
    // Sourced (Liquipedia version history): 2 seconds per stage transition
    // as of the 2025-01-08 balance patch -- 20 ticks to stage 2, 40 to max.
    REQUIRE(miner->rampMidTick == 20);
    REQUIRE(miner->rampFullTick == 40);
    REQUIRE(miner->abilityElixirCost == Catch::Approx(1.0f));
    REQUIRE(miner->abilityCooldownTicks == 130);
    REQUIRE(miner->abilityEffect != nullptr);
    REQUIRE(miner->abilityCooldownRemaining == 0); // ready immediately at deploy
}

TEST_CASE("CardDefinition::isChampion is set for all 8 Champions and no ordinary card", "[card_registry][champion]") {
    for (int id : { 115, 116, 117, 118, 119, 120, 121, 122 }) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(id);
        REQUIRE(def != nullptr);
        REQUIRE(def->isChampion);
    }
    // A handful of ordinary troops/spells/buildings, none of them Champions.
    for (int id : { 0, 4, 25, 52, 114 }) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(id);
        REQUIRE(def != nullptr);
        REQUIRE_FALSE(def->isChampion);
    }
}

TEST_CASE("countChampions counts how many Champion cards appear in a deck", "[card_registry][champion]") {
    REQUIRE(countChampions({ 0, 1, 2, 3, 4, 5, 6, 7 }) == 0); // no Champion at all
    REQUIRE(countChampions({ 115, 1, 2, 3, 4, 5, 6, 7 }) == 1); // Mighty Miner only
    REQUIRE(countChampions({ 115, 118, 2, 3, 4, 5, 6, 7 }) == 2); // Mighty Miner + Archer Queen: the illegal case
    REQUIRE(countChampions({}) == 0); // empty deck: no crash
    REQUIRE(countChampions({ 9999 }) == 0); // unknown id: ignored, not a crash
}

// ---------------- wiki-research pass: closing the "no sourced stats" gaps ----------------

TEST_CASE("Lava Hound splits into 6 Lava Pups on death", "[card_registry][death]") {
    Board board;
    CardRegistry::getInstance().getCard(91)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto hound = board.getEntities()[0];
    hound->takeDamage(hound->hp);
    board.cleanDeadEntities();
    board.commitPendingEntities();

    int pupCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Lava Pups") pupCount++;
    }
    REQUIRE(pupCount == 6);
}

TEST_CASE("Elixir Golem's full split chain grants elixir at every tier", "[card_registry][death][elixir]") {
    Board board;
    CardRegistry::getInstance().getCard(90)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto golem = board.getEntities()[0];
    golem->takeDamage(golem->hp);
    board.cleanDeadEntities(); // Golem dies -> 2 Golemites + 1.0 elixir to team 1
    board.commitPendingEntities();

    REQUIRE(board.pendingElixirGrant[1] == Catch::Approx(1.0f));
    int golemiteCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Elixir Golemite") golemiteCount++;
    }
    REQUIRE(golemiteCount == 2);

    // Kill one Golemite -> 2 Blobs + another 0.5 elixir to team 1.
    for (const auto& e : board.getEntities()) {
        if (e->name == "Elixir Golemite") { e->takeDamage(e->hp); break; }
    }
    board.cleanDeadEntities();
    board.commitPendingEntities();

    REQUIRE(board.pendingElixirGrant[1] == Catch::Approx(1.5f));
    int blobCount = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Elixir Blob") blobCount++;
    }
    REQUIRE(blobCount == 2);
}

TEST_CASE("Mother Witch's curse arms a Cursed Hog that spawns for HER team when the cursed unit dies", "[card_registry][curse]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 1.0f, 10, 10); // dist 1.0

    CardRegistry::getInstance().getCard(68)->spawnEntity(5.0f, 5.0f, 0, board); // Mother Witch, team 0
    board.addEntity(enemy);
    board.commitPendingEntities();

    // Mother Witch is a RangedSquad card -- her onHit effect (CursedHogOnHit)
    // only lands when her Projectile actually arrives, not the instant she
    // fires. Drive the same update/commit loop GameManager::step() uses
    // until the curse lands.
    for (int i = 0; i < 20 && enemy->curseTicksRemaining == 0; ++i) {
        for (const auto& e : board.getEntities()) {
            if (e->isAlive()) e->update(board);
        }
        board.commitPendingEntities();
    }

    REQUIRE(enemy->curseTicksRemaining > 0);
    REQUIRE(enemy->deathEffect != nullptr);

    enemy->takeDamage(enemy->hp);
    board.cleanDeadEntities();
    board.commitPendingEntities();

    bool foundCursedHog = false;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Cursed Hog") {
            REQUIRE(e->team == 0); // fights for Mother Witch's side, not the victim's
            foundCursedHog = true;
        }
    }
    REQUIRE(foundCursedHog);
}

TEST_CASE("Cannon Cart grounds itself once it drops to 50% hp", "[card_registry][transform]") {
    Board board;
    CardRegistry::getInstance().getCard(69)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    auto cart = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
    REQUIRE(cart != nullptr);
    cart->takeDamage(cart->hp / 2 + 1); // just past half: at-or-below the 0.5 threshold

    cart->update(board);

    REQUIRE(cart->hasTransformed);
    REQUIRE(cart->freezeSlow == Catch::Approx(0.0f)); // grounded
}

TEST_CASE("Goblin Hut only summons while an enemy is within its detection range", "[card_registry][periodic]") {
    Board board;
    CardRegistry::getInstance().getCard(95)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto hut = board.getEntities()[0];

    for (int i = 0; i < 22; ++i) hut->update(board); // one full interval, no enemy anywhere
    board.commitPendingEntities();
    int countNoEnemy = static_cast<int>(board.getEntities().size());
    REQUIRE(countNoEnemy == 1); // just the Hut itself -- no spawn

    auto enemy = std::make_shared<DummyEntity>(50, 6.0f, 5.0f, 100, 1); // dist 1.0, inside 6.0 range
    spawn(board, enemy);
    for (int i = 0; i < 22; ++i) hut->update(board);
    board.commitPendingEntities();

    REQUIRE(static_cast<int>(board.getEntities().size()) > countNoEnemy + 1); // Hut + enemy + at least one Spear Goblin
}

TEST_CASE("Goblin Demolisher transforms into a kamikaze that detonates on a building at <=50% hp", "[card_registry][transform]") {
    Board board;
    CardRegistry::getInstance().getCard(66)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto demolisher = board.getEntities()[0];

    demolisher->takeDamage(demolisher->hp / 2 + 1); // just past half: triggers the transform
    demolisher->update(board);

    REQUIRE_FALSE(demolisher->isAlive());
    board.cleanDeadEntities();
    board.commitPendingEntities();

    std::shared_ptr<Entity> kamikaze;
    for (const auto& e : board.getEntities()) {
        if (e->name == "Kamikaze Goblin Demolisher") kamikaze = e;
    }
    REQUIRE(kamikaze != nullptr);
    REQUIRE(kamikaze->team == 0);

    // The kamikaze form only targets buildings and self-destructs on its
    // first hit -- verify it against a building-shaped target, close
    // enough to attack immediately.
    auto building = std::make_shared<Building>(999, 5.0f, 5.4f, 5000, 1, 'C', 5.0f, 10, 10);
    spawn(board, building);
    kamikaze->update(board);

    REQUIRE(building->hp == 5000 - 404);
    REQUIRE_FALSE(kamikaze->isAlive()); // dieAfterFirstHit
}

TEST_CASE("Mega Knight jumps to a distant target instead of walking, via the real registered card", "[card_registry][jump]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(1, 5.0f, 9.0f, 100000, 1); // dist 4.0 from (5,5)
    spawn(board, target);

    CardRegistry::getInstance().getCard(48)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();

    // Mega Knight also has a deploy-slam spawn effect (a separate
    // AreaSpell entity) -- find the actual troop, not just take .back().
    std::shared_ptr<MeleeTroop> knight;
    for (const auto& e : board.getEntities()) {
        knight = std::dynamic_pointer_cast<MeleeTroop>(e);
        if (knight) break;
    }
    REQUIRE(knight != nullptr);

    knight->update(board);

    REQUIRE(target->hp < 100000); // jump landed a hit immediately, no multi-tick walk needed
}

TEST_CASE("Bowler's piercing line hits a bystander behind the primary target, not just around it", "[card_registry][line_splash]") {
    Board board;
    auto primary = std::make_shared<DummyEntity>(1, 5.0f, 8.0f, 100000, 1);   // dist 3.0 from (5,5)
    auto behindPrimary = std::make_shared<DummyEntity>(2, 5.0f, 9.5f, 100000, 1); // further along the same line
    spawn(board, primary);
    spawn(board, behindPrimary);

    CardRegistry::getInstance().getCard(22)->spawnEntity(5.0f, 5.0f, 0, board); // Bowler
    board.commitPendingEntities();
    auto bowler = board.getEntities().back();

    bowler->update(board); // fires the projectile
    board.commitPendingEntities();
    for (int i = 0; i < 10; ++i) {
        for (const auto& e : board.getEntities()) {
            if (e->isAlive()) e->update(board);
        }
        board.commitPendingEntities();
        if (primary->hp < 100000) break;
    }

    REQUIRE(primary->hp < 100000);
    REQUIRE(behindPrimary->hp < 100000); // caught by the piercing line, not just a circle around primary
}

TEST_CASE("X-Bow can't fire until its slow initial deploy delay elapses", "[card_registry][deploy_delay]") {
    Board board;
    auto enemy = std::make_shared<DummyEntity>(1, 5.0f, 6.0f, 100000, 1); // well within X-Bow's range
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(92)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto xbow = board.getEntities().back();

    for (int i = 0; i < 34; ++i) xbow->update(board); // 3.4s: still not ready (delay is 3.5s/35 ticks)
    REQUIRE(enemy->hp == 100000);

    xbow->update(board); // the 35th tick: ready to fire
    REQUIRE(enemy->hp < 100000);
}
