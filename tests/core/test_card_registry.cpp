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
    REQUIRE(enemy->hp == 832); // 1000 - 168
    REQUIRE(axe->isAlive()); // still out on its return trip, not dead after one hit

    // Matches the real GameManager::step() contract (only ever calls
    // update() on entities still isAlive()) -- Projectile, like AreaSpell,
    // has no internal guard against being updated again after it dies.
    while (axe->isAlive()) axe->update(board);
    REQUIRE(enemy->hp == 664); // 1000 - 168*2
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
    REQUIRE(enemy->hp == 100000 - 122 * 3); // all 3 volleys landed
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

    REQUIRE(enemy->hp == 770); // 1000 - 230
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

    REQUIRE(near->hp == 885);  // 1000 - 230/2
    REQUIRE(far->hp == 885);
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

    REQUIRE(enemy->hp == 808); // 1000 - 192
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
