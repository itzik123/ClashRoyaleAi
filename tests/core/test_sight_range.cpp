#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Building.h"
#include "CardStats.h"   // DEPLOY_TIME_TICKS
#include "ClashEnv.h"   // getAllCardIds()
#include <vector>
#include <string>
#include <set>

// --- sight vs attack reach ---
// Attack reach is measured surface to surface (effectiveRangeTo: tower vs troop
// 7.5 + 1.5 + 0.4 = 9.4), sight centre to centre (7.5). Without a floor, a
// Princess Tower could attack a Musketeer standing at 8.0 but never see her,
// and she would destroy it without taking damage; in the real game the
// 7.5-range tower beats the 6.0-range Musketeer every time.

namespace {
constexpr int MUSKETEER = 6;
const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };

int enemyTowerHp(GameManager& game) {
    int t = 0;
    for (const auto& e : game.getBoard().getEntities())
        if (e->isAlive() && e->isTower() && e->team == 1) t += e->hp;
    return t;
}
} // namespace

TEST_CASE("a Princess Tower fights back against a unit inside its attack range",
          "[targeting][sight][regression]") {
    // 8.0 tiles: beyond the tower's raw 7.5 sight, well inside its 9.4 reach.
    GameManager game(DECK, DECK);
    CardRegistry::getInstance().getCard(MUSKETEER)->spawnEntity(4.0f, 19.0f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();

    std::shared_ptr<Entity> musketeer;
    for (const auto& e : game.getBoard().getEntities())
        if (e->cardId == MUSKETEER) musketeer = e;
    REQUIRE(musketeer != nullptr);
    const int startHp = musketeer->hp;
    const int towerBefore = enemyTowerHp(game);

    for (int i = 0; i < 300; ++i) game.step();

    int survivorHp = 0;
    for (const auto& e : game.getBoard().getEntities())
        if (e->id == musketeer->id && e->isAlive()) survivorHp = e->hp;

    INFO("tower damage dealt to the enemy: " << (towerBefore - enemyTowerHp(game)));
    INFO("Musketeer hp " << startHp << " -> " << survivorHp);
    // She must not siege a tower for free.
    REQUIRE(survivorHp < startHp);
}

TEST_CASE("whatever a unit can attack, it can also see",
          "[targeting][sight][regression]") {
    // For every entity, whatever it can attack it can see. Sight (centre to
    // centre) and attack (surface to surface) use different conventions on
    // purpose; effectiveSightWith's floor keeps the invariant (see the
    // centre-to-centre section below). Checked for towers against troops, where
    // the conventions diverge most.
    Board board;
    GameManager game(DECK, DECK);
    for (const auto& e : game.getBoard().getEntities()) {
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!combat || !e->isTower()) continue;
        // A tower's sight covers everything its attacks reach.
        INFO("tower at (" << e->position.x << ", " << e->position.y << ")");
        REQUIRE(combat->sightRange >= combat->getAttackRange());
    }
}

// --- the sight-range catalogue ---
// Not shown in game; from a professional player's audit and the community stats
// behind it. Read off the spawned entity, so this also checks that
// applyCardMetadata copies the value.

namespace {

float spawnedSightRange(Board& board, int cardId) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    REQUIRE(card != nullptr);
    card->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();
    // Exact card id first: a multi-spawn card (Goblin Giant and its Spear
    // Goblins) puts several CombatEntities on the board.
    for (const auto& e : board.getEntities()) {
        if (e->cardId != cardId) continue;
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (combat) return combat->sightRange;
    }
    // An Evolution entry's spawnEntity spawns the base form, which carries the
    // base card's id, so the filter above would never match.
    for (const auto& e : board.getEntities()) {
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (combat) return combat->sightRange;
    }
    return -1.0f;
}

} // namespace

TEST_CASE("every DEFAULT_DECK card carries its catalogued sight range",
          "[sight][registry][catalog]") {
    struct Row { int cardId; const char* name; float sight; };
    // The Log (33) and Fireball (7) are spells: no sight range.
    const Row rows[] = {
        { 15, "Hog Rider",  9.5f },   // building-targeter, the deck's win condition
        {  6, "Musketeer",  6.0f },
        { 25, "Cannon",     5.5f },   // building; sight == attackRange by design
        { 40, "Ice Golem",  7.0f },   // building-targeter
        { 24, "Skeletons",  5.5f },   // the catalog's stated standard
        { 72, "Ice Spirit", 5.5f },
    };
    for (const auto& r : rows) {
        Board board;
        INFO(r.name << " (card id " << r.cardId << ")");
        REQUIRE(spawnedSightRange(board, r.cardId) == Catch::Approx(r.sight));
    }
}

// Two cards outside the deck that bracket the range, both with sight well
// beyond their reach.
TEST_CASE("catalogued outliers keep their sight range too", "[sight][registry][catalog]") {
    struct Row { int cardId; const char* name; float sight; };
    const Row rows[] = {
        {  2, "Giant",     7.5f },
        { 13, "P.E.K.K.A.", 5.0f },
    };
    for (const auto& r : rows) {
        Board board;
        INFO(r.name << " (card id " << r.cardId << ")");
        REQUIRE(spawnedSightRange(board, r.cardId) == Catch::Approx(r.sight));
    }
}

// The invariant across the whole registry, per spawned entity (the radii that
// make a reach effective are per entity): nothing can attack what it cannot
// see.
TEST_CASE("no card can attack further than it can see", "[sight][invariant]") {
    // Collects every violation rather than stopping at the first.
    int checked = 0;
    std::string offenders;
    for (int cardId : getAllCardIds()) {
        const auto* card = CardRegistry::getInstance().getCard(cardId);
        if (!card || card->isSpell) continue;

        Board board;
        card->spawnEntity(9.0f, 10.0f, 0, board);
        board.commitPendingEntities();
        for (const auto& e : board.getEntities()) {
            auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!combat) continue;
            if (combat->sightRange < combat->getAttackRange()) {
                offenders += "\n  card " + std::to_string(cardId) + " " + card->name
                           + ": sight " + std::to_string(combat->sightRange)
                           + " < attackRange " + std::to_string(combat->getAttackRange());
            }
            checked++;
        }
    }
    INFO("cards that can attack what they cannot see:" << offenders);
    REQUIRE(offenders.empty());
    // Guards the guard: a filter bug that skipped every card would pass
    // vacuously.
    REQUIRE(checked > 100);
}

// --- the catalogue, complete ---
// The invariant above only fires when sightRange < attackRange, and most cards
// (every one on the 5.5 default) sit inside that blind spot, so a decision and
// an omission look the same in code. This table records every non-spell card as
// an explicit decision. Generated from the engine
// (tools/audit/sight_audit.cpp), off the spawned entity.
//
// When this fails after adding a card, add the row; 5.5 is a fine answer. When
// it fails after changing a value, that is an aggro-radius rebalance, which is
// gameplay-affecting.

namespace {
struct SightRow { int cardId; const char* name; float sight; };

const SightRow SIGHT_CATALOG[] = {
        {   0, "Knight", 5.5f },
        {   1, "Archers", 5.5f },
        {   2, "Giant", 7.5f },
        {   4, "Goblins", 5.5f },
        {   5, "Mini PEKKA", 5.5f },
        {   6, "Musketeer", 6.0f },
        {   8, "Barbarians", 5.5f },
        {   9, "Bomber", 5.5f },
        {  10, "Valkyrie", 5.5f },
        {  11, "Wizard", 5.5f },
        {  12, "Skeleton Army", 5.5f },
        {  13, "P.E.K.K.A.", 5.0f },
        {  14, "Prince", 5.5f },
        {  15, "Hog Rider", 9.5f },
        {  17, "Elite Barbarians", 5.5f },
        {  18, "Royal Giant", 7.5f },
        {  19, "Golem", 7.0f },
        {  20, "Dart Goblin", 7.5f },
        {  21, "Lumberjack", 5.5f },
        {  22, "Bowler", 4.0f },
        {  23, "Spear Goblins", 5.5f },
        {  24, "Skeletons", 5.5f },
        {  25, "Cannon", 5.5f },
        {  26, "Tesla", 5.5f },
        {  27, "Bomb Tower", 6.0f },
        {  28, "Inferno Tower", 6.0f },
        {  34, "Ice Wizard", 5.5f },
        {  35, "Electro Wizard", 5.5f },
        {  36, "Executioner", 5.5f },
        {  39, "Giant Skeleton", 5.0f },
        {  40, "Ice Golem", 7.0f },
        {  41, "Minions", 5.5f },
        {  42, "Minion Horde", 5.5f },
        {  43, "Mega Minion", 5.5f },
        {  44, "Baby Dragon", 5.5f },
        {  45, "Balloon", 7.7f },
        {  46, "Dark Prince", 5.5f },
        {  47, "Royal Ghost", 5.5f },
        {  48, "Mega Knight", 5.5f },
        {  49, "Battle Healer", 5.5f },
        {  50, "Bandit", 6.0f },
        {  51, "Berserker", 5.5f },
        {  52, "Miner", 5.5f },
        {  53, "Fisherman", 7.5f },
        {  54, "Ronin", 5.5f },
        {  55, "Goblin Machine", 5.5f },
        {  56, "Inferno Dragon", 5.5f },
        {  57, "Electro Dragon", 5.5f },
        {  58, "Night Witch", 5.5f },
        {  59, "Phoenix", 5.5f },
        {  60, "Sparky", 5.0f },
        {  61, "Princess", 9.5f },
        {  62, "Hunter", 5.5f },
        {  63, "Magic Archer", 7.5f },
        {  64, "Firecracker", 8.5f },
        {  65, "Skeleton Dragons", 5.5f },
        {  66, "Goblin Demolisher", 5.5f },
        {  67, "Flying Machine", 6.0f },
        {  68, "Mother Witch", 5.5f },
        {  69, "Cannon Cart", 6.0f },
        {  70, "Furnace", 5.5f },
        {  71, "Witch", 5.5f },
        {  72, "Ice Spirit", 5.5f },
        {  73, "Fire Spirit", 5.5f },
        {  74, "Heal Spirit", 5.5f },
        {  75, "Electro Spirit", 5.5f },
        {  76, "Guards", 5.5f },
        {  77, "Royal Recruits", 5.5f },
        {  78, "Bats", 5.5f },
        {  79, "Zappies", 5.0f },
        {  80, "Three Musketeers", 6.0f },
        {  81, "Battle Ram", 5.5f },
        {  82, "Royal Hogs", 9.5f },
        {  83, "Wall Breakers", 7.0f },
        {  84, "Electro Giant", 5.5f },
        {  85, "Suspicious Bush", 5.5f },
        {  86, "Rune Giant", 5.5f },
        {  87, "Ram Rider", 7.5f },
        {  88, "Goblin Giant", 7.5f },
        {  89, "Skeleton Barrel", 7.7f },
        {  90, "Elixir Golem", 7.5f },
        {  91, "Lava Hound", 5.5f },
        {  92, "X-Bow", 11.5f },
        {  93, "Mortar", 11.5f },
        {  94, "Barbarian Hut", 5.5f },
        {  95, "Goblin Hut", 5.5f },
        {  96, "Tombstone", 5.5f },
        {  97, "Goblin Cage", 5.5f },
        {  98, "Goblin Drill", 5.5f },
        {  99, "Elixir Collector", 5.5f },
        { 112, "Goblin Gang", 5.5f },
        { 113, "Rascals", 5.5f },
        { 115, "Mighty Miner", 5.5f },
        { 116, "Golden Knight", 5.5f },
        { 117, "Skeleton King", 5.5f },
        { 118, "Archer Queen", 5.5f },
        { 119, "Monk", 5.5f },
        { 120, "Little Prince", 5.5f },
        { 121, "Goblinstein", 5.5f },
        { 122, "Boss Bandit", 5.5f },
        { 123, "Wall Breakers", 7.0f },
        { 125, "Skeletons", 5.5f },
        { 126, "Bats", 5.5f },
        { 127, "Bomber", 5.5f },
        { 128, "Archers", 5.5f },
        { 129, "Cannon", 5.5f },
        { 130, "Firecracker", 8.5f },
        { 131, "Dart Goblin", 7.5f },
        { 133, "Skeleton Army", 5.5f },
        { 134, "Skeleton Barrel", 7.7f },
        { 135, "Knight", 5.5f },
        { 136, "Royal Ghost", 5.5f },
        { 137, "Baby Dragon", 5.5f },
        { 138, "Furnace", 5.5f },
        { 139, "Goblin Cage", 5.5f },
        { 140, "Musketeer", 6.0f },
        { 141, "Wizard", 5.5f },
        { 142, "Witch", 5.5f },
        { 143, "Royal Giant", 7.5f },
        { 144, "Ice Spirit", 5.5f },
        { 145, "Princess", 9.5f },
        { 146, "Hunter", 5.5f },
        { 147, "Valkyrie", 5.5f },
        { 148, "P.E.K.K.A.", 5.0f },
        { 149, "Minion Horde", 5.5f },
        { 150, "Royal Recruits", 5.5f },
        { 151, "Electro Dragon", 5.5f },
        { 152, "Mortar", 11.5f },
        { 153, "Goblin Drill", 5.5f },
        { 154, "Tesla", 5.5f },
        { 155, "Barbarians", 5.5f },
        { 156, "Lumberjack", 5.5f },
        { 157, "Executioner", 5.5f },
        { 159, "Goblin Giant", 7.5f },
        { 160, "Mega Knight", 5.5f },
        { 161, "Battle Ram", 5.5f },
        { 162, "Royal Hogs", 9.5f },
        { 163, "Inferno Dragon", 5.5f },
        { 165, "Spirit Empress", 5.5f },
        { 166, "Hero Knight", 5.5f },
        { 167, "Hero Wizard", 5.5f },
        { 168, "Hero Musketeer", 6.0f },
        { 169, "Hero Giant", 7.5f },
        { 170, "Hero Mini P.E.K.K.A.", 5.5f },
        { 171, "Hero Magic Archer", 7.5f },
        { 172, "Hero Goblins", 5.5f },
        { 173, "Hero Mega Minion", 5.5f },
        { 175, "Hero Ice Golem", 7.0f },
};
} // namespace

TEST_CASE("every non-spell card carries its catalogued sight range",
          "[sight][registry][catalog]") {
    for (const auto& r : SIGHT_CATALOG) {
        Board board;
        INFO(r.name << " (card id " << r.cardId << ")");
        REQUIRE(spawnedSightRange(board, r.cardId) == Catch::Approx(r.sight));
    }
}

// The load-bearing half: without it the table is only as complete as its last
// regeneration.
TEST_CASE("no registered card may silently inherit the default sight range",
          "[sight][registry][catalog]") {
    std::set<int> catalogued;
    for (const auto& r : SIGHT_CATALOG) catalogued.insert(r.cardId);

    std::string missing;
    int nonSpell = 0;
    for (const auto& [cardId, def] : CardRegistry::getInstance().getAllCards()) {
        if (def.isSpell) continue;
        nonSpell++;
        if (!catalogued.count(cardId))
            missing += "\n  card " + std::to_string(cardId) + " " + def.name
                     + " -- decide its sight range and add a row";
    }
    INFO("cards with no catalogued sight range:" << missing);
    REQUIRE(missing.empty());
    // Guards the guard. A lower bound, not an exact count, so a correctly
    // catalogued new card does not also need this updated.
    REQUIRE(nonSpell > 100);
}

// Evolutions spawn through spawnEvolvedEntity, which the table never exercises;
// an Evolution that changed aggro radius on evolving would be invisible
// elsewhere.
TEST_CASE("an Evolution's evolved form keeps its base form's sight range",
          "[sight][registry][catalog][evolution]") {
    int checked = 0;
    std::string offenders;
    for (const auto& [cardId, def] : CardRegistry::getInstance().getAllCards()) {
        if (!def.isEvolution || !def.spawnEvolvedEntity || def.isSpell) continue;
        Board baseBoard, evoBoard;
        float base = spawnedSightRange(baseBoard, cardId);

        def.spawnEvolvedEntity(9.0f, 10.0f, 0, evoBoard);
        evoBoard.commitPendingEntities();
        float evolved = -1.0f;
        for (const auto& e : evoBoard.getEntities())
            if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) { evolved = c->sightRange; break; }

        if (evolved != Catch::Approx(base))
            offenders += "\n  card " + std::to_string(cardId) + " " + def.name
                       + ": base " + std::to_string(base)
                       + " vs evolved " + std::to_string(evolved);
        checked++;
    }
    INFO("Evolutions whose evolved form changes sight range:" << offenders);
    REQUIRE(offenders.empty());
    REQUIRE(checked > 10);
}


// --- target selection: nearest building, strict sight ---
// 1. Sight is strict centre-to-centre: a Hog Rider's 9.5 means 9.5 against a
//    Cannon. One floor survives: sight never falls below the unit's own attack
//    reach, for cards whose raw sight covers their raw range (otherwise the
//    free-siege case at the top returns).
//
// 2. The nearest building wins, Crown Towers included: a building-targeter
//    walks at whichever building is closest.

namespace {

// Pure-x separation at a fixed y, so distance to the Cannon and distance
// off-lane are one number. Returns the attacker's net x drift: negative means
// it left its lane for the Cannon, ~0 that it kept walking at its tower.
float driftAt(int attackerCard, float attackerY, float separation) {
    GameManager game(DECK, DECK);
    Board& board = game.getBoard();
    CardRegistry::getInstance().getCard(attackerCard)
        ->spawnEntity(14.0f, attackerY, 1, board);
    CardRegistry::getInstance().getCard(25)                    // Cannon
        ->spawnEntity(14.0f - separation, attackerY, 0, board);
    board.commitPendingEntities();
    std::shared_ptr<Entity> attacker;
    for (const auto& e : board.getEntities())
        if (e->cardId == attackerCard) attacker = e;
    REQUIRE(attacker != nullptr);
    const float x0 = attacker->position.x;
    for (int i = 0; i < DEPLOY_TIME_TICKS + 20; ++i) game.step();
    return attacker->position.x - x0;
}

float catalogSight(int cardId) {
    GameManager probe(DECK, DECK);
    CardRegistry::getInstance().getCard(cardId)
        ->spawnEntity(14.0f, 12.0f, 1, probe.getBoard());
    probe.getBoard().commitPendingEntities();
    for (const auto& e : probe.getBoard().getEntities()) {
        if (e->cardId != cardId) continue;
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) return c->sightRange;
    }
    return -1.0f;
}

// y = 12 puts the attacker 6.0 tiles from the Princess Tower at (14, 6), so the
// Cannon can be placed either side of 6.0 while inside every catalogued sight:
// this isolates rule 2 from rule 1.
constexpr float LANE_Y = 12.0f;
constexpr float TOWER_DIST_AT_LANE_Y = 6.0f;

} // namespace

TEST_CASE("a Cannon FURTHER than the tower never steals a Hog Rider off its lane",
          "[targeting][sight][regression][nearest_building]") {
    // 7.0 > 6.0, so the tower is nearer, yet 7.0 is inside the Hog's 9.5 sight:
    // the tower must win.
    const float drift = driftAt(15, LANE_Y, 7.0f);
    INFO("Cannon 7.0 away, Princess Tower " << TOWER_DIST_AT_LANE_Y
         << " away; net x drift " << drift);
    REQUIRE(drift == Catch::Approx(0.0f).margin(0.05f));
}

TEST_CASE("a Cannon NEARER than the tower does pull a Hog Rider",
          "[targeting][nearest_building]") {
    // The other side: strict must not mean inert, or deleting the Cannon pull
    // would pass the case above.
    const float drift = driftAt(15, LANE_Y, 3.0f);
    INFO("Cannon 3.0 away, Princess Tower " << TOWER_DIST_AT_LANE_Y
         << " away; net x drift " << drift);
    REQUIRE(drift < -0.5f);
}

TEST_CASE("the nearest-building rule holds for every catalogued sight range",
          "[targeting][nearest_building][systemic]") {
    // Over four building-targeters and catalogue values, so this is a property
    // of the rule. All see past 6.0, so both buildings are in sight and only
    // distance decides.
    const int card = GENERATE(15,   // Hog Rider   9.5
                              45,   // Balloon     7.7
                              2,    // Giant       7.5
                              40);  // Ice Golem   7.0
    const float sight = catalogSight(card);
    REQUIRE(sight > TOWER_DIST_AT_LANE_Y);
    INFO("card id " << card << ", catalogued sight " << sight);

    REQUIRE(driftAt(card, LANE_Y, 7.0f) == Catch::Approx(0.0f).margin(0.05f));
    // 3.0 centre => rank 2.0 against the tower's 6.0 - 3.3 = 2.7.
    REQUIRE(driftAt(card, LANE_Y, 3.0f) < -0.3f);
}

TEST_CASE("sight still bounds acquisition at the raw catalogued range",
          "[targeting][sight][centre_to_centre]") {
    // Rule 1 alone. At y = 20 the tower is 14.0 away, so the Cannon is always
    // the nearest building and only sight can refuse it. 10.0 is outside the
    // Hog's 9.5. The gate uses footprint distance, so a Cannon (radius 1.0) is
    // refused past 10.5 centre.
    REQUIRE(driftAt(15, 20.0f, 11.0f) == Catch::Approx(0.0f).margin(0.05f));
    REQUIRE(driftAt(15, 20.0f, 9.0f) < -0.3f);
}

TEST_CASE("sight is never shorter than the unit's own attack reach",
          "[targeting][sight][invariant][regression]") {
    // The floor: a Princess Tower's sight is 7.5 while its attacks reach 9.4,
    // so strict centre-to-centre alone would let a Musketeer at 8.0 remove a
    // tower that never fires back.
    GameManager game(DECK, DECK);
    CardRegistry::getInstance().getCard(MUSKETEER)
        ->spawnEntity(4.0f, 19.0f, 0, game.getBoard());
    game.getBoard().commitPendingEntities();
    std::shared_ptr<Entity> musketeer;
    for (const auto& e : game.getBoard().getEntities())
        if (e->cardId == MUSKETEER) musketeer = e;
    REQUIRE(musketeer != nullptr);
    const int startHp = musketeer->hp;

    for (int i = 0; i < 300; ++i) game.step();

    int survivorHp = 0;
    for (const auto& e : game.getBoard().getEntities())
        if (e->id == musketeer->id && e->isAlive()) survivorHp = e->hp;
    INFO("Musketeer hp " << startHp << " -> " << survivorHp);
    REQUIRE(survivorHp < startHp);
}
