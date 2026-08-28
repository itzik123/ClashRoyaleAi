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

// ---------------- the sight / attack measurement mismatch ----------------
//
// Measured 2026-08-20 by tools/audit/deck_audit.cpp. A lone Musketeer placed
// south of an enemy Princess Tower:
//
//   distance   tower damage taken   Musketeer damage taken
//      6.5           1519                    721  (she dies)
//      7.5           1519                    721  (she dies)
//      8.0           5355                      0  <-- tower NEVER fires back
//     10.5           4704                      0  <-- same
//
// A Princess Tower has 3204 hp, so from 8 tiles a single 4-elixir card removes
// it and starts on the next one without taking a scratch. That is not the real
// game, where a 7.5-range Princess Tower beats a 6.0-range Musketeer every time.
//
// CAUSE: the two gates measure distance differently.
//   attacking  uses effectiveRangeTo() = attackRange + own radius + target
//              radius. Tower vs troop: 7.5 + 1.5 + 0.4 = 9.4.
//   targeting  uses findTarget()'s `dist <= sightRange`, a RAW centre-to-centre
//              distance. Tower: 7.5.
// Between 7.5 and 9.4 the tower is able to attack something it is unable to
// SEE, so it never acquires it and stands idle. The Musketeer stops at her own
// effective range of 6.0 + 0.4 + 1.5 = 7.9, which is inside that band always.
//
// CombatEntity::sightRange's own comment says buildings deliberately get
// sightRange == attackRange because "buildings can't chase, so sight beyond
// attack range would never actually matter". That reasoning is sound and is
// exactly what the mismatch breaks: equal NUMBERS are not equal RANGES when one
// is measured surface-to-surface and the other centre-to-centre.

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
    // 8.0 tiles: outside the tower's raw 7.5 sightRange, well inside the 9.4
    // its attacks actually reach.
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
    // The whole defect in one assertion: she must not siege a tower for free.
    REQUIRE(survivorHp < startHp);
}

TEST_CASE("whatever a unit can attack, it can also see",
          "[targeting][sight][regression]") {
    // The invariant behind the fix, stated directly so it cannot regress
    // quietly: for every entity, whatever it can ATTACK it must also be able to
    // SEE. Anything else is a blind spot in which it stands idle.
    //
    // NOTE (2026-08-28): this case is about the RAW catalogue values, and it is
    // NOT the claim that sight and attack share a measurement convention --
    // they deliberately no longer do. Sight is strict centre-to-centre against
    // sightRange; attack stays surface-to-surface. The invariant survives that
    // split because effectiveSightWith floors sight at the unit's own attack
    // reach -- see the centre-to-centre section at the bottom of this file.
    //
    // Checked here for the shape that actually occurs -- a tower (large radius,
    // sightRange == attackRange) against a troop -- because that is the pairing
    // where the two conventions diverge most.
    Board board;
    GameManager game(DECK, DECK);
    for (const auto& e : game.getBoard().getEntities()) {
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!combat || !e->isTower()) continue;
        // A tower's sight must cover at least everything its attacks reach.
        INFO("tower at (" << e->position.x << ", " << e->position.y << ")");
        REQUIRE(combat->sightRange >= combat->getAttackRange());
    }
}

// ---------------- the sight-range catalog ----------------
//
// Sight range is not shown on the in-game card info screens; these values come
// from a professional player's audit and the community stats breakdown behind
// it. Rule A of that audit ("a unit can ONLY acquire a target inside its own
// sight range") was already implemented here -- 58 per-card withSightRange
// calls plus effectiveSightTo -- but NOTHING pinned the numbers, so a registry
// edit could rebalance every aggro radius in the game silently.
//
// Read off the SPAWNED ENTITY rather than the registry literal, so this also
// covers CardFactories::applyCardMetadata actually copying the value across --
// a registry constant nothing transfers would pass a literal-only check.

namespace {

float spawnedSightRange(Board& board, int cardId) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    REQUIRE(card != nullptr);
    card->spawnEntity(9.0f, 10.0f, 0, board);
    board.commitPendingEntities();
    // Exact cardId first: a multi-spawn card (Goblin Giant and its carried
    // Spear Goblins) puts more than one CombatEntity on the board, and only
    // one of them is the card being asked about.
    for (const auto& e : board.getEntities()) {
        if (e->cardId != cardId) continue;
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        if (combat) return combat->sightRange;
    }
    // An EVOLUTION entry's spawnEntity deliberately spawns the BASE form (see
    // CardDefinition::spawnEvolvedEntity's comment), so the entity carries the
    // BASE card's id and the filter above can never match -- it returned -1 for
    // all 41 Evolution ids, which reads as "no sight range" rather than as
    // "asked the wrong question".
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
    // The Log (33) and Fireball (7) are spells -- no sight range to carry.
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

// Two entries from the catalog that are NOT in the deck, chosen because they
// bracket the range and because both are cards whose sight is famously longer
// than their reach -- the property the whole mechanic exists for.
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

// THE INVARIANT, across the whole registry.
//
// Sight and attack range must be measured the same way -- surface to surface --
// so that effective sight >= effective attack range and nothing can ever attack
// what it cannot see. Two conventions for one geometric question is what let a
// Musketeer destroy a 3204 hp Princess Tower from 8 tiles taking ZERO damage
// (measured 2026-08-20, the case at the top of this file).
//
// Asserted per-card on the spawned entity, because the radii that turn a range
// into an EFFECTIVE range are per-entity.
TEST_CASE("no card can attack further than it can see", "[sight][invariant]") {
    // Collects EVERY violation rather than failing on the first. A one-at-a-time
    // assertion turns an audit into N build cycles, and worse, makes it look
    // like there was only ever one problem.
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
    // Guards the guard: a filter bug that skipped every card would otherwise
    // make this pass vacuously.
    REQUIRE(checked > 100);
}

// ---------------- the catalog, COMPLETE ----------------
//
// The catalog cases above pin 8 cards. That leaves the other 140 unpinned, and
// for most of them the invariant below cannot help either: it only fires when
// sightRange < attackRange, and 130 of the 148 non-spell cards have an
// attackRange of 5.5 or less. Every one of the 102 cards sitting on
// CardStats::sightRange's 5.5 default is inside that blind spot by definition.
//
// So the gap is not "a wrong number" -- commit 1b17844 cross-referenced the
// whole registry against the source table and the default IS the sourced value
// for those 102 ("the main standard for most cards"). The gap is that a
// DECISION and an OMISSION are indistinguishable in the code: a card added
// without a sight range silently inherits 5.5 and every existing test passes,
// unless it happens to out-range it. That is how Bomb Tower and Three
// Musketeers survived until 2026-08-21 -- both were caught only because their
// attackRange was 6.0.
//
// This table closes that by recording all 148 as an explicit decision. It is
// generated from the engine (tools/audit/sight_audit.cpp), read off the spawned
// entity, so it also covers CardFactories::applyCardMetadata transferring the
// value -- a registry literal nothing copies across would pass a source-only
// check.
//
// WHEN THIS FAILS after you add a card: add the row. Choosing 5.5 is a fine
// answer; leaving it unstated is what this exists to prevent. When it fails
// after you CHANGE a value, that is an aggro-radius rebalance -- gameplay-
// affecting, and every win rate in CLAUDE.md predates it.

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

// The load-bearing half. Without it the table is only as complete as whoever
// last regenerated it, and a new card omitted from BOTH the registry override
// and this table would be invisible in exactly the way described above.
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
    // Guards the guard, same reason as the invariant sweep below: a filter bug
    // that skipped every card would make this pass vacuously. A LOWER BOUND,
    // not an equality -- pinning the exact count would fail on a new card that
    // was correctly catalogued, turning the completeness check above into a
    // second thing to update for no benefit.
    REQUIRE(nonSpell > 100);
}

// Evolutions spawn through a SECOND path (spawnEvolvedEntity), which the table
// above never exercises -- it uses spawnEntity, the base form. Measured 0
// mismatches today; this keeps it that way, since an Evolution that quietly
// changed aggro radius on evolving would be invisible everywhere else.
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


// ---------------- target selection: nearest building, strict sight ----------
//
// TWO SEPARATE RULES, both fixed 2026-08-28, and the second one is the defect
// actually reported.
//
// 1. SIGHT IS STRICT CENTRE-TO-CENTRE. effectiveSightWith returned
//    `sightRange + myRadius + radiusOf(target)`, inflating a Hog Rider's
//    catalogued 9.5 to 10.9 against a Cannon. Radii answer a hitbox question;
//    the published sight ranges are centre-to-centre. (One floor survives:
//    sight is floored at the unit's own attack reach for a card whose raw
//    sightRange already covers its raw attackRange -- otherwise a Princess
//    Tower goes blind to a Musketeer inside its own attack range, the
//    free-siege defect this file opens with.)
//
// 2. NEAREST BUILDING WINS, TOWERS INCLUDED. BuildingTargeter::findTarget used
//    an `else if` that made every non-tower building beat every tower at ANY
//    distance. Measured on replays/hog_test_1.json: at the tick it turned, the
//    Hog was 7.714 tiles from the Princess Tower and 9.160 from the Cannon --
//    it abandoned the nearer objective for the further one and stayed wrong
//    for seven more ticks. Fixing sight alone did NOT fix this; it only moved
//    the distance at which the wrong preference fired.
//
// The old code's own comment described tier-two as modelling the real game.
// It does not: a building-targeter walks at whichever BUILDING is closest, and
// Crown Towers are buildings.

namespace {

// Pure-x separation at a fixed y, so "distance to the Cannon" and "how far
// off-lane it sits" are one number. Returns the attacker's net x drift: a
// negative value means it left its lane for the Cannon, ~0 means it kept
// walking at its tower.
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

// y = 12 puts the attacker exactly 6.0 tiles from the Princess Tower at
// (14, 6), so a Cannon's separation can be placed either side of 6.0 while
// staying well inside every catalogued sight range. That is what isolates
// rule 2 from rule 1.
constexpr float LANE_Y = 12.0f;
constexpr float TOWER_DIST_AT_LANE_Y = 6.0f;

} // namespace

TEST_CASE("a Cannon FURTHER than the tower never steals a Hog Rider off its lane",
          "[targeting][sight][regression][nearest_building]") {
    // THE REPORTED BUG. 7.0 > 6.0, so the tower is the nearer building, but
    // 7.0 is comfortably inside the Hog's 9.5 sight -- which is precisely the
    // band the old `else if` got wrong. Fails on the pre-fix engine.
    const float drift = driftAt(15, LANE_Y, 7.0f);
    INFO("Cannon 7.0 away, Princess Tower " << TOWER_DIST_AT_LANE_Y
         << " away; net x drift " << drift);
    REQUIRE(drift == Catch::Approx(0.0f).margin(0.05f));
}

TEST_CASE("a Cannon NEARER than the tower does pull a Hog Rider",
          "[targeting][nearest_building]") {
    // The other side of the same rule: strict must not mean inert. Without
    // this, deleting the Cannon pull outright would pass the case above.
    const float drift = driftAt(15, LANE_Y, 4.0f);
    INFO("Cannon 4.0 away, Princess Tower " << TOWER_DIST_AT_LANE_Y
         << " away; net x drift " << drift);
    REQUIRE(drift < -0.5f);
}

TEST_CASE("the nearest-building rule holds for every catalogued sight range",
          "[targeting][nearest_building][systemic]") {
    // Parameterised over building-targeters spanning four catalogue values, so
    // this is a property of the RULE and not of one card. Every one of these
    // sees past 6.0, so in each case both buildings are in sight and only
    // distance decides.
    const int card = GENERATE(15,   // Hog Rider   9.5
                              45,   // Balloon     7.7
                              2,    // Giant       7.5
                              40);  // Ice Golem   7.0
    const float sight = catalogSight(card);
    REQUIRE(sight > TOWER_DIST_AT_LANE_Y);
    INFO("card id " << card << ", catalogued sight " << sight);

    REQUIRE(driftAt(card, LANE_Y, 7.0f) == Catch::Approx(0.0f).margin(0.05f));
    REQUIRE(driftAt(card, LANE_Y, 4.0f) < -0.3f);
}

TEST_CASE("sight still bounds acquisition at the raw catalogued range",
          "[targeting][sight][centre_to_centre]") {
    // Rule 1, isolated from rule 2. At y = 20 the tower is 14.0 tiles away, so
    // the Cannon is the nearest building at every separation tested here and
    // only SIGHT can refuse it. 10.0 is outside the Hog's 9.5 and inside the
    // 10.9 the old radius-inflated formula produced.
    REQUIRE(driftAt(15, 20.0f, 10.0f) == Catch::Approx(0.0f).margin(0.05f));
    REQUIRE(driftAt(15, 20.0f, 9.0f) < -0.3f);
}

TEST_CASE("sight is never shorter than the unit's own attack reach",
          "[targeting][sight][invariant][regression]") {
    // The floor, stated as the property it protects: a Princess Tower's sight
    // is 7.5 while its attacks reach 7.5 + 1.5 + 0.4 = 9.4, so a strict
    // centre-to-centre reading alone would recreate the free-siege defect this
    // file opens with -- a Musketeer at 8.0 removing a tower that never fires
    // back.
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
