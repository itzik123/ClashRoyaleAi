#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "ClashEnv.h"   // getAllCardIds()
#include <vector>
#include <string>

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

TEST_CASE("sight is measured the same way as attack range, not centre-to-centre",
          "[targeting][sight][regression]") {
    // The invariant behind the fix, stated directly so it cannot regress
    // quietly: for every entity, whatever it can ATTACK it must also be able to
    // SEE. Anything else is a blind spot in which it stands idle.
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
    for (const auto& e : board.getEntities()) {
        if (e->cardId != cardId) continue;
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
