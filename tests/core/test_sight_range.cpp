#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "Board.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include <vector>

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
