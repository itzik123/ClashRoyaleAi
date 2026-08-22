// What a building-targeter's sightRange actually BUYS the defender.
//
// BuildingTargeter::findTarget diverts to "closest non-tower building within
// effectiveSightTo(it), else the lane tower". So for this archetype sightRange
// IS the Cannon-pull range -- the distance at which a defensive building drags
// a win condition off the tower. That is the interaction tactics.py's whole
// Cannon rule is built on.
//
// Measures, per card: the greatest centre-to-centre distance at which a Cannon
// still takes aggro, and the tower HP that pull is worth over a full push.

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "BuildingTargeter.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

namespace {
const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };
constexpr int CANNON = 25;

int team0TowerHp(GameManager& g) {
    int t = 0;
    for (const auto& e : g.getBoard().getEntities())
        if (e->isAlive() && e->isTower() && e->team == 0) t += e->hp;
    return t;
}

// Enemy (team 1) building-targeter walks down the left lane at the bridge.
// Cannon sits `gap` tiles closer to our tower, same column.
// Cannon offset LATERALLY off the lane, so only sight range decides whether
// the unit diverts -- an in-lane Cannon is walked into eventually no matter
// what sightRange is, which would measure the path, not the aggro radius.
//
// The pull is read as TRAJECTORY DEVIATION against a no-Cannon control, not as
// "the Cannon lost hp": Building::update decays hp every 10 ticks on its own,
// so a damage test reports a pull at every offset, saturating the sweep.
// Deviation is decay-free and works for air units the Cannon cannot shoot.
float lateralDeviation(int cardId, float dx, int ticks) {
    GameManager control(DECK, DECK), test(DECK, DECK);
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(4.0f, 18.0f, 1, control.getBoard());
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(4.0f, 18.0f, 1, test.getBoard());
    CardRegistry::getInstance().getCard(CANNON)->spawnEntity(4.0f + dx, 14.0f, 0, test.getBoard());
    control.getBoard().commitPendingEntities();
    test.getBoard().commitPendingEntities();

    auto firstEnemy = [](GameManager& g) -> std::shared_ptr<Entity> {
        for (const auto& e : g.getBoard().getEntities())
            if (e->team == 1 && e->isAlive() && !e->isTower()) return e;
        return nullptr;
    };
    float worst = 0.0f;
    for (int i = 0; i < ticks; i++) {
        control.step(); test.step();
        auto a = firstEnemy(control), b = firstEnemy(test);
        if (!a || !b) break;
        worst = std::max(worst, std::fabs(a->position.x - b->position.x));
    }
    return worst;
}

int towerDamage(int cardId, bool withCannon, int ticks) {
    GameManager g(DECK, DECK);
    int before = team0TowerHp(g);
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(4.0f, 18.0f, 1, g.getBoard());
    if (withCannon)
        CardRegistry::getInstance().getCard(CANNON)->spawnEntity(4.0f, 11.0f, 0, g.getBoard());
    g.getBoard().commitPendingEntities();
    for (int i = 0; i < ticks; i++) g.step();
    return before - team0TowerHp(g);
}

struct Row { int id; const char* name; };
const std::vector<Row> CARDS = {
    {15, "Hog Rider"},   {2,  "Giant"},        {45, "Balloon"},     // covered controls
    {81, "Battle Ram"},  {91, "Lava Hound"},   {84, "Electro Giant"},
    {86, "Rune Giant"},  {121,"Goblinstein"},  {85, "Suspicious Bush"},
    {161,"Battle Ram E"},
};
} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "card                 sight  maxPull  dmgNoCannon  dmgCannon  prevented\n";
    for (const auto& c : CARDS) {
        Board probe;
        CardRegistry::getInstance().getCard(c.id)->spawnEntity(9.0f, 5.0f, 0, probe);
        probe.commitPendingEntities(0);
        float sight = -1.0f;
        for (const auto& e : probe.getEntities())
            if (auto* ce = dynamic_cast<CombatEntity*>(e.get())) { sight = ce->sightRange; break; }

        float maxPull = 0.0f;
        for (float dx = 0.5f; dx <= 13.0f; dx += 0.25f)
            if (lateralDeviation(c.id, dx, 400) > 0.5f) maxPull = dx;

        int d0 = towerDamage(c.id, false, 900);
        int d1 = towerDamage(c.id, true, 900);
        std::cout << std::left << std::setw(20) << c.name << " "
                  << std::right << std::setw(5) << sight << "  "
                  << std::setw(7) << maxPull << "  "
                  << std::setw(11) << d0 << "  "
                  << std::setw(9) << d1 << "  "
                  << std::setw(9) << (d0 - d1) << "\n";
    }
    return 0;
}
