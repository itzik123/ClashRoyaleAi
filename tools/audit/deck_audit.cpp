// Per-card behavioural audit of DEFAULT_DECK: what each card actually does,
// read from the engine rather than the registry, so the Catch2 suite pins
// measured numbers.

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Troop.h"
#include "Building.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>

namespace {

const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };
constexpr int MINIONS = 41;      // a flying probe target
constexpr int HOG_RIDER = 15;

struct Env {
    GameManager game;
    Env() : game(DECK, DECK) { game.reset(); }

    std::shared_ptr<Entity> spawn(int cardId, float x, float y, int team) {
        auto* card = CardRegistry::getInstance().getCard(cardId);
        card->spawnEntity(x, y, team, game.getBoard());
        game.getBoard().commitPendingEntities();
        std::shared_ptr<Entity> last;
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->cardId == cardId && e->team == team && e->isAlive()) last = e;
        }
        return last;
    }
    void step(int n) { for (int i = 0; i < n; ++i) game.step(); }

    int countAlive(int cardId, int team) {
        int n = 0;
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->isAlive() && e->cardId == cardId && e->team == team) n++;
        }
        return n;
    }
    // Total HP of the given team's towers -- the defensive read-out.
    int towerHp(int team) {
        int t = 0;
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->isAlive() && e->isTower() && e->team == team) t += e->hp;
        }
        return t;
    }
};

std::string name(int id) {
    auto* c = CardRegistry::getInstance().getCard(id);
    return c ? c->name : "?";
}

// ---- 1. identity, straight from the registry through a spawned entity ----
void identity() {
    std::cout << "\n=== CARD IDENTITY (read off spawned entities) ===\n";
    std::cout << std::left << std::setw(14) << "card" << std::setw(6) << "cost"
        << std::setw(8) << "bodies" << std::setw(8) << "hp" << std::setw(9) << "dmg"
        << std::setw(7) << "cd" << std::setw(8) << "range" << std::setw(8) << "speed"
        << std::setw(7) << "air?" << std::setw(7) << "fly?" << "kind\n";

    for (int id : DECK) {
        Env env;
        auto* card = CardRegistry::getInstance().getCard(id);
        auto e = env.spawn(id, 9.0f, 10.0f, 0);
        int bodies = env.countAlive(id, 0);

        std::string kind = "spell";
        std::string hp = "-", dmg = "-", cd = "-", rng = "-", spd = "-", air = "-", fly = "-";
        if (e) {
            auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
            auto troop = std::dynamic_pointer_cast<Troop>(e);
            auto bldg = std::dynamic_pointer_cast<Building>(e);
            kind = troop ? "troop" : (bldg ? "building" : "other");
            hp = std::to_string(e->hp);
            if (combat) {
                dmg = std::to_string(static_cast<int>(combat->getDamagePerTick() * combat->getAttackCooldown()));
                cd = std::to_string(combat->getAttackCooldown());
                char b[16]; std::snprintf(b, sizeof(b), "%.2f", combat->getAttackRange()); rng = b;
                air = combat->targetsAir ? "yes" : "no";
                fly = combat->isFlying ? "yes" : "no";
            }
            if (troop) { char b[16]; std::snprintf(b, sizeof(b), "%.3f", troop->getSpeed()); spd = b; }
        }
        std::cout << std::left << std::setw(14) << name(id)
            << std::setw(6) << static_cast<int>(card->cost)
            << std::setw(8) << bodies << std::setw(8) << hp << std::setw(9) << dmg
            << std::setw(7) << cd << std::setw(8) << rng << std::setw(8) << spd
            << std::setw(7) << air << std::setw(7) << fly << kind << "\n";
    }
}

// ---- 2. offence: lone unit vs the full enemy side ----
void offence() {
    std::cout << "\n=== OFFENCE: one unit at the bridge, 600 ticks ===\n";
    std::cout << "tower damage dealt to the enemy, and whether it survived\n\n";
    std::cout << std::left << std::setw(14) << "card" << std::setw(14) << "towerDamage"
        << std::setw(10) << "alive" << "\n";
    for (int id : DECK) {
        Env env;
        int before = env.towerHp(1);
        env.spawn(id, 4.0f, 15.0f, 0);
        env.step(600);
        int dealt = before - env.towerHp(1);
        std::cout << std::left << std::setw(14) << name(id)
            << std::setw(14) << dealt
            << std::setw(10) << env.countAlive(id, 0) << "\n";
    }
}

// ---- 3. defence: does the card stop an incoming Hog? ----
void defence() {
    std::cout << "\n=== DEFENCE: enemy Hog Rider crosses, card answers it ===\n";
    std::cout << "own tower HP lost over 400 ticks; baseline is no answer at all\n\n";

    int baseline = 0;
    {
        Env env;
        int before = env.towerHp(0);
        env.spawn(HOG_RIDER, 4.0f, 18.0f, 1);
        env.step(400);
        baseline = before - env.towerHp(0);
    }
    std::cout << std::left << std::setw(14) << "(no answer)" << baseline << " HP lost\n";

    for (int id : DECK) {
        Env env;
        int before = env.towerHp(0);
        env.spawn(HOG_RIDER, 4.0f, 18.0f, 1);
        env.step(20);                       // let it commit to the bridge
        env.spawn(id, 4.0f, 11.0f, 0);      // answer it in our own half
        env.step(380);
        int lost = before - env.towerHp(0);
        std::cout << std::left << std::setw(14) << name(id)
            << std::setw(10) << lost << " HP lost   prevented "
            << (baseline - lost) << "\n";
    }
}

// --- 4. air interaction ---
// Every number is a difference against a control run with no card played,
// because Minions fly at our own towers and the towers' damage would otherwise
// be credited to the card.
void air() {
    std::cout << "\n=== AIR: can the card damage flying Minions? ===\n";
    std::cout << "measured as a DIFFERENCE against a no-card control, so our own\n";
    std::cout << "towers' shooting is subtracted out rather than credited\n\n";

    auto minionHpAfter = [](int cardId) {
        Env env;
        // The centre of the board on the river line, 11.2 tiles from every
        // Princess Tower (beyond their 9.4 reach), and a short window before
        // the Minions reach a tower. The control HP is printed: if it is not
        // the full 690, the probe is measuring the towers again.
        env.spawn(MINIONS, 9.0f, 16.5f, 1);
        if (cardId >= 0) env.spawn(cardId, 9.0f, 15.5f, 0);
        env.step(35);
        int hp = 0;
        for (const auto& e : env.game.getBoard().getEntities())
            if (e->cardId == MINIONS && e->isAlive()) hp += e->hp;
        return hp;
    };

    const int control = minionHpAfter(-1);
    std::cout << "  " << std::left << std::setw(14) << "(control)"
        << "minion HP remaining " << control << "\n";
    for (int id : DECK) {
        int withCard = minionHpAfter(id);
        int attributable = control - withCard;
        std::cout << "  " << std::left << std::setw(14) << name(id)
            << "remaining " << std::setw(6) << withCard
            << "  attributable " << std::setw(6) << attributable
            << (attributable > 0 ? "  HITS AIR" : "  cannot touch air") << "\n";
    }
}

// --- 7. who is doing the defending ---
// Read off MatchStatistics by the reserved tower cardIds: attribution, not
// inference.
void defenders() {
    std::cout << "\n=== DEFENSIVE DAMAGE ATTRIBUTION (lone attacker at the bridge) ===\n";
    std::cout << std::left << std::setw(14) << "attacker" << std::setw(12) << "byKing"
        << std::setw(14) << "byPrincess" << "kingShare\n";
    for (int id : { 15, 6, 40 }) {
        Env env;
        env.spawn(id, 4.0f, 15.0f, 0);
        env.step(600);
        int king = env.game.getStatistics().damageDealtByCard(GameManager::TOWER_KING_ID, 1);
        int princess = env.game.getStatistics().damageDealtByCard(GameManager::TOWER_PRINCESS_ID, 1);
        double share = (king + princess) > 0 ? (100.0 * king / (king + princess)) : 0.0;
        std::cout << std::left << std::setw(14) << name(id)
            << std::setw(12) << king << std::setw(14) << princess
            << std::fixed << std::setprecision(1) << share << "%\n";
    }
}

// ---- 5. deploy time ----
void deploy() {
    std::cout << "\n=== DEPLOY TIME: inert for DEPLOY_TIME_TICKS, then acts ===\n";
    for (int id : DECK) {
        Env env;
        auto e = env.spawn(id, 9.0f, 10.0f, 0);
        if (!e) { std::cout << "  " << std::left << std::setw(14) << name(id) << "spell (no deploy)\n"; continue; }
        auto combat = std::dynamic_pointer_cast<CombatEntity>(e);
        int startDeploy = combat ? combat->deployTicksRemaining : -1;
        Vector2D p0 = e->position;
        env.step(DEPLOY_TIME_TICKS);
        Vector2D pMid = e->position;
        env.step(20);
        Vector2D pEnd = e->position;
        std::cout << "  " << std::left << std::setw(14) << name(id)
            << "deployTicks=" << std::setw(4) << startDeploy
            << " movedDuringDeploy=" << std::fixed << std::setprecision(4)
            << p0.distanceTo(pMid)
            << "  movedAfter=" << pMid.distanceTo(pEnd) << "\n";
    }
}

// --- 6. the sight/attack band ---
// Sight is centre-to-centre while attacks use effectiveRangeTo (attack range
// plus both radii). For a Princess Tower against a troop the two differ by 1.9
// tiles; a unit inside that band could hit a tower that cannot see it.
void deadBand() {
    std::cout << "\n=== SIGHT vs ATTACK DEAD BAND (lone unit vs one Princess Tower) ===\n";
    std::cout << "unit placed at a fixed distance south of the enemy tower at (4, 27)\n\n";
    std::cout << std::left << std::setw(9) << "dist" << std::setw(16) << "towerDmgTaken"
        << std::setw(16) << "unitDmgTaken" << "verdict\n";

    for (float d = 6.5f; d <= 10.5f; d += 0.5f) {
        Env env;
        int towerBefore = env.towerHp(1);
        auto m = env.spawn(6, 4.0f, 27.0f - d, 0);   // Musketeer
        int unitBefore = m ? m->hp : 0;
        env.step(300);
        int unitAfter = 0;
        for (const auto& e : env.game.getBoard().getEntities())
            if (e->id == m->id && e->isAlive()) unitAfter = e->hp;
        int towerDmg = towerBefore - env.towerHp(1);
        int unitDmg = unitBefore - unitAfter;
        std::cout << std::left << std::setw(9) << std::fixed << std::setprecision(1) << d
            << std::setw(16) << towerDmg << std::setw(16) << unitDmg
            << ((towerDmg > 0 && unitDmg == 0) ? "FREE HIT -- tower never fired back" : "")
            << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "all";
    if (mode == "deadband") deadBand();
    if (mode == "defenders") defenders();
    if (mode == "all" || mode == "identity") identity();
    if (mode == "all" || mode == "offence") offence();
    if (mode == "all" || mode == "defence") defence();
    if (mode == "all" || mode == "air") air();
    if (mode == "all" || mode == "deploy") deploy();
    return 0;
}
