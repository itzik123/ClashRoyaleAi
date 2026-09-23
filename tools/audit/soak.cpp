// Full-match soak test for navigation stalls. Unlike bridge_audit.cpp's
// clean-room sweeps, this plays whole randomized matches with both sides
// spending, so units meet, collide, die, retarget and crowd the bridges as in
// training, and it reports any unit that stops moving with nothing to fight.
//
// A stall: alive, past deploy time, and stationary for STALL_TICKS consecutive
// ticks during every one of which nothing was inside its own effective attack
// reach.

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Troop.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace {

constexpr int STALL_TICKS = 50;      // 5 seconds
constexpr float RIVER_START = 15.5f;
constexpr float RIVER_END = 17.5f;

struct Watch {
    Vector2D last{ 0, 0 };
    int stillFor = 0;
    bool reported = false;
    bool seen = false;
    std::vector<Vector2D> hist;   // recent positions, for the first-stall dump
};

struct Report {
    int matches = 0;
    long long unitTicks = 0;
    int stalls = 0;
    int stallsOnBridge = 0;
    std::vector<std::string> examples;
};

const std::vector<int>& defaultDeck() {
    static const std::vector<int> d = { 15, 6, 25, 40, 24, 72, 33, 7 };
    return d;
}

} // namespace

// Deterministic re-run of one seeded match, tracing one entity tick by tick:
// the frames around the dump distinguish "never moved" from "stopped for a
// reason that has gone away".
int trace(int m, int entityId, int fromTick, int toTick) {
    GameManager game(defaultDeck(), defaultDeck());
    game.seed(static_cast<unsigned>(1000 + m));
    game.reset();
    std::mt19937 rng(9000 + m);
    std::uniform_real_distribution<float> ux(0.5f, 16.5f);
    std::uniform_int_distribution<int> uslot(0, 3);
    std::uniform_int_distribution<int> uplay(0, 9);

    for (int tick = 0; tick < toTick && !game.isGameOver(); ++tick) {
        for (int team = 0; team < 2; ++team) {
            if (uplay(rng) != 0) continue;
            PlayerState& p = (team == 0) ? game.playerAI : game.playerOpponent;
            if (p.hand.empty()) continue;
            int slot = uslot(rng) % static_cast<int>(p.hand.size());
            int cardId = p.hand[slot];
            float x = ux(rng);
            std::uniform_real_distribution<float> uy(
                team == 0 ? 1.0f : 18.5f, team == 0 ? 14.5f : 32.0f);
            game.playCard(team, cardId, x, uy(rng));
        }
        game.step();
        if (tick < fromTick) continue;

        std::shared_ptr<Entity> me;
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->id == entityId && e->isAlive()) { me = e; break; }
        }
        if (!me) { std::cout << "tick " << tick << ": entity gone\n"; continue; }
        auto troop = std::dynamic_pointer_cast<Troop>(me);

        // Everything it could legally be shooting, nearest-by-slack first.
        float bestSlack = 1e9f; std::string what = "-"; float bestDist = 0;
        for (const auto& o : game.getBoard().getEntities()) {
            if (!o->isAlive() || o->team == me->team) continue;
            if (!o->isTargetable() && !o->isTower()) continue;
            float r = o->getCollisionRadius();
            if (r <= 0.0f) r = Entity::IMPLICIT_TROOP_RADIUS;
            float reach = troop->getAttackRange() + Entity::IMPLICIT_TROOP_RADIUS + r;
            float d = me->position.distanceTo(o->position);
            if (d - reach < bestSlack) {
                bestSlack = d - reach; bestDist = d;
                auto* oc = CardRegistry::getInstance().getCard(o->cardId);
                what = o->isTower() ? "TOWER" : (oc ? oc->name : "?");
            }
        }
        std::cout << "tick " << std::setw(4) << tick
            << "  pos=(" << std::fixed << std::setprecision(4) << me->position.x
            << ", " << me->position.y << ")"
            << "  freeze=" << troop->freezeTicks
            << "  hp=" << std::setw(4) << me->hp
            << "  best=" << std::setw(12) << what << "@" << std::setprecision(2) << bestDist
            << " slack=" << std::setw(7) << bestSlack << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "trace") {
        return trace(std::atoi(argv[2]), std::atoi(argv[3]),
                     std::atoi(argv[4]), std::atoi(argv[5]));
    }
    int matches = (argc > 1) ? std::atoi(argv[1]) : 60;
    int maxTicks = (argc > 2) ? std::atoi(argv[2]) : 1800;

    Report rep;
    bool dumped = false;

    for (int m = 0; m < matches; ++m) {
        GameManager game(defaultDeck(), defaultDeck());
        game.seed(static_cast<unsigned>(1000 + m));
        game.reset();

        std::mt19937 rng(9000 + m);
        std::uniform_real_distribution<float> ux(0.5f, 16.5f);
        std::uniform_int_distribution<int> uslot(0, 3);
        std::uniform_int_distribution<int> uplay(0, 9);

        std::map<int, Watch> watch;

        for (int tick = 0; tick < maxTicks && !game.isGameOver(); ++tick) {
            // Both sides spend, roughly one attempt per second per side.
            for (int team = 0; team < 2; ++team) {
                if (uplay(rng) != 0) continue;
                PlayerState& p = (team == 0) ? game.playerAI : game.playerOpponent;
                if (p.hand.empty()) continue;
                int slot = uslot(rng) % static_cast<int>(p.hand.size());
                int cardId = p.hand[slot];
                float x = ux(rng);
                // Own half, clear of the river buffer.
                std::uniform_real_distribution<float> uy(
                    team == 0 ? 1.0f : 18.5f, team == 0 ? 14.5f : 32.0f);
                game.playCard(team, cardId, x, uy(rng));
            }

            game.step();

            // Snapshot the living, so proximity is evaluated at the same
            // instant as the positions.
            std::vector<std::shared_ptr<Entity>> living;
            for (const auto& e : game.getBoard().getEntities()) {
                if (e->isAlive()) living.push_back(e);
            }

            for (const auto& e : living) {
                if (e->isTower()) continue;
                // Troop, not CombatEntity: buildings are supposed to stand
                // still.
                auto troop = std::dynamic_pointer_cast<Troop>(e);
                if (!troop) continue;                    // buildings, spells, projectiles
                if (troop->deployTicksRemaining > 0) continue;
                if (troop->getSpeed() <= 0.0f) continue;

                rep.unitTicks++;
                Watch& w = watch[e->id];
                w.hist.push_back(e->position);
                if (w.hist.size() > 70) w.hist.erase(w.hist.begin());
                // An explicit first-sighting flag.
                if (!w.seen) { w.seen = true; w.last = e->position; continue; }
                float moved = w.last.distanceTo(e->position);

                // Is anything within this unit's effective reach right now?
                // Mirrors CombatEntity::effectiveRangeTo. Evaluated every tick
                // and folded into the counter: a unit that shot a target which
                // died on the tick the counter tripped would otherwise be
                // flagged.
                float nearestSlack = 1e9f;
                float nearestDist = 1e9f;
                std::string nearestWhat = "nothing";
                for (const auto& other : living) {
                    if (other->team == e->team) continue;
                    if (!other->isTargetable() && !other->isTower()) continue;
                    float targetRadius = other->getCollisionRadius();
                    if (targetRadius <= 0.0f) targetRadius = Entity::IMPLICIT_TROOP_RADIUS;
                    float reach = troop->getAttackRange() + Entity::IMPLICIT_TROOP_RADIUS + targetRadius;
                    float d = e->position.distanceTo(other->position);
                    if (d - reach < nearestSlack) {
                        nearestSlack = d - reach;
                        nearestDist = d;
                        auto* oc = CardRegistry::getInstance().getCard(other->cardId);
                        nearestWhat = other->isTower() ? "TOWER" : (oc ? oc->name : "?");
                    }
                }
                const bool hadNothingToShoot = (nearestSlack > 0.0f);

                if (moved < 1e-4f && hadNothingToShoot) {
                    w.stillFor++;
                }
                else {
                    w.stillFor = 0;
                    w.reported = false;
                }
                w.last = e->position;

                if (w.stillFor < STALL_TICKS || w.reported) continue;

                // Decisive dump on the first stall: the trace, not a guess.
                if (!dumped) {
                    dumped = true;
                    std::cout << "--- FIRST STALL, full board state ---\n";
                    std::cout << "stalled unit id=" << e->id << " pos=(" << e->position.x
                        << ", " << e->position.y << ") team=" << e->team
                        << " speed=" << troop->getSpeed()
                        << " range=" << troop->getAttackRange()
                        << " freezeTicks=" << troop->freezeTicks
                        << " freezeSlow=" << troop->freezeSlow
                        << " deployLeft=" << troop->deployTicksRemaining
                        << " flying=" << troop->isFlying
                        << " hp=" << troop->hp << "\n";
                    for (const auto& o : living) {
                        auto* oc = CardRegistry::getInstance().getCard(o->cardId);
                        std::cout << "   id=" << std::setw(4) << o->id
                            << " team=" << o->team
                            << " " << std::setw(14) << (o->isTower() ? "TOWER" : (oc ? oc->name : "?"))
                            << " pos=(" << std::fixed << std::setprecision(3)
                            << std::setw(8) << o->position.x << ","
                            << std::setw(8) << o->position.y << ")"
                            << " hp=" << std::setw(5) << o->hp
                            << " r=" << o->getCollisionRadius()
                            << " tgtable=" << o->isTargetable()
                            << " bldg=" << o->isBuilding()
                            << " dist=" << e->position.distanceTo(o->position) << "\n";
                    }
                    // What does navigation actually tell it to do?
                    for (const auto& o : living) {
                        if (o->team == e->team || !o->isTower()) continue;
                        Vector2D wp = game.getBoard().getNextWaypoint(e->position, o->position);
                        float dist = e->position.distanceTo(wp);
                        // Reproduce Troop::moveTowards, then ask collision
                        // resolution what it does to the result: separates
                        // "never tried to move" from "moved and was pushed
                        // back".
                        Vector2D stepped{
                            e->position.x + (wp.x - e->position.x) / dist * troop->getSpeed(),
                            e->position.y + (wp.y - e->position.y) / dist * troop->getSpeed() };
                        Vector2D resolved = game.getBoard().resolvePositionAgainstBuildings(stepped, e->id);
                        Vector2D clamped = game.getBoard().clampToBoard(resolved, false);
                        std::cout << "   waypoint toward tower id=" << o->id << " -> ("
                            << wp.x << ", " << wp.y << ")  distToWaypoint=" << dist
                            << "  stepWouldGoTo=(" << stepped.x << ", " << stepped.y << ")"
                            << "  afterCollision=(" << resolved.x << ", " << resolved.y << ")"
                            << "  afterClamp=(" << clamped.x << ", " << clamped.y << ")\n";
                    }
                    std::cout << "   position history (oldest first):\n";
                    for (size_t k = 0; k < w.hist.size(); ++k) {
                        std::cout << "     [" << std::setw(2) << k << "] ("
                            << std::setprecision(4) << w.hist[k].x << ", "
                            << w.hist[k].y << ")\n";
                    }
                    std::cout << "--- end dump ---\n";
                }

                w.reported = true;
                rep.stalls++;
                bool onBridge = (e->position.y > RIVER_START - 1.0f && e->position.y < RIVER_END + 1.0f);
                if (onBridge) rep.stallsOnBridge++;
                if (rep.examples.size() < 10) {
                    auto* card = CardRegistry::getInstance().getCard(e->cardId);
                    char buf[220];
                    std::snprintf(buf, sizeof(buf),
                        "match %d tick %d: %s (team %d) still %d ticks at (%.3f, %.3f) "
                        "range=%.1f nearest=%s@%.2f (slack %.2f)%s",
                        m, tick, card ? card->name.c_str() : "?", e->team, w.stillFor,
                        e->position.x, e->position.y, troop->getAttackRange(),
                        nearestWhat.c_str(), nearestDist, nearestSlack,
                        onBridge ? "  [ON BRIDGE]" : "");
                    rep.examples.push_back(buf);
                }
            }
        }
        rep.matches++;
    }

    std::cout << "=== FULL-MATCH SOAK ===\n";
    std::cout << "matches:            " << rep.matches << "\n";
    std::cout << "unit-ticks watched: " << rep.unitTicks << "\n";
    std::cout << "stalls (>" << STALL_TICKS << " ticks idle, nothing within own reach): "
        << rep.stalls << "\n";
    std::cout << "  ...of those, on or beside a bridge: " << rep.stallsOnBridge << "\n";
    for (const auto& s : rep.examples) std::cout << "  " << s << "\n";
    return rep.stalls == 0 ? 0 : 1;
}
