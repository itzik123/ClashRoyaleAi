// Does a spawner building trap its own spawns? (UPSTREAM_REQUESTS.md item 33)
//
// Found by the Reflex Lab's curation suite: a Tombstone flush against the river
// and the board edge spawns Skeletons that never leave the corner. This places
// each spawner at EVERY legal cell, for each team, on an otherwise empty board, runs
// 30 s, and reports the cells where one of its spawns stalls -- alive, past
// deploy, stationary for STALL_TICKS with nothing within its own reach (the
// criterion tools/audit/soak.cpp uses).
//
// Usage:  powershell -File tools/audit/build.ps1 building_trap_audit
//         tools/audit/bin/building_trap_audit.exe [cardId ...]   (default 96 95)
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ClashEnv.h"
#include "CardRegistry.h"
#include "Troop.h"

constexpr int WINDOW = 300;
constexpr int STALL_TICKS = 30;

struct Stall { int x, y; float sx, sy; int tick; std::string name; };

int main(int argc, char** argv) {
    std::vector<int> ids;
    for (int i = 1; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
    if (ids.empty()) ids = {96, 95};
    const std::vector<int> deck = {15, 6, 25, 40, 24, 72, 33, 7};

    ClashEnv base(deck, deck, 3600);
    base.seed(1);
    Board& board = base.debugGame().getBoard();

    for (int team = 0; team < 2; ++team)
    for (int id : ids) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(id);
        if (!def) { std::printf("unknown card %d\n", id); continue; }
        int legal = 0;
        std::vector<Stall> trapped;
        for (int y = 0; y < board.getHeight(); ++y)
            for (int x = 0; x < board.getWidth(); ++x) {
                if (!base.isValidPlacementForCard(id, float(x), float(y), team)) continue;
                ++legal;
                ClashEnv env = base.snapshot();
                env.inject(id, float(x), float(y), team);
                std::unordered_map<int, std::pair<Vector2D, int>> still;
                bool found = false;
                for (int t = 1; t <= WINDOW && !found; ++t) {
                    env.stepSelfPlayFast(4, 0.0f, 0.0f, 4, 0.0f, 0.0f, 1);
                    std::vector<std::shared_ptr<Entity>> living;
                    for (const auto& e : env.debugGame().getBoard().getEntities())
                        if (e->isAlive()) living.push_back(e);
                    for (const auto& e : living) {
                        auto troop = std::dynamic_pointer_cast<Troop>(e);
                        if (!troop || e->team != team || troop->deployTicksRemaining > 0 || troop->getSpeed() <= 0.0f) continue;
                        float slack = 1e9f;
                        for (const auto& o : living) {
                            if (o->team == e->team || (!o->isTargetable() && !o->isTower())) continue;
                            float r = o->getCollisionRadius();
                            if (r <= 0.0f) r = Entity::IMPLICIT_TROOP_RADIUS;
                            const float reach = troop->getAttackRange() + Entity::IMPLICIT_TROOP_RADIUS + r;
                            const float d = e->position.distanceTo(o->position) - reach;
                            if (d < slack) slack = d;
                        }
                        auto it = still.find(e->id);
                        if (it == still.end()) { still[e->id] = {e->position, 0}; continue; }
                        const bool idle = it->second.first.distanceTo(e->position) < 1e-4f && slack > 0.0f;
                        it->second = {e->position, idle ? it->second.second + 1 : 0};
                        if (it->second.second >= STALL_TICKS) {
                            trapped.push_back({x, y, e->position.x, e->position.y, t, e->name});
                            found = true;
                            break;
                        }
                    }
                }
            }
        std::printf("team %d %s (%d): %d of %d legal cells trap a spawn\n", team, def->name.c_str(), id,
                    int(trapped.size()), legal);
        for (const Stall& s : trapped)
            std::printf("  placed (%d,%d): %s stuck at (%.2f,%.2f) from tick %d\n", s.x, s.y, s.name.c_str(),
                        s.sx, s.sy, s.tick - STALL_TICKS);
    }
    return 0;
}
