// Minimal reproduction of the soak's residual stall.
//
// Places one team-1 Musketeer at the exact position the soak dump reported
// (4.001, 15.132) and watches it. If it walks, the stall needs another entity
// to reproduce and the soak dump tells us which. If it stands still here, this
// is the minimal case.

#include "GameManager.h"
#include "CardRegistry.h"
#include "Troop.h"
#include <iostream>
#include <iomanip>
#include <vector>

namespace {
const std::vector<int> DECK = { 15, 6, 25, 40, 24, 72, 33, 7 };

void watch(const char* label, int cardId, int team, float x, float y,
           const std::vector<std::tuple<int, int, float, float>>& extras = {}) {
    GameManager game(DECK, DECK);
    game.reset();
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(x, y, team, game.getBoard());
    for (const auto& ex : extras) {
        CardRegistry::getInstance().getCard(std::get<0>(ex))
            ->spawnEntity(std::get<2>(ex), std::get<3>(ex), std::get<1>(ex), game.getBoard());
    }
    game.getBoard().commitPendingEntities();

    int watchId = -1;
    for (const auto& e : game.getBoard().getEntities()) {
        if (!e->isTower() && e->team == team && e->cardId == cardId) { watchId = e->id; break; }
    }

    std::cout << "\n=== " << label << " ===\n";
    Vector2D prev{ -1, -1 };
    int still = 0;
    for (int tick = 1; tick <= 120; ++tick) {
        game.step();
        Vector2D p{ -1, -1 };
        bool alive = false;
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->id == watchId && e->isAlive()) { p = e->position; alive = true; break; }
        }
        if (!alive) { std::cout << "  died at tick " << tick << "\n"; return; }
        if (tick > 12 && prev.distanceTo(p) < 1e-4f) still++; else still = 0;
        if (tick <= 16 || tick % 20 == 0 || still == 1) {
            std::cout << "  t=" << std::setw(4) << tick << "  ("
                << std::fixed << std::setprecision(4) << p.x << ", " << p.y << ")"
                << (still > 0 ? "   [still]" : "") << "\n";
        }
        prev = p;
    }
    std::cout << "  final still-run: " << still << " ticks\n";
}
} // namespace

int main() {
    // The soak's exact reported position.
    watch("lone team-1 Musketeer at the soak's stall position", 6, 1, 4.001f, 15.132f);
    // Same unit further from the bridge column, as a control.
    watch("lone team-1 Musketeer mid-lane", 6, 1, 8.0f, 15.132f);
    // And well inside team 0's half.
    watch("lone team-1 Musketeer deep in enemy half", 6, 1, 4.001f, 12.0f);
    // The soak's actual board: a team-0 Cannon was alive at (9.537, 8.485).
    // If the Musketeer stalls only with that present, the Cannon is the cause.
    watch("with the soak's team-0 Cannon present", 6, 1, 4.001f, 15.132f,
          { { 25, 0, 9.537f, 8.485f } });
    // Start where the soak's position history starts, on the bridge, with the
    // full board the soak dump reported.
    watch("soak board, from the history's first position", 6, 1, 3.9762f, 16.7321f,
          { { 25, 0, 9.537f, 8.485f }, { 25, 1, 8.571f, 25.223f } });
    // ...and the same without the friendly Cannon, to ablate it.
    watch("soak board minus the friendly Cannon", 6, 1, 3.9762f, 16.7321f,
          { { 25, 0, 9.537f, 8.485f } });
    // The collision wedge, with the real card and the real board: a team-0
    // Ice Golem in the pocket between its own King Tower (9, 2.5) r=2.0 and a
    // team-0 Cannon at (12.032, 4.169) r=1.0. Approached from slightly outside
    // the fixed point, because the soak trace shows it CONVERGES to that point
    // rather than sitting exactly on it.
    watch("collision wedge: Ice Golem between own King and own Cannon",
          40, 0, 11.58f, 2.84f, { { 25, 0, 12.032f, 4.169f } });
    return 0;
}
