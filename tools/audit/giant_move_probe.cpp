// Is a Giant at (9,8) frozen, or just moving less than one cell in 2 seconds?
#include <cstdio>
#include "ArenaLayout.h"
#include "GameManager.h"
#include "CardRegistry.h"
#include "LanePath.h"

int main() {
    GameManager game({15,6,25,40,24,72,33,7}, {15,6,25,40,24,72,33,7});
    Board& b = game.getBoard();
    CardRegistry::getInstance().getCard(2)->spawnEntity(9.0f, 8.0f, 0, b);
    b.commitPendingEntities();
    std::shared_ptr<Entity> g;
    for (const auto& e : b.getEntities()) if (e->cardId == 2) g = e;

    auto obj = LanePath::laneObjective(b, 0, g->position);
    std::printf("lane objective: %s team %d at (%.2f, %.2f)\n",
                obj->symbol == 'R' ? "King" : "Princess", obj->team,
                obj->position.x, obj->position.y);
    std::printf("nearest bridge from x=9.0: left %.2f right %.2f\n",
                ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::RIGHT_BRIDGE_X);

    for (int t = 1; t <= 40; ++t) {
        game.step();
        if (t <= 12 || t % 10 == 0)
            std::printf("t=%2d pos=(%.4f, %.4f)  cell=(%d,%d)\n", t,
                        g->position.x, g->position.y,
                        (int)g->position.x, (int)g->position.y);
    }
    return 0;
}
