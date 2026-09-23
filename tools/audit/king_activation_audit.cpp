// How much does King dormancy change a lone Hog push? Same match and placement;
// the Kings are either woken up front or left dormant.
#include <cstdio>
#include <vector>
#include <memory>
#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"
#include "Tower.h"
#include "CardRegistry.h"

static int enemyTowerHp(GameManager& g, int team) {
    int t = 0;
    for (const auto& e : g.getBoard().getEntities())
        if (e->isAlive() && e->isTower() && e->team == team) t += e->hp;
    return t;
}

int main() {
    const std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };

    for (int arm = 0; arm < 2; ++arm) {
        const bool wakeKings = (arm == 0);
        GameManager game(deck, deck);
        Board& board = game.getBoard();
        if (wakeKings)
            for (const auto& e : board.getEntities())
                if (auto t = std::dynamic_pointer_cast<Tower>(e)) t->wake();

        const int before = enemyTowerHp(game, 0);
        CardRegistry::getInstance().getCard(15)->spawnEntity(
            ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::BRIDGE_Y + 1.0f, 1, board);
        board.commitPendingEntities();

        std::shared_ptr<Entity> hog;
        for (const auto& e : board.getEntities())
            if (e->team == 1 && !e->isTower() && !e->isBuilding()) hog = e;

        int diedAt = -1;
        for (int i = 0; i < 600; ++i) {
            game.step();
            if (diedAt < 0 && !hog->isAlive()) diedAt = i;
        }
        std::printf("%-18s tower damage dealt=%5d   hog survived=%s (tick %d)\n",
                    wakeKings ? "Kings AWAKE (old)" : "Kings DORMANT (new)",
                    before - enemyTowerHp(game, 0),
                    hog->isAlive() ? "yes" : "no ", diedAt);
    }
    return 0;
}
