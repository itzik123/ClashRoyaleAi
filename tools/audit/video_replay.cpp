// Replays one observed push inside the engine, for comparison against video.
//
// Places a card at the tile position read off a recording and prints its
// trajectory tick by tick, so the engine's movement can be diffed against what
// the real game actually did from the same start. Positions only -- HP is not
// comparable, because the recording is a levelled match and this engine has no
// card levels at all.

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Troop.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

namespace {
const std::vector<int> DECK = { 2, 6, 25, 40, 24, 72, 33, 7 };   // Giant present

struct Args { int cardId = 2; float x = 3.9f; float y = 14.0f; int team = 0; int ticks = 300; };
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if      (k == "--card")  a.cardId = std::stoi(argv[i+1]);
        else if (k == "--x")     a.x = std::stof(argv[i+1]);
        else if (k == "--y")     a.y = std::stof(argv[i+1]);
        else if (k == "--team")  a.team = std::stoi(argv[i+1]);
        else if (k == "--ticks") a.ticks = std::stoi(argv[i+1]);
    }

    GameManager game(DECK, DECK);
    const auto* card = CardRegistry::getInstance().getCard(a.cardId);
    if (!card) { std::cerr << "no such card " << a.cardId << "\n"; return 1; }
    card->spawnEntity(a.x, a.y, a.team, game.getBoard());
    game.getBoard().commitPendingEntities();

    int id = -1;
    for (const auto& e : game.getBoard().getEntities())
        if (e->cardId == a.cardId && e->team == a.team) { id = e->id; break; }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "tick\tsec\tx\ty\thp\n";
    for (int t = 0; t <= a.ticks; t++) {
        if (t) game.step();
        for (const auto& e : game.getBoard().getEntities()) {
            if (e->id != id) continue;
            if (t % 2 == 0)
                std::cout << t << "\t" << t / 10.0 << "\t"
                          << e->position.x << "\t" << e->position.y << "\t"
                          << e->hp << "\n";
        }
    }
    return 0;
}
