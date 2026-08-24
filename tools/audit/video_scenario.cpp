// Replays an observed PUSH (several cards) and reports the defending tower's HP
// over time, so a video timeline can be diffed against the engine's.
//
// Placements are given as cardId:x:y:team, applied on tick 0.

#include "GameManager.h"
#include "CardRegistry.h"
#include "Tower.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>

namespace {
const std::vector<int> DECK = { 2, 1, 25, 40, 24, 72, 33, 7 };   // Giant + Archers

// The enemy LEFT princess tower: team 1, the low-x one.
int leftPrincessHp(GameManager& g) {
    for (const auto& e : g.getBoard().getEntities())
        if (e->team == 1 && e->isTower() && e->position.x < 9.0f
            && e->cardId == GameManager::TOWER_PRINCESS_ID)
            return e->isAlive() ? e->hp : 0;
    return -1;
}
}

int main(int argc, char** argv) {
    int ticks = 200;
    std::vector<std::array<float,4>> place;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--ticks" && i + 1 < argc) { ticks = std::stoi(argv[++i]); continue; }
        std::stringstream ss(s); std::string tok; std::vector<float> v;
        while (std::getline(ss, tok, ':')) v.push_back(std::stof(tok));
        if (v.size() == 4) place.push_back({v[0], v[1], v[2], v[3]});
    }

    GameManager game(DECK, DECK);
    for (auto& p : place) {
        const auto* c = CardRegistry::getInstance().getCard((int)p[0]);
        if (!c) { std::cerr << "no card " << p[0] << "\n"; return 1; }
        c->spawnEntity(p[1], p[2], (int)p[3], game.getBoard());
    }
    game.getBoard().commitPendingEntities();

    int start = leftPrincessHp(game);
    std::cout << "start tower hp: " << start << "\n";
    std::cout << "sec\ttowerHp\tlost\n" << std::fixed << std::setprecision(1);
    int destroyedAt = -1;
    for (int t = 0; t <= ticks; t++) {
        if (t) game.step();
        int hp = leftPrincessHp(game);
        if (hp <= 0 && destroyedAt < 0) destroyedAt = t;
        if (t % 10 == 0) std::cout << t/10.0 << "\t" << hp << "\t" << (start-hp) << "\n";
    }
    if (destroyedAt >= 0)
        std::cout << "DESTROYED at " << destroyedAt/10.0 << " s\n";
    else
        std::cout << "survived " << ticks/10.0 << " s\n";
    return 0;
}
