// End to end: does a real timed-out replay carry a decisive verdict? The match
// runs to the 3600-tick limit with both Kings alive and one Princess Tower
// badly damaged.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "ArenaLayout.h"
#include "ClashEnv.h"
#include "GameLogger.h"

int main() {
    const std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };
    GameManager game(deck, deck);
    Board& board = game.getBoard();

    // Team 0's left Princess at 90 hp: tower counts stay 3-3, so only the
    // tie-break can decide.
    for (const auto& e : board.getEntities()) {
        if (e->isTower() && e->team == 0 && e->symbol != 'R') { e->takeDamage(e->hp - 90); break; }
    }

    GameLogger logger;
    for (int t = 1; t <= 3600; ++t) { game.step(); if (t % 10 == 0) logger.logTick(t, game); }

    std::printf("verdict written into the replay:\n  %s\n\n", logger.resultJson().c_str());

    const std::string path = "tools/audit/bin/verdict_probe_replay.json";
    if (!logger.save(path)) { std::printf("save FAILED\n"); return 1; }

    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"result\"") != std::string::npos) {
            std::printf("as it appears in the JSON the viewer loads:\n %s\n", line.c_str());
            return line.find("\"loserTeam\": 0") != std::string::npos ? 0 : 1;
        }
    }
    std::printf("no \"result\" key found in the replay!\n");
    return 1;
}
