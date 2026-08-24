// Records the observed push as a REAL replay file, for web/viewer.html.
//
// Uses ClashEnv rather than a bare GameManager because the logger lives there
// and is driven from inside step/stepSelfPlay -- so the replay is produced by
// the same code path a training episode uses, not by a second hand-rolled
// writer that could describe the match differently.
//
// stepSelfPlay (not step) so the C++ HeuristicOpponent never plays: the point
// is to watch ONE injected push cross an otherwise empty board, matching what
// the recording shows.

#include "ClashEnv.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

int main(int argc, char** argv) {
    std::string out = "replay.json";
    int ticks = 260;
    std::vector<std::array<float,4>> place;   // cardId : x : y : team
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--out" && i + 1 < argc)   { out = argv[++i]; continue; }
        if (s == "--ticks" && i + 1 < argc) { ticks = std::stoi(argv[++i]); continue; }
        std::stringstream ss(s); std::string tok; std::vector<float> v;
        while (std::getline(ss, tok, ':')) v.push_back(std::stof(tok));
        if (v.size() == 4) place.push_back({v[0], v[1], v[2], v[3]});
    }

    std::vector<int> deck = { 2, 1, 25, 40, 24, 72, 33, 7 };
    ClashEnv env(deck, deck, 3600);
    env.seed(4242);            // item 7: the opening shuffle is seedable now
    env.reset();

    for (auto& p : place)
        env.inject((int)p[0], p[1], p[2], (int)p[3]);

    // inject() QUEUES a spawn -- nothing is on the board, or in any
    // observation, until one tick is stepped. Documented in CLAUDE.md and
    // measured (0.0 enemy mass immediately after inject, 0.399 after a tick).
    for (int t = 0; t < ticks; t++) {
        env.stepSelfPlay(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 1);
        if (env.isGameOver()) break;
    }
    env.saveLog(out);
    std::cout << "wrote " << out << " (" << ticks << " ticks)\n";
    return 0;
}
