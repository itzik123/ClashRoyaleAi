// Replays a list of placements through the engine and dumps every entity's
// state every tick. Deliberately TSV in and TSV out: this has to build under
// bare g++ with no JSON dependency, and the Python side owns the schema.
//
//   in :  tick <TAB> cardId <TAB> x <TAB> y <TAB> team
//   out:  T <TAB> tick <TAB> id <TAB> cardId <TAB> team <TAB> x <TAB> y <TAB> hp <TAB> name
//         (and one "P" line echoing each placement the engine ACCEPTED)
#include "GameManager.h"
#include "CardRegistry.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>

struct Place { int tick, cardId, team; float x, y; };

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: simdrive <places.tsv> <ticks>\n"; return 2; }
    std::ifstream in(argv[1]);
    int maxTicks = std::atoi(argv[2]);

    std::vector<Place> places;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Place p{};
        if (ss >> p.tick >> p.cardId >> p.x >> p.y >> p.team) places.push_back(p);
    }
    std::sort(places.begin(), places.end(),
              [](const Place& a, const Place& b) { return a.tick < b.tick; });

    // The deck only decides what the (unused) hand holds -- every unit here
    // is injected directly, exactly as ClashEnv::inject does it. The recorded
    // deck is passed anyway so elixir and cycle stay plausible.
    std::vector<int> deck{10, 1, 41, 25, 7, 2, 6, 5};
    GameManager game(deck, deck);
    game.seed(12345);
    game.reset();
    auto& reg = CardRegistry::getInstance();

    std::cout << std::fixed << std::setprecision(3);
    size_t next = 0;
    for (int t = 0; t <= maxTicks; ++t) {
        while (next < places.size() && places[next].tick <= t) {
            const Place& p = places[next];
            // deploy_ticks left at the engine default: these are placements,
            // not units already on the board, so the deploy second is real.
            const auto* card = reg.getCard(p.cardId);
            if (!card) { std::cerr << "unknown card " << p.cardId << std::endl; ++next; continue; }
            card->spawnEntity(p.x, p.y, p.team, game.getBoard());
            std::cout << "P\t" << t << "\t" << p.cardId << "\t" << p.team
                      << "\t" << p.x << "\t" << p.y << "\n";
            ++next;
        }
        // Commit before the dump so a unit appears on the tick it was placed,
        // not one tick later -- the video side timestamps the placement, and
        // an off-by-one here would read as latency the engine does not have.
        game.getBoard().commitPendingEntities(t);
        for (auto& e : game.getBoard().getEntities()) {
            if (!e->isAlive()) continue;
            std::cout << "T\t" << t << "\t" << e->id << "\t" << e->cardId << "\t"
                      << e->team << "\t" << e->position.x << "\t" << e->position.y
                      << "\t" << e->hp << "\t" << e->name << "\n";
        }
        if (game.isGameOver()) break;
        game.step();              // one raw engine tick; no policy, no opponent
    }
    return 0;
}
