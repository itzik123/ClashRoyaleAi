// Bridge navigation stress instrument.
//
// NOT a test: a measurement harness. It drives the real GameManager tick loop
// and reads every unit's position every tick, which nothing in the Catch2
// suite does -- those assert on end states, and the reported symptom
// ("units lag or get stuck on bridges") is a property of the TRAJECTORY.
//
// Compiled standalone against the header-only engine (tools/audit/build.ps1)
// so it needs no CMake target and cannot perturb the shipped build.

#include "GameManager.h"
#include "CardRegistry.h"
#include "Board.h"
#include "CombatEntity.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace {

constexpr float RIVER_START = 15.5f;
constexpr float RIVER_END = 17.5f;

struct Track {
    int entityId = -1;
    std::string card;
    std::vector<Vector2D> path;      // position at each tick, while alive
    bool crossed = false;
    int ticksToCross = -1;
    int deathTick = -1;
};

// One trajectory, reduced to the numbers that describe "did it navigate
// cleanly".
struct Verdict {
    bool crossed = false;
    int ticksToCross = -1;
    int longestStall = 0;       // consecutive ticks with ~zero displacement
    int stallStartTick = -1;
    Vector2D stallPos{ 0, 0 };
    int bankSnaps = 0;          // entered the river band, then snapped back to a bank
    int reversals = 0;          // sign flips of dy -- path jitter
    float totalPath = 0.0f;
};

Verdict judge(const Track& t) {
    Verdict v;
    v.crossed = t.crossed;
    v.ticksToCross = t.ticksToCross;

    int run = 0;
    int lastDySign = 0;
    bool wasInRiver = false;

    for (size_t i = 1; i < t.path.size(); ++i) {
        const Vector2D& a = t.path[i - 1];
        const Vector2D& b = t.path[i];
        float d = a.distanceTo(b);
        v.totalPath += d;

        // A stall only counts BEFORE the crossing completes -- after that the
        // unit is legitimately standing still hitting a tower.
        bool beforeCross = (t.ticksToCross < 0) || (static_cast<int>(i) <= t.ticksToCross);
        if (d < 1e-4f && beforeCross) {
            ++run;
            if (run > v.longestStall) {
                v.longestStall = run;
                v.stallStartTick = static_cast<int>(i) - run;
                v.stallPos = b;
            }
        }
        else {
            run = 0;
        }

        float dy = b.y - a.y;
        int s = (dy > 1e-5f) ? 1 : (dy < -1e-5f ? -1 : 0);
        // Captured BEFORE lastDySign is advanced: both the reversal count and
        // the snap test below need the PREVIOUS direction, and reading
        // lastDySign after the update makes "s != lastDySign" identically
        // false -- a detector that reports zero no matter what happens.
        int prevDySign = lastDySign;
        bool reversed = (s != 0 && prevDySign != 0 && s != prevDySign);
        if (reversed) v.reversals++;
        if (s != 0) lastDySign = s;

        bool inRiver = (b.y > RIVER_START && b.y < RIVER_END);
        // Snapped out of the river band back onto a bank edge -- clampToBoard's
        // signature, and a loss of crossing progress.
        //
        // The REVERSAL clause is load-bearing and was added after the first
        // version of this detector reported 2 snaps each for Giant, Musketeer
        // and Valkyrie that were all false positives: those three have speeds
        // (0.06, 0.10, 0.10) that divide the 7.5 tiles from the spawn row
        // evenly, so they LAND on y=17.5000 exactly in the ordinary course of
        // walking 17.44 -> 17.50 -> 17.56. Landing on a bank edge is only
        // evidence of a clamp if the unit was pushed BACKWARD onto it.
        if (wasInRiver && !inRiver && reversed &&
            (std::fabs(b.y - RIVER_START) < 1e-4f || std::fabs(b.y - RIVER_END) < 1e-4f)) {
            v.bankSnaps++;
        }
        wasInRiver = inRiver;
    }
    return v;
}

std::string cardName(int id) {
    auto* c = CardRegistry::getInstance().getCard(id);
    return c ? c->name : ("card" + std::to_string(id));
}

struct Spawn { int cardId; float x; float y; int team; };

struct Scenario {
    std::vector<Spawn> spawns;
    int maxTicks = 600;
    int trackTeam = 0;        // whose units to follow
    bool northbound = true;   // which bank counts as "crossed"
};

// python_ai/envs/gym_wrapper.py DEFAULT_DECK -- the 2.6 Hog Cycle. Both sides
// get it so the board is the one training actually runs on.
const std::vector<int>& defaultDeck() {
    static const std::vector<int> d = { 15, 6, 25, 40, 24, 72, 33, 7 };
    return d;
}

std::vector<Track> runScenario(const Scenario& sc, double* msPerTick = nullptr) {
    GameManager game(defaultDeck(), defaultDeck());

    for (const auto& s : sc.spawns) {
        auto* card = CardRegistry::getInstance().getCard(s.cardId);
        if (!card) { std::cerr << "no card " << s.cardId << "\n"; continue; }
        card->spawnEntity(s.x, s.y, s.team, game.getBoard());
    }
    game.getBoard().commitPendingEntities();

    std::map<int, Track> tracks;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->isTower()) continue;
        if (e->team != sc.trackTeam) continue;
        Track t;
        t.entityId = e->id;
        t.card = cardName(e->cardId);
        t.path.push_back(e->position);
        tracks[e->id] = t;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int tick = 1; tick <= sc.maxTicks; ++tick) {
        game.step();
        std::map<int, Vector2D> present;
        for (const auto& e : game.getBoard().getEntities()) {
            if (!e->isAlive()) continue;
            present[e->id] = e->position;
        }
        for (auto& kv : tracks) {
            auto it = present.find(kv.first);
            if (it == present.end()) {
                if (kv.second.deathTick < 0) kv.second.deathTick = tick;
                continue;
            }
            kv.second.path.push_back(it->second);
            bool over = sc.northbound ? (it->second.y >= RIVER_END)
                                      : (it->second.y <= RIVER_START);
            if (!kv.second.crossed && over) {
                kv.second.crossed = true;
                kv.second.ticksToCross = tick;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (msPerTick) {
        *msPerTick = std::chrono::duration<double, std::milli>(t1 - t0).count() / sc.maxTicks;
    }

    std::vector<Track> out;
    for (auto& kv : tracks) out.push_back(kv.second);
    return out;
}

// ---------------------------------------------------------------- solo sweep

void soloSweep(const std::vector<int>& cards, int team) {
    // Team 1 crosses the other way: spawned in the north half, must reach the
    // south bank. Tested explicitly because the trap this harness was written
    // to find is direction-specific -- the old Catch2 sweep missed it precisely
    // by pairing each bank with only one direction of travel.
    const bool north = (team == 0);
    const float spawnY = north ? 10.0f : 23.0f;
    const float goal = north ? RIVER_END : RIVER_START;

    std::cout << "\n=== SOLO CROSSING SWEEP (team " << team << ", "
        << (north ? "northbound" : "southbound") << ") ===\n";
    std::cout << "one unit, otherwise empty board, spawned at y=" << spawnY
        << ", must reach y" << (north ? ">=" : "<=") << goal << "\n\n";
    std::cout << std::left << std::setw(16) << "card"
        << std::setw(7) << "runs"
        << std::setw(10) << "crossed"
        << std::setw(11) << "medTicks"
        << std::setw(11) << "maxStall"
        << std::setw(9) << "snaps"
        << std::setw(8) << "revs" << "\n";

    for (int cardId : cards) {
        int runs = 0, crossed = 0, maxStall = 0, snaps = 0, revs = 0;
        std::vector<int> times;
        Vector2D worstStallPos{ 0, 0 };

        for (float x = 0.5f; x <= 17.0f; x += 0.5f) {
            Scenario sc;
            sc.spawns.push_back({ cardId, x, spawnY, team });
            sc.maxTicks = 900;
            sc.trackTeam = team;
            sc.northbound = north;
            auto tracks = runScenario(sc);
            for (const auto& t : tracks) {
                Verdict v = judge(t);
                runs++;
                if (v.crossed) { crossed++; times.push_back(v.ticksToCross); }
                if (v.longestStall > maxStall) { maxStall = v.longestStall; worstStallPos = v.stallPos; }
                snaps += v.bankSnaps;
                revs += v.reversals;
            }
        }
        int med = -1;
        if (!times.empty()) { std::sort(times.begin(), times.end()); med = times[times.size() / 2]; }
        std::cout << std::left << std::setw(16) << cardName(cardId)
            << std::setw(7) << runs
            << std::setw(10) << crossed
            << std::setw(11) << med
            << std::setw(11) << maxStall
            << std::setw(9) << snaps
            << std::setw(8) << revs;
        if (maxStall > 5) {
            std::cout << "  <-- stalled at (" << std::fixed << std::setprecision(3)
                << worstStallPos.x << ", " << worstStallPos.y << ")";
        }
        std::cout << "\n";
    }
}

// --------------------------------------------------------------- crowd sweep

void crowdSweep(int cardId, int count) {
    std::cout << "\n=== CROWD CROSSING: " << count << "x " << cardName(cardId) << " ===\n";

    Scenario sc;
    for (int i = 0; i < count; ++i) {
        float x = 2.0f + static_cast<float>(i % 5);
        float y = 8.0f + static_cast<float>(i / 5) * 1.0f;
        sc.spawns.push_back({ cardId, x, y, 0 });
    }
    sc.maxTicks = 1200;
    auto tracks = runScenario(sc);

    int crossed = 0, maxStall = 0, snaps = 0;
    Vector2D worst{ 0, 0 };
    int worstTick = -1;
    std::vector<int> times;
    for (const auto& t : tracks) {
        Verdict v = judge(t);
        if (v.crossed) { crossed++; times.push_back(v.ticksToCross); }
        if (v.longestStall > maxStall) { maxStall = v.longestStall; worst = v.stallPos; worstTick = v.stallStartTick; }
        snaps += v.bankSnaps;
    }
    std::sort(times.begin(), times.end());
    std::cout << "units:        " << tracks.size() << "\n";
    std::cout << "crossed:      " << crossed << "\n";
    std::cout << "cross ticks:  ";
    for (int t : times) std::cout << t << " ";
    std::cout << "\n";
    std::cout << "longest stall " << maxStall << " ticks at ("
        << std::fixed << std::setprecision(3) << worst.x << ", " << worst.y
        << ") from tick " << worstTick << "\n";
    std::cout << "bank snaps:   " << snaps << "\n";
}

// ---------------------------------------------------------------- snap diag
//
// Prints the trajectory around every "bank snap" -- a unit that was inside the
// river band and then found itself exactly on a bank edge, i.e. clampToBoard
// moved it. Losing crossing progress that way is the second signal the solo
// sweep reports, and it needs to be looked at rather than guessed at.

void snapDiag(int cardId) {
    std::cout << "\n=== BANK SNAP DIAGNOSTIC: " << cardName(cardId) << " ===\n";
    int found = 0;
    for (float x = 0.5f; x <= 17.0f && found < 3; x += 0.5f) {
        Scenario sc;
        sc.spawns.push_back({ cardId, x, 10.0f, 0 });
        sc.maxTicks = 900;
        auto tracks = runScenario(sc);
        for (const auto& t : tracks) {
            if (judge(t).bankSnaps == 0) continue;
            bool wasInRiver = false;
            for (size_t i = 1; i < t.path.size(); ++i) {
                const Vector2D& b = t.path[i];
                bool inRiver = (b.y > RIVER_START && b.y < RIVER_END);
                if (wasInRiver && !inRiver &&
                    (std::fabs(b.y - RIVER_START) < 1e-4f || std::fabs(b.y - RIVER_END) < 1e-4f)) {
                    std::cout << "spawn x=" << std::fixed << std::setprecision(2) << x
                        << "  snap at tick " << i << ":\n";
                    for (size_t k = (i >= 4 ? i - 4 : 0); k < std::min(t.path.size(), i + 4); ++k) {
                        std::cout << "    t=" << std::setw(4) << k << "  ("
                            << std::setprecision(4) << std::setw(9) << t.path[k].x << ", "
                            << std::setw(9) << t.path[k].y << ")"
                            << (k == i ? "   <-- snapped" : "") << "\n";
                    }
                    found++;
                    break;
                }
                wasInRiver = inRiver;
            }
        }
    }
    if (!found) std::cout << "no snaps found\n";
}

// ------------------------------------------------------------------- latency

void latency() {
    std::cout << "\n=== TICK LATENCY ===\n";
    for (int n : { 1, 5, 15, 30 }) {
        Scenario sc;
        for (int i = 0; i < n; ++i) {
            sc.spawns.push_back({ 15, 2.0f + static_cast<float>(i % 14), 10.0f, 0 });
        }
        sc.maxTicks = 300;
        double ms = 0;
        runScenario(sc, &ms);
        std::cout << std::setw(3) << n << " units: " << std::fixed << std::setprecision(4)
            << ms << " ms/tick\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "all";

    // DEFAULT_DECK movers, plus tanks that are not in it, because the report
    // named tanks specifically.
    std::vector<int> movers = {
        15,  // Hog Rider      (win condition)
        6,   // Musketeer      (ranged)
        40,  // Ice Golem      (mini tank)
        24,  // Skeletons      (swarm)
        72,  // Ice Spirit     (cheap)
        2,   // Giant          (tank)
        10,  // Valkyrie
        5,   // Mini P.E.K.K.A
        41,  // Minions        (air)
    };

    if (mode == "all" || mode == "solo") { soloSweep(movers, 0); soloSweep(movers, 1); }
    if (mode == "all" || mode == "crowd") {
        crowdSweep(24, 15);
        crowdSweep(15, 10);
        crowdSweep(2, 6);
    }
    if (mode == "all" || mode == "latency") latency();
    if (mode == "snap") { snapDiag(2); snapDiag(6); snapDiag(10); }
    return 0;
}
