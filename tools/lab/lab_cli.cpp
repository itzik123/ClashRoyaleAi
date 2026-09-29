// Native driver for the Reflex Lab engine (web/lab/engine/lab_engine.h).
//
//   lab_cli probe                      the pinned numbers; exit 1 on a mismatch
//   lab_cli ref < reference_lines      reproduce tools/lab/pyd_reference.py exactly
//   lab_cli bench [att def]            rollouts per second
//   lab_cli table att def out_prefix   every spawn x cell x delay -> .bin, .surv.bin, .json
//   lab_cli check att def              stalls and tower reach over sampled rollouts
//   lab_cli serve                      line protocol on stdin/stdout (dev backend)
//   lab_cli trace att x y              an attacker alone, tick by tick
//
// Built by tools/lab/build_cli.ps1 (cl.exe against the header-only engine,
// like tools/audit -- deliberately not a CMake target).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "lab_engine.h"
#include "Troop.h"

using lab::LabEngine;

static int fails = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++fails;
}

static int spawnIndex(const LabEngine& e, int x, int y) {
    for (int i = 0; i < e.spawnCount(); ++i)
        if (e.spawnAt(i).x == x && e.spawnAt(i).y == y) return i;
    return -1;
}

static int cellIndex(const LabEngine& e, int x, int y) {
    for (int i = 0; i < e.cellCount(); ++i)
        if (e.cellAt(i).x == x && e.cellAt(i).y == y) return i;
    return -1;
}

// Numbers measured through the .pyd by tools/lab/pyd_reference.py (same
// placements, written independently, no early termination): Hog Rider (15)
// dropped at (14,20) for team 1, Cannon (25) for team 0, both with the real
// deploy time. `lab_cli ref` checks every case that script prints.
static int probe() {
    LabEngine e(1);
    e.setMatchup(15, 25);
    std::printf("Hog Rider vs Cannon: %d spawns, %d cells\n", e.spawnCount(), e.cellCount());
    const int s = spawnIndex(e, 14, 20);
    expect(s >= 0, "spawn (14,20) is in the band");
    if (s < 0) return 1;
    const int c = cellIndex(e, 11, 9);
    expect(c >= 0, "cell (11,9) is legal for the Cannon");
    expect(e.d0(s) == 2534.0f, "no defence: 2534 (whole Princess Tower), got " + std::to_string(e.d0(s)));
    const float at0 = e.rolloutAtTick(s, c, 0), at20 = e.rolloutAtTick(s, c, 20),
                at33 = e.rolloutAtTick(s, c, 33);
    expect(at0 == 0.0f, "Cannon (11,9) at once: 0, got " + std::to_string(at0));
    expect(at20 == 0.0f, "Cannon (11,9) 2.0 s late: 0, got " + std::to_string(at20));
    expect(at33 == 951.0f, "Cannon (11,9) 3.3 s late: 951, got " + std::to_string(at33));
    // (14,8) is inside the Princess Tower's clearance: inject would place it,
    // playCard would not, and neither does the lab.
    expect(cellIndex(e, 14, 8) < 0, "cell (14,8) is illegal for the Cannon (tower clearance)");
    const int c2 = cellIndex(e, 14, 9);
    expect(c2 >= 0 && e.rolloutAtTick(s, c2, 10) == 0.0f, "Cannon (14,9) 1 s late: 0");
    expect(e.rollout(s, c, 4) == at20, "rollout's delay index 4 is tick 20");

    // Determinism and no hidden history: the same rollout, fresh and after
    // a thousand others, agrees exactly.
    const float first = e.rollout(s, c, 4);
    std::mt19937 rng(7);
    for (int i = 0; i < 1000; ++i)
        e.rollout(int(rng() % e.spawnCount()), int(rng() % e.cellCount()), int(rng() % lab::DELAY_STEPS));
    expect(e.rollout(s, c, 4) == first, "a rollout after 1000 others equals a fresh one");
    LabEngine e2(1);
    e2.setMatchup(15, 25);
    expect(e2.rollout(s, c, 4) == first, "a second engine instance agrees");

    // Survival: how much of the defender stands when the rollout ends.
    expect(e.outcomeAtTick(s, lab::NO_DEFENCE, 0).survival == 0.0f, "no defence: survival 0");
    // (11,9) 2.0 s late saves the whole tower and the Hog destroys the Cannon
    // doing it; 3.3 s late the tower takes hits and the Cannon lives. Measured
    // 2026-09-29.
    const lab::Outcome o20 = e.outcomeAtTick(s, c, 20), o33 = e.outcomeAtTick(s, c, 33);
    expect(o20.damage == at20 && o33.damage == at33, "outcome's damage is the rollout's");
    expect(o20.survival == 0.0f, "Cannon (11,9) 2.0 s late is destroyed: survival 0, got " +
                                     std::to_string(o20.survival));
    expect(o33.survival > 0.5f && o33.survival < 1.0f,
           "Cannon (11,9) 3.3 s late survives, hit and decaying: " + std::to_string(o33.survival));
    // A Cannon the Hog walks past is never hit, but it decays for as long as
    // the Hog lives: well above 0, below 1.
    const int corner = cellIndex(e, 1, 1);
    expect(corner >= 0, "cell (1,1) is legal for the Cannon");
    if (corner >= 0) {
        const lab::Outcome oc = e.outcomeAtTick(s, corner, 20);
        expect(oc.survival > 0.2f && oc.survival < 1.0f,
               "a Cannon out of the Hog's way only decays: " + std::to_string(oc.survival));
    }
    // The live stepper is the same rollout, one tick at a time.
    for (int dropTick : {0, 7, 20, 33}) {
        e.liveStart(s);
        e.liveStepJson(dropTick);
        e.livePlace(c);
        while (!e.liveDone()) e.liveStepJson(10);
        const lab::Outcome want = e.outcomeAtTick(s, c, dropTick);
        expect(e.liveDamage() == want.damage && e.liveSurvival() == want.survival,
               "live placement at tick " + std::to_string(dropTick) + " equals the rollout (" +
                   std::to_string(e.liveDamage()) + ", survival " + std::to_string(e.liveSurvival()) + ")");
    }
    std::printf(fails ? "\n%d failure(s)\n" : "\nall probe checks passed\n", fails);
    return fails ? 1 : 0;
}

// Reads pyd_reference.py's lines on stdin and reproduces each one.
static int ref() {
    std::string line;
    int n = 0;
    LabEngine e(1);
    int curA = -1, curD = -1;
    while (std::getline(std::cin, line)) {
        for (char& ch : line)
            if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
        std::istringstream in(line);
        int a, sx, sy, d, cx, cy, drop;
        float want;
        if (!(in >> a >> sx >> sy >> d >> cx >> cy >> drop >> want)) continue;
        if (a != curA || d != curD) { e.setMatchup(a, d); curA = a; curD = d; }
        const int s = spawnIndex(e, sx, sy);
        const int c = cx < 0 ? lab::NO_DEFENCE : cellIndex(e, cx, cy);
        std::ostringstream what;
        what << "att " << a << " (" << sx << "," << sy << ") def " << d << " (" << cx << "," << cy
             << ") drop " << drop << ": .pyd " << want;
        if (s < 0 || (cx >= 0 && c < 0)) { expect(false, what.str() + " -- spawn or cell not in LabEngine's sets"); continue; }
        const float got = e.rolloutAtTick(s, c, drop);
        what << ", LabEngine " << got;
        expect(got == want, what.str());
        ++n;
    }
    std::printf(fails ? "\n%d failure(s) in %d cases\n" : "\nall %d reference cases reproduced\n", fails ? fails : n, n);
    return fails || n == 0 ? 1 : 0;
}

static int bench(int att, int def) {
    LabEngine e(1);
    e.setMatchup(att, def);
    std::mt19937 rng(3);
    const int n = 2000;
    auto t0 = std::chrono::steady_clock::now();
    double sum = 0;
    for (int i = 0; i < n; ++i)
        sum += e.rollout(int(rng() % e.spawnCount()), int(rng() % e.cellCount()), int(rng() % lab::DELAY_STEPS));
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d rollouts in %.2f s: %.3f ms each, %.0f per second (mean damage %.1f)\n", n, dt,
                1000.0 * dt / n, n / dt, sum / n);
    return 0;
}

// For every spawn x cell x delay, row-major in that order: the share of the
// no-defence damage prevented, (d0 - d) / d0 clipped to [-1, 1] (.bin), and
// how much of the defender survived (.surv.bin). The learner's reward is made
// from the two by web/lab/learner.js's Learner.reward, as on the page.
static int table(int att, int def, const std::string& prefix) {
    LabEngine e(1);
    const std::string matchup = e.setMatchup(att, def);
    const int S = e.spawnCount(), C = e.cellCount(), D = lab::DELAY_STEPS;
    std::vector<float> r(size_t(S) * C * D), sv(r.size());
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < S; ++s)
        for (int c = 0; c < C; ++c)
            for (int d = 0; d < D; ++d) {
                const lab::Outcome o = e.outcome(s, c, d);
                const size_t k = (size_t(s) * C + c) * D + d;
                r[k] = std::max(-1.0f, std::min(1.0f, (e.d0(s) - o.damage) / e.d0(s)));
                sv[k] = o.survival;
            }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream bin(prefix + ".bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(r.data()), std::streamsize(r.size() * sizeof(float)));
    std::ofstream surv(prefix + ".surv.bin", std::ios::binary);
    surv.write(reinterpret_cast<const char*>(sv.data()), std::streamsize(sv.size() * sizeof(float)));
    std::ofstream js(prefix + ".json");
    js << "{\"matchup\":" << matchup << ",\"shape\":[" << S << "," << C << "," << D
       << "],\"seconds\":" << dt << "}";
    std::printf("%s: %d x %d x %d = %zu rollouts in %.1f s\n", prefix.c_str(), S, C, D, r.size(), dt);
    return 0;
}

// Stalls (the soak's criterion: a troop past deploy, stationary for
// STALL_TICKS with nothing within its own reach) and whether the attacker
// reaches a tower, over no-defence rollouts from every spawn plus sampled
// defended ones.
static int check(int att, int def) {
    constexpr int STALL_TICKS = 30;
    LabEngine e(1);
    e.setMatchup(att, def);
    std::mt19937 rng(11);
    int rollouts = 0, stalls = 0;
    std::string firstStall;
    auto observe = [&](int s, int c, int dropTick) {
        std::unordered_map<int, std::pair<Vector2D, int>> still;
        bool stalled = false;
        e.simulateWith(s, c, dropTick, [&](ClashEnv& env, int t) {
            std::vector<std::shared_ptr<Entity>> living;
            for (const auto& en : env.debugGame().getBoard().getEntities())
                if (en->isAlive()) living.push_back(en);
            for (const auto& en : living) {
                auto troop = std::dynamic_pointer_cast<Troop>(en);
                if (!troop || troop->deployTicksRemaining > 0 || troop->getSpeed() <= 0.0f) continue;
                float slack = 1e9f;
                for (const auto& o : living) {
                    if (o->team == en->team) continue;
                    if (!o->isTargetable() && !o->isTower()) continue;
                    float r = o->getCollisionRadius();
                    if (r <= 0.0f) r = Entity::IMPLICIT_TROOP_RADIUS;
                    const float reach = troop->getAttackRange() + Entity::IMPLICIT_TROOP_RADIUS + r;
                    slack = std::min(slack, en->position.distanceTo(o->position) - reach);
                }
                auto it = still.find(en->id);
                if (it == still.end()) { still[en->id] = {en->position, 0}; continue; }
                const bool idle = it->second.first.distanceTo(en->position) < 1e-4f && slack > 0.0f;
                it->second = {en->position, idle ? it->second.second + 1 : 0};
                if (it->second.second == STALL_TICKS && !stalled) {
                    stalled = true;
                    std::ostringstream o;
                    o << en->name << " at (" << en->position.x << "," << en->position.y << ") t=" << t
                      << " spawn=(" << e.spawnAt(s).x << "," << e.spawnAt(s).y << ")";
                    if (c >= 0) o << " cell=(" << e.cellAt(c).x << "," << e.cellAt(c).y << ") drop=" << dropTick;
                    if (firstStall.empty()) firstStall = o.str();
                }
            }
        });
        ++rollouts;
        if (stalled) ++stalls;
    };
    for (int s = 0; s < e.spawnCount(); ++s) {
        observe(s, lab::NO_DEFENCE, 0);
        for (int k = 0; k < 6; ++k)
            observe(s, int(rng() % e.cellCount()), int(rng() % lab::DELAY_STEPS) * lab::DELAY_TICK_STEP);
    }
    // Reach: the band before dropping zero-damage spawns vs after.
    std::printf("{\"attacker\":%d,\"defender\":%d,\"spawns\":%d,\"rollouts\":%d,\"stalls\":%d,"
                "\"firstStall\":\"%s\"}\n",
                att, def, e.spawnCount(), rollouts, stalls, lab::jsonEscape(firstStall).c_str());
    return 0;
}

// What an attacker does on its own, tick by tick: when its bodies appear and
// die, when the tower is first hit, and the damage of each hit. For checking a
// card against the real one before it ships.
static int trace(int att, int sx, int sy) {
    LabEngine e(1);
    e.setMatchup(att, 25);
    const int s = spawnIndex(e, sx, sy);
    if (s < 0) { std::fprintf(stderr, "(%d,%d) is not a spawn for card %d\n", sx, sy, att); return 2; }
    std::unordered_map<int, std::string> seen;
    int lastHp = -1;
    const float dmg = e.simulateWith(s, lab::NO_DEFENCE, 0, [&](ClashEnv& env, int t) {
        std::unordered_map<int, bool> alive;
        for (const auto& en : env.debugGame().getBoard().getEntities()) {
            if (!en->isAlive() || en->team != 1 || en->isTower()) continue;
            alive[en->id] = true;
            if (!seen.count(en->id)) {
                seen[en->id] = en->name;
                std::printf("t=%3d  %s #%d appears at (%.2f, %.2f) hp %d\n", t, en->name.c_str(), en->id,
                            en->position.x, en->position.y, en->hp);
            }
        }
        for (auto it = seen.begin(); it != seen.end();) {
            if (!alive.count(it->first)) {
                std::printf("t=%3d  %s #%d gone\n", t, it->second.c_str(), it->first);
                it = seen.erase(it);
            } else {
                ++it;
            }
        }
        const int hp = lab::towerHpSum(env, 0);
        if (lastHp >= 0 && hp < lastHp) std::printf("t=%3d  tower hit: -%d\n", t, lastHp - hp);
        lastHp = hp;
    });
    std::printf("total tower damage %.0f\n", dmg);
    return 0;
}

// One request per line, one response per line. Errors: "error <message>".
static int serve() {
    std::ios::sync_with_stdio(false);
    LabEngine e(1);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        try {
            if (cmd == "matchup") {
                int a, d;
                in >> a >> d;
                std::cout << e.setMatchup(a, d) << "\n";
            } else if (cmd == "arena") {
                std::cout << e.arenaJson() << "\n";
            } else if (cmd == "cards") {
                std::vector<int> ids;
                int id;
                while (in >> id) ids.push_back(id);
                std::cout << e.cardsJson(ids) << "\n";
            } else if (cmd == "batch") {
                int n;
                in >> n;
                std::vector<int> sca(size_t(n) * 3);
                for (auto& v : sca) in >> v;
                // damage survival, damage survival, ...: the WASM build's layout
                std::vector<float> dmg(n), surv(n);
                e.rolloutBatch(sca.data(), n, dmg.data(), surv.data());
                // 9 significant digits round-trip a float exactly (parity.mjs).
                const auto prec = std::cout.precision(9);
                for (int i = 0; i < n; ++i) std::cout << (i ? " " : "") << dmg[i] << " " << surv[i];
                std::cout << "\n";
                std::cout.precision(prec);
            } else if (cmd == "frames") {
                int s, c, t;
                in >> s >> c >> t;
                std::cout << e.framesJson(s, c, t) << "\n";
            } else if (cmd == "live_start") {
                int s;
                in >> s;
                e.liveStart(s);
                std::cout << "ok\n";
            } else if (cmd == "live_step") {
                int n;
                in >> n;
                std::cout << e.liveStepJson(n) << "\n";
            } else if (cmd == "live_place") {
                int c;
                in >> c;
                std::cout << (e.livePlace(c) ? "true" : "false") << "\n";
            } else if (cmd == "live_state") {
                std::cout << e.liveStateJson() << "\n";
            } else {
                std::cout << "error unknown command " << cmd << "\n";
            }
        } catch (const std::exception& ex) {
            std::cout << "error " << ex.what() << "\n";
        }
        std::cout.flush();
    }
    return 0;
}

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "probe";
    try {
        if (cmd == "probe") return probe();
        if (cmd == "ref") return ref();
        if (cmd == "bench") return bench(argc > 3 ? std::atoi(argv[2]) : 15, argc > 3 ? std::atoi(argv[3]) : 25);
        if (cmd == "table" && argc > 4) return table(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]);
        if (cmd == "check" && argc > 3) return check(std::atoi(argv[2]), std::atoi(argv[3]));
        if (cmd == "serve") return serve();
        if (cmd == "trace" && argc > 4) return trace(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
    std::fprintf(stderr, "usage: lab_cli probe | bench [att def] | table att def prefix | check att def | serve\n");
    return 2;
}
