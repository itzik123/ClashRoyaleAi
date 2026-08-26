// A/B for the 2026-08-26 hot-path rework of Board's two per-tick collision
// functions.
//
// NOT a test: a measurement harness, built standalone against the header-only
// engine (tools/audit/build.ps1 collision_bench) so it cannot perturb the
// generated solution the .pyd and the Catch2 suite build from.
//
// WHY BOTH IMPLEMENTATIONS LIVE IN THIS ONE FILE. The alternative -- time the
// new code, stash the diff, rebuild, time the old code -- also swaps in the
// gameplay fixes that shipped alongside (freeze duration, building expiry,
// Ice Golem's slow), so the two runs would be simulating different matches and
// the comparison would be measuring the workload, not the code. Holding the
// board fixed and swapping only the FUNCTION removes that confound entirely.
//
// The `old*` functions below are verbatim copies of what Board.h contained
// before the rework, rewritten against the public getEntities() surface. That
// surface returns a const vector of shared_ptr, and the pointees are not const,
// so they can mutate positions exactly as the originals did.
//
// Reports the MINIMUM of N repeats with the median alongside, matching
// engine_profile.cpp: the cost is deterministic and the noise is strictly
// additive, so the minimum is the least contaminated estimate and a wide
// min/median gap flags a noisy box rather than hiding inside an average.

#include "Board.h"
#include "CardRegistry.h"
#include "ClashEnv.h"
#include "Entity.h"
#include "GameManager.h"
#include "Tower.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

using Clock = std::chrono::steady_clock;

// ------------------------------------------------------------------ old code

static Vector2D oldPushAwayFrom(Vector2D pos, const Vector2D& obstacleCenter, float minDist) {
    float dx = pos.x - obstacleCenter.x;
    float dy = pos.y - obstacleCenter.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist < minDist) {
        if (dist < 0.001f) {
            dx = 1.0f; dy = 0.0f; dist = 1.0f;
        }
        float push = minDist - dist;
        pos.x += (dx / dist) * push + (dy / dist) * 0.05f;
        pos.y += (dy / dist) * push - (dx / dist) * 0.05f;
    }
    return pos;
}

static Vector2D oldResolvePositionAgainstBuildings(const Board& board, const Vector2D& pos, int entityId) {
    Vector2D resolved = pos;
    for (const auto& entity : board.getEntities()) {
        float radius = entity->getCollisionRadius();
        if (radius <= 0.0f || entity->id == entityId || !entity->isAlive()) continue;
        resolved = oldPushAwayFrom(resolved, entity->position, radius + Entity::IMPLICIT_TROOP_RADIUS);
    }
    return resolved;
}

static void oldResolveCollisions(Board& board) {
    const auto& e = board.getEntities();
    for (size_t i = 0; i < e.size(); ++i) {
        for (size_t j = i + 1; j < e.size(); ++j) {
            const auto& e1 = e[i];
            const auto& e2 = e[j];

            if (!e1->isAlive() || !e2->isAlive()) continue;

            float r1 = e1->getCollisionRadius();
            float r2 = e2->getCollisionRadius();

            bool isBuilding1 = r1 > 0.0f;
            bool isBuilding2 = r2 > 0.0f;
            bool isTroop1 = !isBuilding1 && e1->isTargetable();
            bool isTroop2 = !isBuilding2 && e2->isTargetable();

            if (isTroop1 && isTroop2 && e1->isFlying == e2->isFlying) {
                float dx = e1->position.x - e2->position.x;
                float dy = e1->position.y - e2->position.y;
                float dist = std::sqrt(dx * dx + dy * dy);
                float minRadius = 2.0f * Entity::IMPLICIT_TROOP_RADIUS;

                if (dist < minRadius) {
                    if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                    float overlap = minRadius - dist;
                    float pushX = (dx / dist) * overlap * 0.5f;
                    float pushY = (dy / dist) * overlap * 0.5f;
                    float noise = 0.01f;
                    pushX += (dy / dist) * noise;
                    pushY -= (dx / dist) * noise;
                    e1->position.x += pushX;
                    e1->position.y += pushY;
                    e2->position.x -= pushX;
                    e2->position.y -= pushY;
                }
            }

            if (isTroop1 && isBuilding2 && !e1->isFlying) {
                e1->position = oldPushAwayFrom(e1->position, e2->position, r2 + Entity::IMPLICIT_TROOP_RADIUS);
            }
            if (isTroop2 && isBuilding1 && !e2->isFlying) {
                e2->position = oldPushAwayFrom(e2->position, e1->position, r1 + Entity::IMPLICIT_TROOP_RADIUS);
            }
        }
    }
    for (const auto& entity : board.getEntities()) {
        if (entity->isAlive()) entity->clampPosition(board);
    }
}

// ------------------------------------------------------------------ fixtures

// A board shaped like a real mid-match position: the six Crown Towers, a
// couple of deployed buildings, and `troops` ground units clustered in the two
// lanes where they actually fight.
static void populate(Board& board, int troops) {
    auto addTower = [&](float x, float y, int team, char sym) {
        auto t = std::make_shared<Tower>(board.allocateId(), x, y, 2534, team, 7.5f, 90, 8, sym);
        board.addEntity(t);
    };
    addTower(8.5f, 2.5f, 0, 'R');  addTower(8.5f, 30.5f, 1, 'R');
    addTower(3.0f, 6.0f, 0, 'P');  addTower(14.0f, 6.0f, 0, 'P');
    addTower(3.0f, 27.0f, 1, 'P'); addTower(14.0f, 27.0f, 1, 'P');

    CardRegistry::getInstance().getCard(25)->spawnEntity(5.0f, 9.0f, 0, board);   // Cannon
    CardRegistry::getInstance().getCard(25)->spawnEntity(12.0f, 24.0f, 1, board); // Cannon

    // Musketeers and Hog Riders, alternating teams, spread through both lanes.
    for (int i = 0; i < troops; ++i) {
        const int team = i % 2;
        const int card = (i % 3 == 0) ? 15 : 6;
        const float x = (i % 2 == 0) ? 2.5f + (i % 5) * 0.35f : 14.5f - (i % 5) * 0.35f;
        const float y = (team == 0) ? 13.0f + (i % 7) * 0.4f : 20.0f - (i % 7) * 0.4f;
        CardRegistry::getInstance().getCard(card)->spawnEntity(x, y, team, board);
    }
    board.commitPendingEntities();
}

struct Result { double minUs; double medianUs; };

template <typename F>
static Result timeIt(int repeats, int innerCalls, F&& fn) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        for (int k = 0; k < innerCalls; ++k) fn();
        auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / innerCalls);
    }
    std::sort(samples.begin(), samples.end());
    return { samples.front(), samples[samples.size() / 2] };
}

static void report(const char* label, Result oldR, Result newR) {
    std::printf("  %-34s old %8.3f us   new %8.3f us   %5.2fx   (median %.3f / %.3f)\n",
                label, oldR.minUs, newR.minUs,
                newR.minUs > 0.0 ? oldR.minUs / newR.minUs : 0.0,
                oldR.medianUs, newR.medianUs);
}

int main() {
    const int repeats = 40;

    for (int troops : { 12, 24, 40 }) {
        // --- resolveCollisions ------------------------------------------
        // Each call pushes overlapping units apart, so the board settles as it
        // is measured. Each arm therefore gets its OWN freshly populated board:
        // measuring them in sequence against one board would hand the second
        // arm a settled position the first one paid to create.
        Result oldColl, newColl;
        size_t n = 0;
        {
            Board board; populate(board, troops);
            n = board.getEntities().size();
            oldColl = timeIt(repeats, 20, [&] { oldResolveCollisions(board); });
        }
        {
            Board board; populate(board, troops);
            newColl = timeIt(repeats, 20, [&] { board.resolveCollisions(); });
        }

        std::printf("board population: %zu entities (%d troops + 6 towers + 2 buildings)\n",
                    n, troops);

        // --- resolvePositionAgainstBuildings ----------------------------
        // Read-only, so one shared board is fine here. Called once per MOVING
        // TROOP per tick, so the per-tick cost is this number times the troop
        // count.
        Board board; populate(board, troops);
        const Vector2D probe{ 7.0f, 15.0f };
        Result oldRes = timeIt(repeats, 2000, [&] {
            volatile float sink = oldResolvePositionAgainstBuildings(board, probe, 99999).x;
            (void)sink;
        });
        Result newRes = timeIt(repeats, 2000, [&] {
            volatile float sink = board.resolvePositionAgainstBuildings(probe, 99999).x;
            (void)sink;
        });

        report("resolveCollisions (whole board)", oldColl, newColl);
        report("resolvePositionAgainstBuildings", oldRes, newRes);
        std::printf("  %-34s old %8.3f us   new %8.3f us   %5.2fx\n",
                    "  x troops = per-tick movement cost",
                    oldRes.minUs * troops, newRes.minUs * troops,
                    newRes.minUs > 0.0 ? oldRes.minUs / newRes.minUs : 0.0);
        std::printf("\n");
    }

    // --- observation construction, with the contents actually consumed ---
    //
    // engine_profile.cpp defeats dead-store removal with `if (obs.empty())`,
    // which forces the VECTOR to exist but not its CONTENTS -- everything in
    // this header is inlinable, so a compiler could in principle drop writes
    // nobody reads and make the encoder look arbitrarily cheap. This block
    // checksums every element, so no write is dead, and prints the checksum so
    // the whole loop cannot be folded away. Read the number as an UPPER bound
    // on the encoder: it includes a 13,606-element summation that the real
    // caller does not pay.
    {
        ClashEnv env({ 15, 6, 25, 40, 24, 72, 33, 7 }, { 15, 6, 25, 40, 24, 72, 33, 7 });
        env.reset();
        for (int i = 0; i < 60; ++i) env.stepSelfPlayFast(-1, 0.0f, 0.0f, -1, 0.0f, 0.0f, 1);

        double checksum = 0.0;
        Result r = timeIt(repeats, 200, [&] {
            const std::vector<float> obs = env.getObservationForTeam(0);
            double s = 0.0;
            for (float v : obs) s += v;
            checksum += s;
        });
        std::printf("observation build + full checksum: min %.4f us   median %.4f us   (checksum %.1f)\n\n",
                    r.minUs, r.medianUs, checksum);
    }

    // --- equivalence, not just speed ------------------------------------
    // A faster function that answers differently is not an optimisation. Same
    // board, same probe, both implementations: the results must be BIT
    // identical, which for the non-degenerate path they are by construction
    // (identical operands in identical order).
    {
        Board board;
        populate(board, 24);
        // Probes sitting EXACTLY on a collider's centre are excluded, and that
        // exclusion is the point rather than a fudge: the degenerate branch of
        // pushAwayFrom was itself fixed in this same pass (it under-pushed by
        // exactly 1.0 tile), so those cells are SUPPOSED to differ. Everywhere
        // else the two must agree bit for bit.
        auto onACentre = [&](const Vector2D& p) {
            for (const auto& e : board.getEntities())
                if (e->getCollisionRadius() > 0.0f && p.distanceTo(e->position) < 0.001f) return true;
            return false;
        };

        int mismatches = 0, probed = 0, skipped = 0;
        for (float x = 0.5f; x < 17.0f; x += 0.25f) {
            for (float y = 0.5f; y < 33.0f; y += 0.25f) {
                const Vector2D p{ x, y };
                if (onACentre(p)) { ++skipped; continue; }
                ++probed;
                const Vector2D a = oldResolvePositionAgainstBuildings(board, p, 99999);
                const Vector2D b = board.resolvePositionAgainstBuildings(p, 99999);
                if (a.x != b.x || a.y != b.y) ++mismatches;
            }
        }
        std::printf("resolvePositionAgainstBuildings bit-equivalence: %s (%d cells probed, %d coincident cells skipped)\n",
                    mismatches == 0 ? "IDENTICAL" : "*** DIVERGED ***", probed, skipped);
        if (mismatches) std::printf("  mismatching cells: %d\n", mismatches);
    }

    return 0;
}
