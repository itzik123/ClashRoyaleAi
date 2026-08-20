// Analytic probe of the waypoint function's ABSORBING SET.
//
// Independent of any card: asks Board::getNextWaypoint directly, over a fine
// sweep of positions, "is the waypoint you just handed me one I am already
// within WAYPOINT_ARRIVAL_EPS of, while my real destination is still far
// away?" Every such position is a state Troop::moveTowards refuses to leave --
// position unchanged, so waypoint unchanged, so stuck forever.
//
// This is the direct proof of the mechanism. The card-level sweep in
// bridge_audit.cpp shows the SYMPTOM; this shows the SET.

#include "Board.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main() {
    Board board;
    const float eps = Board::WAYPOINT_ARRIVAL_EPS;

    // Two destinations that force a crossing: a tower on each side.
    struct Dest { const char* label; Vector2D pos; };
    std::vector<Dest> dests = {
        { "north tower (y=30.5)", Vector2D{ 9.0f, 30.5f } },
        { "south tower (y=2.5)",  Vector2D{ 9.0f,  2.5f } },
    };

    std::cout << "WAYPOINT_ARRIVAL_EPS = " << eps << "\n";
    std::cout << "river band = [" << board.getRiverStart() << ", " << board.getRiverEnd() << ")\n\n";

    for (const auto& d : dests) {
        std::cout << "=== destination: " << d.label << " ===\n";
        int trapped = 0;
        float minY = 1e9f, maxY = -1e9f;

        // Sweep the two bridge columns finely through the whole river region.
        for (float bx : { 4.0f, 14.0f, 3.0f, 5.0f, 13.0f, 15.0f }) {
            for (float y = 14.0f; y <= 19.0f; y += 0.0005f) {
                Vector2D pos{ bx, y };
                // Only positions a unit can legally occupy.
                Vector2D legal = board.clampToBoard(pos, false);
                if (std::fabs(legal.y - pos.y) > 1e-6f) continue;

                Vector2D wp = board.getNextWaypoint(pos, d.pos);
                float toWp = pos.distanceTo(wp);
                float toDest = pos.distanceTo(d.pos);

                // Refuses to move (toWp <= eps) but has not arrived (toDest large).
                if (toWp <= eps && toDest > 1.0f) {
                    if (trapped < 6) {
                        std::cout << "  TRAPPED at (" << std::fixed << std::setprecision(4)
                            << pos.x << ", " << pos.y << ")  waypoint=(" << wp.x << ", " << wp.y
                            << ")  distToWaypoint=" << toWp
                            << "  distToDest=" << std::setprecision(2) << toDest << "\n";
                    }
                    trapped++;
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                }
            }
        }
        if (trapped == 0) {
            std::cout << "  none\n";
        }
        else {
            std::cout << "  ... " << trapped << " trapped sample positions, y in ["
                << std::fixed << std::setprecision(4) << minY << ", " << maxY << "]\n";
        }
        std::cout << "\n";
    }

    // ---- whole-board generalization ----
    //
    // The sweeps above target the place the bug was found. This one asks the
    // same question everywhere, against a spread of destinations, so a trap
    // somewhere nobody thought to look still gets reported. Coarser in y (the
    // trap discs are 0.01 wide, so 0.002 still lands several samples inside
    // one) but it covers the whole 18x34 board.
    std::cout << "=== whole-board absorbing-state sweep ===\n";
    std::vector<Vector2D> allDests = {
        { 4.0f, 27.0f }, { 14.0f, 27.0f }, { 9.0f, 30.5f },
        { 4.0f,  6.0f }, { 14.0f,  6.0f }, { 9.0f,  2.5f },
        { 0.0f, 16.0f }, { 17.0f, 16.0f },
    };
    long long tested = 0;
    int trapped = 0;
    for (const auto& dest : allDests) {
        for (float x = 0.0f; x <= 17.0f; x += 0.25f) {
            for (float y = 0.0f; y <= 33.0f; y += 0.002f) {
                Vector2D pos{ x, y };
                Vector2D legal = board.clampToBoard(pos, false);
                if (std::fabs(legal.y - pos.y) > 1e-6f) continue;   // in the river, off-bridge
                float toDest = pos.distanceTo(dest);
                if (toDest <= 1.0f) continue;                        // legitimately arrived
                tested++;
                Vector2D wp = board.getNextWaypoint(pos, dest);
                if (pos.distanceTo(wp) <= eps) {
                    if (trapped < 8) {
                        std::cout << "  TRAPPED at (" << std::fixed << std::setprecision(4)
                            << pos.x << ", " << pos.y << ")  dest=(" << dest.x << ", " << dest.y
                            << ")  waypoint=(" << wp.x << ", " << wp.y << ")\n";
                    }
                    trapped++;
                }
            }
        }
    }
    std::cout << "  positions tested: " << tested << "\n";
    std::cout << "  absorbing states: " << trapped << "\n";
    return trapped == 0 ? 0 : 1;
}
