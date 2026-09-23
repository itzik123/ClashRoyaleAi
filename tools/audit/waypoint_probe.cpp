// Analytic probe of the waypoint function's absorbing set. Asks
// Board::getNextWaypoint directly, over a fine sweep of positions: is the
// waypoint handed back one I am already within WAYPOINT_ARRIVAL_EPS of, while
// my destination is still far? Every such position is one Troop::moveTowards
// never leaves. bridge_audit.cpp shows the symptom; this shows the set.

#include "ArenaLayout.h"
#include "Board.h"
#include "GameManager.h"
#include "LanePath.h"
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
        { "north King", Vector2D{ ArenaLayout::CENTER_X, ArenaLayout::kingY(1) } },
        { "south King", Vector2D{ ArenaLayout::CENTER_X, ArenaLayout::kingY(0) } },
    };

    std::cout << "WAYPOINT_ARRIVAL_EPS = " << eps << "\n";
    std::cout << "river band = [" << board.getRiverStart() << ", " << board.getRiverEnd() << ")\n\n";

    for (const auto& d : dests) {
        std::cout << "=== destination: " << d.label << " ===\n";
        int trapped = 0;
        float minY = 1e9f, maxY = -1e9f;

        // Sweep the two bridge columns finely through the river region,
        // positions read off the board plus a tile either side.
        const float lb = board.getLeftBridge().x, rb = board.getRightBridge().x;
        for (float bx : { lb, rb, lb - 1.0f, lb + 1.0f, rb - 1.0f, rb + 1.0f }) {
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

    // --- whole board ---
    // The same question everywhere, against a spread of destinations. Coarser
    // in y (the trap discs are 0.01 wide, so 0.002 still samples each several
    // times), but covering the whole board.
    std::cout << "=== whole-board absorbing-state sweep ===\n";
    std::vector<Vector2D> allDests = {
        { ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(1) },
        { ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(1) },
        { ArenaLayout::CENTER_X,     ArenaLayout::kingY(1) },
        { ArenaLayout::LEFT_LANE_X,  ArenaLayout::princessY(0) },
        { ArenaLayout::RIGHT_LANE_X, ArenaLayout::princessY(0) },
        { ArenaLayout::CENTER_X,     ArenaLayout::kingY(0) },
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
    std::cout << "\n";

    // --- the lane composition ---
    // At runtime LanePath::approachPoint feeds Board::getNextWaypoint; either
    // can be trap-free while the pair is not, since W1 is a point
    // getNextWaypoint is never otherwise asked about. Run three ways, because
    // the curve engages only once a lane's Princess is gone.
    std::cout << "=== lane-composition absorbing-state sweep ===\n";
    long long laneTested = 0;
    int laneTrapped = 0;

    struct Scenario { const char* label; int killTeam; bool killLeft; };
    const Scenario scenarios[] = {
        { "all towers alive",           -1, false },
        { "team 1 LEFT Princess dead",   1, true  },
        { "team 1 RIGHT Princess dead",  1, false },
    };

    for (const auto& sc : scenarios) {
        GameManager game({ 0,1,2,3,4,5,6,7 }, { 0,1,2,3,4,5,6,7 });
        Board& b = game.getBoard();
        if (sc.killTeam >= 0) {
            for (const auto& e : b.getEntities()) {
                if (!e->isTower() || e->team != sc.killTeam || e->symbol == 'R') continue;
                if (ArenaLayout::isLeftLane(e->position.x) == sc.killLeft) e->takeDamage(e->hp);
            }
            b.cleanDeadEntities(0);
        }

        int here = 0;
        for (int myTeam = 0; myTeam < 2; ++myTeam) {
            for (float x = 0.0f; x <= 17.0f; x += 0.25f) {
                for (float y = 0.0f; y <= 33.0f; y += 0.005f) {
                    Vector2D from{ x, y };
                    if (std::fabs(b.clampToBoard(from, false).y - y) > 1e-6f) continue;
                    auto objective = LanePath::laneObjective(b, myTeam, from);
                    if (!objective) continue;
                    if (from.distanceTo(objective->position) <= 1.0f) continue;
                    laneTested++;
                    Vector2D approach = LanePath::approachPoint(b, myTeam, from, objective);
                    Vector2D wp = b.getNextWaypoint(from, approach);
                    if (from.distanceTo(wp) <= eps) {
                        if (laneTrapped < 8) {
                            std::cout << "  TRAPPED at (" << std::fixed << std::setprecision(4)
                                << from.x << ", " << from.y << ") team " << myTeam
                                << "  objective=(" << objective->position.x << ", "
                                << objective->position.y << ")  waypoint=(" << wp.x
                                << ", " << wp.y << ")\n";
                        }
                        laneTrapped++; here++;
                    }
                }
            }
        }
        std::cout << "  " << sc.label << ": " << here << " absorbing\n";
    }
    std::cout << "  positions tested: " << laneTested << "\n";
    std::cout << "  absorbing states: " << laneTrapped << "\n";

    return (trapped == 0 && laneTrapped == 0) ? 0 : 1;
}
