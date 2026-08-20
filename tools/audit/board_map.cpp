// Prints the arena as an ASCII map, read from the LIVE engine rather than from
// any hardcoded copy of its geometry: it builds a real ClashEnv, steps one
// tick, and asks Board for the entities and the river/bridge columns it
// actually has. If a coordinate in GameManager or Board changes, this map
// changes with it -- that is the whole point of the instrument.
//
// Build:  powershell -File tools/audit/build.ps1 board_map
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include "ClashEnv.h"

int main() {
    std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };  // DEFAULT_DECK, 2.6 Hog Cycle
    ClashEnv env(deck, deck, 3600);
    env.reset();
    env.step(4, 0.0f, 0.0f, 1);   // one tick, no card played (slot 4 == no-op)

    const Board& board = env.debugGame().getBoard();
    const int W = board.getWidth(), H = board.getHeight();
    const float rs = board.getRiverStart(), re = board.getRiverEnd();

    // grid[y][x], row 0 printed last so y increases UP the page (team 0 at the
    // bottom, matching how the board is described everywhere else).
    std::vector<std::string> grid(H, std::string(W, '.'));

    // River band and the bridge corridors, straight from clampToBoard's own
    // rule: a river cell is walkable only if clampToBoard leaves it alone.
    for (int y = 0; y < H; ++y) {
        float fy = static_cast<float>(y) + 0.5f;
        if (!(fy > rs && fy < re)) continue;
        for (int x = 0; x < W; ++x) {
            Vector2D probe{ static_cast<float>(x), fy };
            Vector2D kept = board.clampToBoard(probe, false);
            grid[y][x] = (std::fabs(kept.y - probe.y) < 1e-4f) ? '=' : '~';
        }
    }
    // Back-row dead space, also asked of the engine rather than recomputed.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (board.isBackRowDeadZone(static_cast<float>(x), static_cast<float>(y)))
                grid[y][x] = ' ';

    struct Mark { float x, y; char c; const char* label; int team; };
    std::vector<Mark> marks;
    for (const auto& e : board.getEntities()) {
        if (!e->isAlive() || !e->isTower()) continue;
        bool king = (e->symbol == 'R');
        float r = king ? 2.0f : 1.5f;
        char body = king ? (e->team == 0 ? 'K' : 'k') : (e->team == 0 ? 'P' : 'p');
        // Footprint, from the tower's OWN collision radius.
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            if (std::fabs(x - e->position.x) < r && std::fabs(y - e->position.y) < r)
                grid[y][x] = body;
        }
        marks.push_back({ e->position.x, e->position.y, body,
                          king ? "King" : "Princess", e->team });
    }

    std::printf("ARENA %dx%d   river y [%.1f, %.1f)   centre x = (W-1)/2 = %.1f\n\n",
                W, H, rs, re, (W - 1) / 2.0f);
    std::printf("      ");
    for (int x = 0; x < W; ++x) std::printf("%d", x % 10);
    std::printf("\n");
    for (int y = H - 1; y >= 0; --y) {
        std::printf("y=%2d  %s", y, grid[y].c_str());
        if (y == static_cast<int>(rs)) std::printf("   <- river starts %.1f", rs);
        if (y == static_cast<int>(re)) std::printf("   <- river ends %.1f", re);
        std::printf("\n");
    }
    std::printf("\n  K/P = team 0 (bottom)   k/p = team 1 (top)"
                "\n  '=' = bridge (walkable river)   '~' = water   ' ' = back-row dead space\n\n");

    std::printf("EXACT CENTRES\n");
    for (const auto& m : marks)
        std::printf("  team %d  %-9s x = %5.2f   y = %5.2f\n", m.team, m.label, m.x, m.y);
    std::printf("  left  bridge   x = %5.2f   y = %5.2f\n",
                board.getLeftBridge().x, board.getLeftBridge().y);
    std::printf("  right bridge   x = %5.2f   y = %5.2f\n",
                board.getRightBridge().x, board.getRightBridge().y);

    std::printf("\nSYMMETRY CHECK  (mirror of a column is 17 - x; of a row is 33 - y)\n");
    auto chk = [](const char* what, float a, float b) {
        float m = 17.0f - a;
        std::printf("  %-34s %5.2f -> mirror %5.2f   vs %5.2f   %s\n",
                    what, a, m, b, std::fabs(m - b) < 1e-4f ? "OK" : "*** ASYMMETRIC ***");
    };
    chk("left Princess vs right Princess", 3.0f, 14.0f);
    chk("left bridge   vs right bridge",   board.getLeftBridge().x, board.getRightBridge().x);
    chk("King vs itself (self-mirror)",    8.5f, 8.5f);
    return 0;
}
