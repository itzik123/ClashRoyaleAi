// Sensitivity check for the encoder's river row: prints the row as the physics
// sees it against the old hardcoded encoder row, and counts the columns where
// the network would have been told the wrong thing.
#include <cstdio>
#include <string>
#include "ArenaLayout.h"
#include "Board.h"

int main() {
    Board board;
    std::string physics, oldEncoder;
    for (int x = 0; x < ArenaLayout::WIDTH; ++x) {
        physics += board.isOnBridge(static_cast<float>(x)) ? 'B' : 'W';
        // The hardcoded columns the encoder used before it read
        // Board::isOnBridge.
        oldEncoder += ((x >= 3 && x <= 4) || (x >= 13 && x <= 14)) ? 'B' : 'W';
    }
    std::printf("column        012345678901234567\n");
    std::printf("physics       %s\n", physics.c_str());
    std::printf("old encoder   %s\n", oldEncoder.c_str());

    std::string diff;
    int wrong = 0, falsePos = 0, falseNeg = 0;
    for (int x = 0; x < ArenaLayout::WIDTH; ++x) {
        bool same = physics[x] == oldEncoder[x];
        diff += same ? ' ' : '^';
        if (!same) {
            wrong++;
            if (oldEncoder[x] == 'B') falsePos++; else falseNeg++;
        }
    }
    std::printf("mismatch      %s\n\n", diff.c_str());
    std::printf("columns the network was told wrong: %d"
                "  (%d water shown as bridge, %d bridge shown as water)\n",
                wrong, falsePos, falseNeg);
    return wrong > 0 ? 0 : 1;   // 0 == the test has something to catch
}
