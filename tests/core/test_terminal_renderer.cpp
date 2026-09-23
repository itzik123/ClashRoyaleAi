#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "TerminalRenderer.h"
#include "Board.h"
#include <string>

// The terminal renderer's arena topography, against the physics and the real
// river row.

namespace {

// Is a column passable at mid-river, according to the movement code?
//
// Not a fully independent oracle: clampToBoard and riverRow() share isOnBridge.
// It catches the renderer diverging from the physics in every other way (row
// index, width, the paint loop); the literal river row below is the independent
// anchor.
bool physicallyCrossableAt(const Board& board, int x) {
    const float midRiver = (board.getRiverStart() + board.getRiverEnd()) * 0.5f;
    const Vector2D probe{ static_cast<float>(x), midRiver };
    const Vector2D settled = board.clampToBoard(probe, /*ignoresRiver=*/false);
    return settled.y == Catch::Approx(midRiver);   // not snapped to a bank
}

} // namespace

TEST_CASE("the rendered river row matches what a troop can actually walk on",
          "[renderer][regression]") {
    Board board;
    const std::string row = TerminalRenderer::riverRow(board);
    REQUIRE(row.size() == static_cast<size_t>(board.getWidth()));

    std::string physics;
    for (int x = 0; x < board.getWidth(); ++x)
        physics += physicallyCrossableAt(board, x) ? 'B' : 'W';

    INFO("rendered: " << row);
    INFO("physics : " << physics);
    REQUIRE(row == physics);
}

TEST_CASE("the rendered river row is the real arena's row", "[renderer][regression]") {
    // The independent anchor: the real arena's river row (ArenaLayout.h). Two
    // bridge tiles per lane.
    Board board;
    REQUIRE(TerminalRenderer::riverRow(board) == "WWBBWWWWWWWWWWBBWW");
}

TEST_CASE("the river is drawn on the row the board says it is", "[renderer]") {
    Board board;
    const int drawn = TerminalRenderer::riverRowIndex(board);
    REQUIRE(static_cast<float>(drawn) < board.getRiverEnd());
    REQUIRE(static_cast<float>(drawn + 1) >= board.getRiverEnd());
}
