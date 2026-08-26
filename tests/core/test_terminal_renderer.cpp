#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "TerminalRenderer.h"
#include "Board.h"
#include <string>

// ============================================================================
// The terminal renderer's arena topography.
//
// This file exists because TerminalRenderer.h held yet another hardcoded copy
// of the bridge columns and nothing compared it to anything -- the eighth this
// project has found. CLAUDE.md keeps the running list. This one had no test at
// all, which is why it survived the 2026-08-21 re-centring that moved the
// bridges under it.
// ============================================================================

namespace {

// Is a column passable at mid-river, according to the movement code?
//
// HONEST ABOUT ITS LIMIT: clampToBoard calls isOnBridge internally, and
// riverRow() is built from isOnBridge too, so this is NOT a fully independent
// oracle -- both paths share that predicate. CLAUDE.md's "never validate a mask
// against the predicate that generated it" applies, and the genuinely
// independent anchor is the literal river row in the case below.
//
// It still earns its place: it catches the renderer diverging from the physics
// in any of the ways that are NOT the predicate -- wrong row index, wrong
// width, an off-by-one in the paint loop -- which is most of what actually went
// wrong here.
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
    // The third anchor, independent of both the renderer and the physics: the
    // real arena's river row as documented in ArenaLayout.h and used to verify
    // the observation-encoder fix. Two tiles of bridge per lane, not three.
    Board board;
    REQUIRE(TerminalRenderer::riverRow(board) == "WWBBWWWWWWWWWWBBWW");
}

TEST_CASE("the river is drawn on the row the board says it is", "[renderer]") {
    Board board;
    const int drawn = TerminalRenderer::riverRowIndex(board);
    REQUIRE(static_cast<float>(drawn) < board.getRiverEnd());
    REQUIRE(static_cast<float>(drawn + 1) >= board.getRiverEnd());
}
