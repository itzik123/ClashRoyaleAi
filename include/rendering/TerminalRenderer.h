#pragma once
#include "GameManager.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <iomanip> // std::setprecision, for the one-decimal elixir readout

class TerminalRenderer {
private:
    int width, height;

    // Card id -> display name ("None" for an id the registry does not know).
    std::string getCardName(int id) const {
        const auto* card = CardRegistry::getInstance().getCard(id);
        std::string name = card ? card->name : "None";
        return name;
    }

public:
    TerminalRenderer(int w = 18, int h = 34) : width(w), height(h) {}

    // The river row, DERIVED from the board instead of restated.
    //
    // This was `(x >= 3 && x <= 5) || (x >= 13 && x <= 15)` -- another stale
    // copy of the arena's bridge columns, the EIGHTH this project has found
    // (CLAUDE.md keeps the list; the seventh was python_ai/envs/
    // scenario_offense.py, found the same day).
    // The real river row is
    //
    //     column  012345678901234567
    //             WWBBWWWWWWWWWWBBWW      (W water, B bridge)
    //
    // and the old test painted B at {3,4,5,13,14,15} against a real
    // {2,3,14,15}: FOUR of eighteen columns wrong in both directions -- column
    // 2 is real bridge and was drawn as water, while 4, 5 and 13 are water and
    // were drawn as bridge. It also drew a THREE-tile bridge, the
    // pre-2026-08-21 shape from before the bridges were re-centred on the seam
    // between their two tiles.
    //
    // Unlike web/viewer.html, which CLAUDE.md records as structurally unable to
    // derive this (its only input is a replay JSON that carries no geometry),
    // this renderer holds a `const Board&` and `Board::isOnBridge` is public.
    // There was never a reason for the copy.
    //
    // Public and string-returning so a test can check it without parsing ANSI
    // escapes out of stdout.
    static std::string riverRow(const Board& board) {
        std::string row(static_cast<size_t>(board.getWidth()), 'W');
        for (int x = 0; x < board.getWidth(); ++x) {
            if (board.isOnBridge(static_cast<float>(x))) row[static_cast<size_t>(x)] = 'B';
        }
        return row;
    }

    // Which grid row the river is drawn on. getRiverEnd() is 17.5, so this is
    // 17 -- the same row ClashEnv::extractObservationForTeam paints channel 8
    // on, derived rather than restated so both follow if the river moves again.
    static int riverRowIndex(const Board& board) {
        return static_cast<int>(board.getRiverEnd());
    }

    // Draws the whole match state: the board, then the AI player's elixir and hand.
    void render(const GameManager& game) {
        const Board& board = game.getBoard();

        // 1. An empty grid, with the river row derived from the board.
        std::vector<std::string> grid(height, std::string(width, '.'));

        const std::string river = riverRow(board);
        const int riverY = riverRowIndex(board);
        if (riverY >= 0 && riverY < height) {
            for (int x = 0; x < width && x < static_cast<int>(river.size()); ++x) {
                grid[riverY][x] = river[static_cast<size_t>(x)];
            }
        }

        // 2. Place every living entity on the grid by its truncated cell.
        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive()) continue;

            int x = static_cast<int>(entity->position.x);
            int y = static_cast<int>(entity->position.y);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                grid[y][x] = entity->symbol;
            }
        }

        // 3. Clear the terminal and print the grid, top row first.
        std::cout << "\033[2J\033[1;1H";
        std::cout << "=== Micro Royale Simulation ===" << std::endl;

        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                char c = grid[y][x];
                if (c != '.' && c != 'W' && c != 'B' && c != '*' && c != 'O' && c != '-') {
                    int team = -1;
                    for (const auto& e : board.getEntities()) {
                        if (static_cast<int>(e->position.x) == x && static_cast<int>(e->position.y) == y && e->isAlive()) {
                            team = e->team; break;
                        }
                    }
                    if (team == 0) std::cout << "\033[1;34m" << c << "\033[0m ";
                    else if (team == 1) std::cout << "\033[1;31m" << c << "\033[0m ";
                }
                else if (c == 'W') {
                    std::cout << "\033[36m" << "W" << "\033[0m ";
                }
                else if (c == 'B') {
                    std::cout << "\033[33m" << "B" << "\033[0m ";
                }
                else {
                    std::cout << "\033[90m" << "." << "\033[0m ";
                }
            }
            std::cout << std::endl;
        }

        // --- 4. The AI player's status under the board: elixir, next card, hand ---
        std::cout << "===============================" << std::endl;

        // Elixir in magenta, with the next card in the queue.
        std::cout << std::fixed << std::setprecision(1); // one decimal place
        std::cout << "[\033[35mElixir: " << game.getElixirAI() << " / 10.0\033[0m] | Next: "
            << getCardName(game.playerAI.deckQueue.front()) << std::endl;

        // The four cards in hand, by slot.
        std::cout << "Hand: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "[" << i << "] " << getCardName(game.playerAI.hand[i]) << "  ";
        }
        std::cout << "\n===============================" << std::endl;
    }
};