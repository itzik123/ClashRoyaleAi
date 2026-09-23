#pragma once
#include "GameManager.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <iomanip> // std::setprecision

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

    // The river row, derived from the board (e.g. WWBBWWWWWWWWWWBBWW). A
    // string, so a test can check it without parsing ANSI escapes.
    static std::string riverRow(const Board& board) {
        std::string row(static_cast<size_t>(board.getWidth()), 'W');
        for (int x = 0; x < board.getWidth(); ++x) {
            if (board.isOnBridge(static_cast<float>(x))) row[static_cast<size_t>(x)] = 'B';
        }
        return row;
    }

    // The grid row the river is drawn on: the row
    // ClashEnv::extractObservationForTeam paints channel 8 on.
    static int riverRowIndex(const Board& board) {
        return static_cast<int>(board.getRiverEnd());
    }

    // Draws the board, then the AI player's elixir and hand.
    void render(const GameManager& game) {
        const Board& board = game.getBoard();

        // 1. An empty grid with the river row.
        std::vector<std::string> grid(height, std::string(width, '.'));

        const std::string river = riverRow(board);
        const int riverY = riverRowIndex(board);
        if (riverY >= 0 && riverY < height) {
            for (int x = 0; x < width && x < static_cast<int>(river.size()); ++x) {
                grid[riverY][x] = river[static_cast<size_t>(x)];
            }
        }

        // 2. Every living entity at its truncated cell.
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

        // 4. The AI player's elixir, next card and hand.
        std::cout << "===============================" << std::endl;

        // Elixir in magenta, with the next card in the queue.
        std::cout << std::fixed << std::setprecision(1);
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