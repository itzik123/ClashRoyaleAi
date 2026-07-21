#pragma once
#include "GameManager.h" // שמנו לב: הייבוא שונה ל-GameManager
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // מאפשר לנו לעצב את הדפסת האליקסיר עם נקודה עשרונית

class TerminalRenderer {
private:
    int width, height;

    // פונקציית עזר לתרגום מזהה קלף לשם ועלות
    std::string getCardName(int id) const {
        const auto* card = CardRegistry::getInstance().getCard(id);
        std::string name = card ? card->name : "None";
        return name;
    }

public:
    TerminalRenderer(int w = 18, int h = 34) : width(w), height(h) {}

    // הפונקציה כעת מקבלת את מנהל המשחק כולו
    void render(const GameManager& game) {
        const Board& board = game.getBoard();

        // 1. יצירת לוח ריק עם טופוגרפיה
        std::vector<std::string> grid(height, std::string(width, '.'));

        for (int x = 0; x < width; ++x) {
            if ((x >= 3 && x <= 5) || (x >= 13 && x <= 15)) {
                grid[17][x] = 'B';
            }
            else {
                grid[17][x] = 'W';
            }
        }

        // 2. מיפוי הישויות על הלוח
        for (const auto& entity : board.getEntities()) {
            if (!entity->isAlive()) continue;

            int x = static_cast<int>(entity->position.x);
            int y = static_cast<int>(entity->position.y);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                grid[y][x] = entity->symbol; // פשוט לוקח את המאפיין ישירות
            }
        }

        // 3. ניקוי המסך והדפסה
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

        // --- 4. תצוגת סטטוס השחקן (אליקסיר, יד ותור) תחת הלוח ---
        std::cout << "===============================" << std::endl;

        // הדפסת אליקסיר בסגול
        std::cout << std::fixed << std::setprecision(1); // הצגת ספרה אחת אחרי הנקודה העשרונית
        std::cout << "[\033[35mElixir: " << game.getElixirAI() << " / 10.0\033[0m] | Next: "
            << getCardName(game.playerAI.deckQueue.front()) << std::endl;

        // הדפסת 4 הקלפים שביד
        std::cout << "Hand: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "[" << i << "] " << getCardName(game.playerAI.hand[i]) << "  ";
        }
        std::cout << "\n===============================" << std::endl;
    }
};