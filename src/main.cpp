#include "GameManager.h"
#include "CardRegistry.h"
#include "GameLogger.h"
#include <iostream>
#include <map>
#include <string>

void printHand(const std::string& label, const std::vector<int>& hand) {
    std::cout << label << ": ";
    for (size_t i = 0; i < hand.size(); ++i) {
        auto* card = CardRegistry::getInstance().getCard(hand[i]);
        if (card) {
            std::cout << "[" << i << "] " << card->name << "(" << static_cast<int>(card->cost) << ")";
        }
        if (i + 1 < hand.size()) std::cout << " | ";
    }
    std::cout << std::endl;
}

int main() {
    // Configure decks with proper ordering to match our manual script plays
    GameManager game(
        { 15, 6, 0, 25, 7, 24, 34, 29 },  // AI: Hog, Musketeer, Knight, Cannon, Fireball, Skeletons, IceWiz, Zap
        { 8, 2, 71, 13, 44, 32, 29, 12 }   // Opp: Barbarians, Giant, Witch, PEKKA, BabyDragon, Poison, Zap, SkelArmy
    );

    GameLogger logger;
    std::map<int, std::string> aliveEntities;

    std::cout << "--- Starting Battle Simulation ---" << std::endl;
    std::cout << "AI Deck: Hog Cycle" << std::endl;
    std::cout << "Opponent Deck: Giant Beatdown\n" << std::endl;

    logger.logTick(0, game);

    for (int tick = 1; tick <= 1800; ++tick) {

        if (tick == 1) {
            game.playerAI.elixir = 10.0f;
            game.playerOpponent.elixir = 10.0f;
        }

        // Phase 1 (Tick 30): AI plays Hog Rider right lane
        if (tick == 30) {
            game.playCard(0, 15, 14.0f, 14.0f);
        }

        // Phase 2 (Tick 50): AI backs up Hog with Musketeer
        if (tick == 50) {
            game.playCard(0, 6, 14.0f, 10.0f);
        }

        // Phase 3 (Tick 100): Opponent defends with Barbarians
        if (tick == 100) {
            game.playCard(1, 8, 14.0f, 22.0f);
        }

        // Phase 4 (Tick 300): Opponent builds Giant push left lane
        if (tick == 300) {
            game.playCard(1, 2, 3.0f, 25.0f);
        }

        // Phase 5 (Tick 350): Opponent adds Witch behind Giant
        if (tick == 350) {
            game.playCard(1, 38, 3.0f, 28.0f);
        }

        // Phase 6 (Tick 400): AI defends with Cannon + Knight
        if (tick == 400) {
            game.playCard(0, 25, 5.0f, 10.0f);
            game.playCard(0, 0, 3.0f, 12.0f);
        }

        // Phase 7 (Tick 500): AI drops Fireball on witch+giant cluster
        if (tick == 500) {
            game.playCard(0, 7, 3.0f, 17.0f);
        }

        // Phase 8 (Tick 800): Opponent drops PEKKA right lane
        if (tick == 800) {
            game.playCard(1, 13, 14.0f, 28.0f);
        }

        // Phase 9 (Tick 850): AI counters with Skeletons + Ice Wizard
        if (tick == 850) {
            game.playCard(0, 24, 14.0f, 10.0f);
            game.playCard(0, 34, 12.0f, 8.0f);
        }

        // Phase 10 (Tick 1200): Second Hog push by AI
        if (tick == 1200) {
            game.playCard(0, 15, 3.0f, 14.0f);
        }

        game.step();
        logger.logTick(tick, game);

        std::map<int, std::string> currentEntities;
        int aiPrincessHP = 0, oppPrincessHP = 0, aiKingHP = 0, oppKingHP = 0;

        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive()) continue;

            // isTower() first. 'P' and 'R' are RENDERER symbols, not
            // identities: card id 92 (X-Bow) is registered with 'P' and card
            // id 93 (Mortar) with 'R', so a deployed X-Bow was being added to
            // the Princess Tower total and a Mortar to the King's. Same alias
            // that let a Mortar answer for a King in MatchRules::evaluate;
            // here it is only a display figure, but it is the same mistake.
            if (entity->isTower()) {
                if (entity->symbol == 'R') {
                    if (entity->team == 0) aiKingHP += entity->hp;
                    else oppKingHP += entity->hp;
                } else {
                    if (entity->team == 0) aiPrincessHP += entity->hp;
                    else oppPrincessHP += entity->hp;
                }
            }

            if (entity->symbol != '-' && entity->symbol != '*' && entity->symbol != 'O'
                && entity->symbol != 'Z' && entity->symbol != 'r' && entity->symbol != 'j'
                && entity->symbol != 'n' && entity->symbol != 'o') {
                std::string displayName = entity->name.empty() ? "Unknown" : entity->name;
                std::string entityDesc = displayName + " (Team " + std::to_string(entity->team) + ")";
                currentEntities[entity->id] = entityDesc;

                if (aliveEntities.find(entity->id) == aliveEntities.end()) {
                    std::cout << "[Tick " << tick << "] SPAWN: " << entityDesc
                        << " [ID: " << entity->id << "] HP: " << entity->hp << std::endl;
                }
            }
        }

        for (const auto& pair : aliveEntities) {
            if (currentEntities.find(pair.first) == currentEntities.end()) {
                std::cout << "[Tick " << tick << "] DEATH: " << pair.second
                    << " [ID: " << pair.first << "]" << std::endl;
            }
        }

        aliveEntities = currentEntities;

        if (tick > 0 && tick % 300 == 0) {
            std::cout << "\n--- STATUS (Tick " << tick << ") ---" << std::endl;
            std::cout << "Blue Towers -> Princesses: " << aiPrincessHP << " | King: " << aiKingHP << std::endl;
            std::cout << "Red  Towers -> Princesses: " << oppPrincessHP << " | King: " << oppKingHP << std::endl;
            printHand("AI Hand  ", game.playerAI.hand);
            printHand("Opp Hand ", game.playerOpponent.hand);
            std::cout << "---\n" << std::endl;
        }

        if (game.isGameOver()) {
            int loser = game.getLoserTeam();
            std::cout << "\nKing Tower destroyed! Team " << (loser == 0 ? "Blue" : "Red") << " lost." << std::endl;
            std::cout << "Game Over at Tick " << tick << std::endl;
            break;
        }
    }

    if (logger.save("game_log.json")) {
        std::cout << "\n--- Game log saved to game_log.json ---" << std::endl;
        std::cout << "Open web/viewer.html in a browser and load this file to replay." << std::endl;
    } else {
        std::cerr << "ERROR: Failed to save game log!" << std::endl;
    }

    std::cout << "--- Simulation Ended ---" << std::endl;
    return 0;
}