#pragma once
#include "Board.h"
#include "PlayerState.h"
#include "Tower.h"
#include "MatchRules.h"
#include "MatchStatistics.h"
#include <algorithm>
#include <vector>
#include <string>

class GameManager {
public:
    // Reserved cardId sentinels for Towers -- they're built directly here,
    // never through CardRegistry, so they need a stable, non-clashing id of
    // their own for stats collectors to key on instead of string-matching
    // `name`. Negative so they can never collide with a real CardRegistry id.
    static constexpr int TOWER_KING_ID = -2;
    static constexpr int TOWER_PRINCESS_ID = -3;

private:
    Board board;
    MatchStatistics stats;
    int currentTick;
    bool gameOver;
    int loserTeam;
    const float ELIXIR_REGEN_RATE = 0.035f;
    // Curriculum hook: scales the opponent's elixir regen relative to the base rate.
    // 1.0 = normal opponent, >1.0 = faster-elixir opponent for later training stages.
    float oppElixirMultiplier = 1.0f;
    // How far short of the river a non-spell placement must stay on the
    // caller's own side (Board itself only enforces the river during
    // movement/clamping, not placement).
    static constexpr float OWN_HALF_RIVER_BUFFER = 0.5f;

    std::vector<int> aiDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<int> oppDeckConfig = { 0, 1, 2, 3, 4, 5, 6, 7 };

    void addTower(float x, float y, int hp, int team, float attackRange, int damage, int attackCooldown,
        char symbol, const std::string& towerName) {
        auto tower = std::make_shared<Tower>(board.allocateId(), x, y, hp, team, attackRange, damage, attackCooldown, symbol);
        tower->name = towerName;
        tower->cardId = (symbol == 'R') ? TOWER_KING_ID : TOWER_PRINCESS_ID;
        board.addEntity(tower);
    }

public:
    PlayerState playerAI;
    PlayerState playerOpponent;

    GameManager(const std::vector<int>& aiDeck, const std::vector<int>& opponentDeck)
        : gameOver(false), loserTeam(-1) {
        aiDeckConfig = aiDeck;
        oppDeckConfig = opponentDeck;
        reset();
    }

    void setOpponentDeck(const std::vector<int>& deck) {
        oppDeckConfig = deck;
    }

    void setOpponentElixirMultiplier(float multiplier) {
        oppElixirMultiplier = std::max(0.0f, multiplier);
    }

    Board& getBoard() { return board; }
    const Board& getBoard() const { return board; }
    const MatchStatistics& getStatistics() const { return stats; }

    float getElixirAI() const { return playerAI.elixir; }
    float getElixirOpp() const { return playerOpponent.elixir; }

    bool isGameOver() const { return gameOver; }
    int getLoserTeam() const { return loserTeam; }

    const std::vector<int>& getHand(int team) const {
        return (team == 0) ? playerAI.hand : playerOpponent.hand;
    }

    float getElixir(int team) const {
        return (team == 0) ? playerAI.elixir : playerOpponent.elixir;
    }

    bool isValidPlacement(int team, float x, float y, bool isSpell, float placedRadius, bool deployAnywhere = false) const {
        float maxX = static_cast<float>(board.getWidth() - 1);
        float maxY = static_cast<float>(board.getHeight() - 1);
        if (x < 0.0f || x > maxX || y < 0.0f || y > maxY) return false;

        if (!isSpell) {
            // Miner/Goblin Drill skip the own-half restriction (they can
            // deploy anywhere on the board) but still can't overlap an
            // existing building -- that check runs unconditionally below.
            if (!deployAnywhere) {
                if (team == 0 && y > board.getRiverStart() - OWN_HALF_RIVER_BUFFER) return false;
                if (team == 1 && y < board.getRiverEnd() + OWN_HALF_RIVER_BUFFER) return false;
            }

            // Prevent placing on top of an existing building -- Clash Royale
            // forbids this outright regardless of what's being placed, so the
            // required gap is the building's own radius plus whatever
            // footprint the new card will actually spawn with.
            for (const auto& entity : board.getEntities()) {
                float r = entity->getCollisionRadius();
                if (entity->isAlive() && r > 0.0f) {
                    float dx = x - entity->position.x;
                    float dy = y - entity->position.y;
                    float requiredDist = placedRadius + r;
                    if (dx*dx + dy*dy < requiredDist*requiredDist) return false;
                }
            }
        }
        return true;
    }

    bool playCard(int team, int targetCardId, float x, float y) {
        if (gameOver) return false;

        PlayerState& player = (team == 0) ? playerAI : playerOpponent;

        int handIndex = -1;
        for (int i = 0; i < static_cast<int>(player.hand.size()); ++i) {
            if (player.hand[i] == targetCardId) {
                handIndex = i;
                break;
            }
        }

        if (handIndex == -1) return false;

        const CardDefinition* cardDef = CardRegistry::getInstance().getCard(targetCardId);
        if (!cardDef) return false;

        if (!isValidPlacement(team, x, y, cardDef->isSpell, cardDef->placementRadius, cardDef->deployAnywhere)) return false;

        int cardId = player.playCard(handIndex);
        if (cardId != -1) {
            cardDef->spawnEntity(x, y, team, board);
            board.statsEvents.notifyCardPlayed({ team, cardId, cardDef->cost, x, y, currentTick });
            return true;
        }
        return false;
    }

    void reset() {
        board = Board();
        // First line after replacing board -- board = Board() destroys the
        // old Board's statsEvents subscriber list along with it, so stats
        // needs fresh collectors subscribed to the *new* bus before
        // anything below (the addTower() spawns, the final
        // commitPendingEntities()) fires a single event.
        stats.attach(board);
        currentTick = 0;
        gameOver = false;
        loserTeam = -1;

        playerAI.initializeDeck(aiDeckConfig);
        playerOpponent.initializeDeck(oppDeckConfig);

        // King Tower is rendered as a 4x4-tile footprint (see web/viewer.html's
        // sizeInTiles), which only sits flush on whole tile boundaries when
        // centered on a half-integer coordinate (a 4-wide span covering tiles
        // i..i+3 runs from i-0.5 to i+3.5, so its center is always X.5) --
        // these were previously on whole-integer coordinates, straddling
        // tile boundaries. Corrected per-team by the actual visual offset
        // needed (the two sides weren't symmetric to begin with), not a
        // shared mirror formula. Princess Tower positions are unaffected by
        // the alignment fix (3-wide footprint, already correctly aligned);
        // the Red-side princesses moved slightly only to match the King's
        // corrected position, preserving the two teams' visual symmetry.
        addTower(8.5f, 2.5f, 4008, 0, 7.0f, 90, 10, 'R', "King Tower");
        addTower(8.5f, 28.5f, 4008, 1, 7.0f, 90, 10, 'R', "King Tower");

        addTower(3.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(14.0f, 5.0f, 2534, 0, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(3.0f, 26.0f, 2534, 1, 7.5f, 90, 8, 'P', "Princess Tower");
        addTower(14.0f, 26.0f, 2534, 1, 7.5f, 90, 8, 'P', "Princess Tower");

        board.commitPendingEntities(currentTick);
    }

    void step() {
        if (gameOver) return;

        currentTick++;
        board.currentTick = currentTick;

        playerAI.elixir = std::min(playerAI.elixir + ELIXIR_REGEN_RATE, 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + ELIXIR_REGEN_RATE * oppElixirMultiplier, 10.0f);

        board.commitPendingEntities(currentTick);

        for (auto& entity : board.getEntities()) {
            if (entity->isAlive()) entity->update(board);
        }

        // Elixir Collector: drain whatever ElixirGrantEffect accumulated
        // this tick into the real PlayerState, then reset -- same cap as
        // normal regen above.
        playerAI.elixir = std::min(playerAI.elixir + board.pendingElixirGrant[0], 10.0f);
        playerOpponent.elixir = std::min(playerOpponent.elixir + board.pendingElixirGrant[1], 10.0f);
        board.pendingElixirGrant[0] = 0.0f;
        board.pendingElixirGrant[1] = 0.0f;

        board.commitPendingEntities(currentTick);
        board.resolveCollisions();

        MatchRules::Outcome outcome = MatchRules::evaluate(board);
        if (outcome.over) {
            gameOver = true;
            loserTeam = outcome.loserTeam;
            board.statsEvents.notifyMatchEnded({ loserTeam, currentTick });
        }

        board.cleanDeadEntities(currentTick);
    }
};