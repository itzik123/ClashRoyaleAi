#pragma once
#include "GameManager.h"
#include "CardRegistry.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct EntitySnapshot {
    int id;
    float x, y;
    int hp;
    int team;
    char symbol;
};

struct TickSnapshot {
    int tick;
    float elixirAI;
    float elixirOpp;
    std::vector<EntitySnapshot> entities;
    std::vector<int> aiHand;
    std::vector<int> oppHand;
};

class GameLogger {
private:
    bool enabled;
    int boardWidth;
    int boardHeight;
    std::vector<TickSnapshot> snapshots;

    static std::string escapeChar(char c) {
        std::string s(1, c);
        return s;
    }

public:
    GameLogger(int boardWidth = 18, int boardHeight = 32)
        : enabled(true), boardWidth(boardWidth), boardHeight(boardHeight) {}

    void setEnabled(bool value) { enabled = value; }
    bool isEnabled() const { return enabled; }

    void logTick(int tick, const GameManager& game) {
        if (!enabled) return;

        TickSnapshot snap;
        snap.tick = tick;
        snap.elixirAI = game.getElixirAI();
        snap.elixirOpp = game.playerOpponent.elixir;
        snap.aiHand = game.playerAI.hand;
        snap.oppHand = game.playerOpponent.hand;

        for (const auto& entity : game.getBoard().getEntities()) {
            if (!entity->isAlive()) continue;

            EntitySnapshot es;
            es.id = entity->id;
            es.x = entity->position.x;
            es.y = entity->position.y;
            es.hp = entity->hp;
            es.team = entity->team;
            es.symbol = entity->symbol;
            snap.entities.push_back(es);
        }

        snapshots.push_back(std::move(snap));
    }

    void clear() {
        snapshots.clear();
    }

    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        file << "{\n";
        file << "  \"boardWidth\": " << boardWidth << ",\n";
        file << "  \"boardHeight\": " << boardHeight << ",\n";
        file << "  \"totalTicks\": " << snapshots.size() << ",\n";

        // Write card name lookup table
        file << "  \"cardNames\": {";
        const auto& allCards = CardRegistry::getInstance().getAllCards();
        bool firstCard = true;
        for (const auto& pair : allCards) {
            if (!firstCard) file << ",";
            file << "\"" << pair.first << "\":\"" << pair.second.name
                 << " (" << static_cast<int>(pair.second.cost) << ")\"";
            firstCard = false;
        }
        file << "},\n";

        file << "  \"ticks\": [\n";

        for (size_t t = 0; t < snapshots.size(); ++t) {
            const auto& snap = snapshots[t];
            file << "    {\n";
            file << "      \"tick\": " << snap.tick << ",\n";

            std::ostringstream elixirAI, elixirOpp;
            elixirAI << std::fixed;
            elixirAI.precision(2);
            elixirAI << snap.elixirAI;
            elixirOpp << std::fixed;
            elixirOpp.precision(2);
            elixirOpp << snap.elixirOpp;

            file << "      \"elixirAI\": " << elixirAI.str() << ",\n";
            file << "      \"elixirOpp\": " << elixirOpp.str() << ",\n";

            // Write hands
            file << "      \"aiHand\": [";
            for (size_t h = 0; h < snap.aiHand.size(); ++h) {
                if (h > 0) file << ",";
                file << snap.aiHand[h];
            }
            file << "],\n";

            file << "      \"oppHand\": [";
            for (size_t h = 0; h < snap.oppHand.size(); ++h) {
                if (h > 0) file << ",";
                file << snap.oppHand[h];
            }
            file << "],\n";

            file << "      \"entities\": [\n";

            for (size_t e = 0; e < snap.entities.size(); ++e) {
                const auto& ent = snap.entities[e];

                std::ostringstream ex, ey;
                ex << std::fixed;
                ex.precision(2);
                ex << ent.x;
                ey << std::fixed;
                ey.precision(2);
                ey << ent.y;

                file << "        {\"id\":" << ent.id
                     << ",\"x\":" << ex.str()
                     << ",\"y\":" << ey.str()
                     << ",\"hp\":" << ent.hp
                     << ",\"team\":" << ent.team
                     << ",\"symbol\":\"" << escapeChar(ent.symbol) << "\"}";

                if (e + 1 < snap.entities.size()) file << ",";
                file << "\n";
            }

            file << "      ]\n";
            file << "    }";
            if (t + 1 < snapshots.size()) file << ",";
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        file.close();
        return true;
    }
};
