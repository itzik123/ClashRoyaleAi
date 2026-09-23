#pragma once
#include "GameManager.h"
#include "CardRegistry.h"
#include "TimeoutRules.h"
#include <limits>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

struct EntitySnapshot {
    int id;
    float x, y;
    int hp;
    int team;
    char symbol;
    bool isFlying;
    // The unambiguous key into save()'s "cardMeta" block; symbols are shared
    // between cards (e.g. 'M' is Mini PEKKA and Monk). Negative for entities
    // without a registered card: towers use GameManager::TOWER_KING_ID /
    // TOWER_PRINCESS_ID, and death-spawn children have their own negative ids.
    // cardMeta never resolves a negative id, which is what `name` is for.
    int cardId;
    // The entity's display name, straight off the board. A death-spawn child
    // cannot be named through cardMeta, and many symbols are ambiguous;
    // applyCardMetadata already sets the name on every troop and building.
    std::string name;
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

    // Real JSON escaping: Ram Rider's symbol is '"'.
    static std::string escapeChar(char c) {
        switch (c) {
            case '"': return "\\\"";
            case '\\': return "\\\\";
            default:
                // Control characters are illegal bare in a JSON string; none
                // are used as symbols, but escape them anyway.
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    return std::string(buf);
                }
                return std::string(1, c);
        }
    }

    // escapeChar applied across a string.
    static std::string escapeString(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) out += escapeChar(c);
        return out;
    }

public:
    GameLogger(int boardWidth = 18, int boardHeight = 34)
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
            es.isFlying = entity->isFlying;
            es.cardId = entity->cardId;
            es.name = entity->name;
            snap.entities.push_back(es);
        }

        snapshots.push_back(std::move(snap));
    }

    void clear() {
        snapshots.clear();
    }

    // The match verdict from the engine's own rules, written into the replay so
    // consumers read it rather than re-deriving it. MatchRules only says
    // whether a King has died; the winner at the tick limit is TimeoutRules.
    // Computed from the final snapshot (the logger holds no Board) through
    // TimeoutRules::decide, so the rule is shared.
    std::string resultJson() const {
        if (snapshots.empty()) return "null";
        const auto& last = snapshots.back();

        int aliveCount[2] = { 0, 0 };
        int weakestHp[2] = { std::numeric_limits<int>::max(),
                             std::numeric_limits<int>::max() };
        bool kingAlive[2] = { false, false };

        for (const auto& e : last.entities) {
            if (e.hp <= 0) continue;
            if (e.team != 0 && e.team != 1) continue;
            // Towers only: a Cannon is not a crown. The snapshot has no
            // isTower(), so this keys on the reserved tower cardIds, not the
            // symbol.
            if (e.cardId != GameManager::TOWER_KING_ID &&
                e.cardId != GameManager::TOWER_PRINCESS_ID) continue;
            aliveCount[e.team]++;
            weakestHp[e.team] = std::min(weakestHp[e.team], e.hp);
            if (e.cardId == GameManager::TOWER_KING_ID) kingAlive[e.team] = true;
        }

        const bool timedOut = kingAlive[0] && kingAlive[1];
        MatchRules::Outcome outcome =
            timedOut ? TimeoutRules::decide(aliveCount, weakestHp)
                     : MatchRules::Outcome{ true, (!kingAlive[0] && !kingAlive[1]) ? -1
                                                  : (kingAlive[0] ? 1 : 0) };

        std::string reason;
        if (!timedOut) {
            reason = (outcome.loserTeam == -1) ? "Both King Towers fell the same tick"
                   : (outcome.loserTeam == 0)  ? "Blue's King Tower destroyed"
                                               : "Red's King Tower destroyed";
        } else if (outcome.loserTeam == -1) {
            reason = "Timeout - exact tie on towers and weakest-tower HP";
        } else if (aliveCount[0] != aliveCount[1]) {
            reason = "Timeout - decided on surviving tower count";
        } else {
            reason = "Timeout - decided on the weakest tower's HP";
        }

        std::ostringstream out;
        out << "{\"loserTeam\": " << outcome.loserTeam
            << ", \"timedOut\": " << (timedOut ? "true" : "false")
            << ", \"reason\": \"" << reason << "\"}";
        return out.str();
    }

    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        file << "{\n";
        file << "  \"boardWidth\": " << boardWidth << ",\n";
        file << "  \"boardHeight\": " << boardHeight << ",\n";
        file << "  \"totalTicks\": " << snapshots.size() << ",\n";
        file << "  \"result\": " << resultJson() << ",\n";

        // Card name lookup table.
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

        // Per-card display metadata, read live from CardRegistry so every
        // replay describes its own roster and the viewer keeps no table of its
        // own. Keyed by id, since symbols are shared; the viewer builds its
        // symbol lookup from this. maxHp is 0 for spells.
        file << "  \"cardMeta\": {";
        bool firstMeta = true;
        for (const auto& pair : allCards) {
            if (!firstMeta) file << ",";
            const auto& def = pair.second;
            file << "\"" << pair.first << "\":{"
                 << "\"name\":\"" << def.name << "\""
                 << ",\"symbol\":\"" << escapeChar(def.symbol) << "\""
                 << ",\"maxHp\":" << def.hp
                 << ",\"isFlying\":" << (def.isFlying ? "true" : "false")
                 << ",\"isBuilding\":" << (def.isBuilding ? "true" : "false")
                 << ",\"isSpell\":" << (def.isSpell ? "true" : "false")
                 // Rolling-sweep shape (The Log, Barbarian Barrel); 0 for every
                 // other card, which tells the viewer to draw a circle. Emitted
                 // because the viewer cannot derive engine geometry.
                 << ",\"rollWidth\":" << def.rollWidth
                 << ",\"rollRange\":" << def.rollRange
                 << "}";
            firstMeta = false;
        }
        file << "},\n";

        // The elixir phase schedule in ticks, emitted for the same reason as
        // cardMeta: the viewer's only input is this JSON, so anything it is not
        // told it must hardcode. An old replay lacks the block, and the viewer
        // then assumes no phases.
        file << "  \"elixirPhases\": {"
             << "\"doubleTick\":" << GameManager::DOUBLE_ELIXIR_TICK
             << ",\"tripleTick\":" << GameManager::TRIPLE_ELIXIR_TICK
             << "},\n";

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

            // Hands.
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
                     << ",\"symbol\":\"" << escapeChar(ent.symbol) << "\""
                     << ",\"isFlying\":" << (ent.isFlying ? "true" : "false")
                     << ",\"cardId\":" << ent.cardId
                     << ",\"name\":\"" << escapeString(ent.name) << "\"}";

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
