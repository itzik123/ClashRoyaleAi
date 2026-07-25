#pragma once
#include "GameManager.h"
#include "CardRegistry.h"
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
    // Some symbols are legitimately shared by unrelated cards with
    // different maxHp (e.g. 'M' is both Mini PEKKA at 1390 and Monk at
    // 2214) -- a single-char symbol alphabet ran out of room long before
    // the card roster did. cardId is the unambiguous key into GameLogger::
    // save()'s "cardMeta" block; symbol alone is not enough to look up
    // correct metadata for every card. -1 for entities with no registry
    // entry other than towers (GameManager::TOWER_KING_ID/TOWER_PRINCESS_ID,
    // -2/-3 -- see GameManager.h), which the viewer already has separate,
    // accurate per-corner metadata for (TOWER_DEFS).
    int cardId;
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

    // Despite the name, this used to just wrap c in a 1-char string with no
    // actual escaping -- harmless as long as every symbol in use was JSON-
    // safe, until Ram Rider's own symbol turned out to be '"' (see
    // CardRegistry.h's troop(87, "Ram Rider", ...)): writing it unescaped
    // produces a bare `"` inside an already-open JSON string, corrupting
    // the whole file the moment that symbol is written -- previously only
    // when a Ram Rider entity actually appeared in a given replay's entity
    // list, now unconditionally too via cardMeta's own per-card symbol
    // (every registered card, every replay). Real JSON escaping fixes both.
    static std::string escapeChar(char c) {
        switch (c) {
            case '"': return "\\\"";
            case '\\': return "\\\\";
            default:
                // Control characters (< 0x20) are also illegal bare in a
                // JSON string; none are known to be in use as a card
                // symbol today, but \u-escape defensively rather than
                // assume that stays true.
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    return std::string(buf);
                }
                return std::string(1, c);
        }
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

        // Authoritative per-card display metadata, sourced live from
        // CardRegistry (see CardDefinition's own comment on why) instead of
        // a hand-maintained table on the viewer side -- every replay is
        // self-describing and can never drift out of sync with whatever
        // cards exist at the time it was generated, even as the roster
        // grows. Keyed by card id (same convention as cardNames above, and
        // some ids share a symbol -- e.g. a card's own death-spawn reusing
        // its parent's stats -- so this can't be symbol-keyed without
        // silently collapsing those); the viewer derives its own
        // symbol-keyed lookup from this at load time. maxHp is 0 for
        // spells (no persistent HP to bar-render).
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
                 << "}";
            firstMeta = false;
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
                     << ",\"symbol\":\"" << escapeChar(ent.symbol) << "\""
                     << ",\"isFlying\":" << (ent.isFlying ? "true" : "false")
                     << ",\"cardId\":" << ent.cardId << "}";

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
