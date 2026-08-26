// Do SPAWNED units carry a real speed tier?
//
// The 2026-08-24 speed rework put every playable card on one of the five real
// tiers and round-tripped "109 / 109 match". 109 is the count of RESOLVABLE
// TROOPS -- cards with an official row. The child CardStats that death effects
// and periodic effects spawn (Golemites, Lava Pups, a Tombstone's Skeletons,
// a Barbarian Barrel's Barbarians) are not playable cards and have no row of
// their own, so whether they were in that 109 is exactly the question this
// instrument answers by measurement instead of by reading the diff.
//
// It spawns each parent, fires its death effect, and reports what actually
// reached the board -- reading Troop::getSpeed(), the value movement uses.
//
// TWO CHECKS, and the second is the one that matters. The first version of this
// tool only asked "is this speed near SOME tier", which is far weaker than it
// looks: a unit that should be MEDIUM but sits on SLOW passes it, because SLOW
// is a tier. That is exactly how a Goblin at 2.000 tiles/s hid while its own
// playable card sits at VERY_FAST (2.651). The second check asks whether a
// SPAWNED unit agrees with THE PLAYABLE CARD OF THE SAME NAME -- pure internal
// consistency, needing no external source, and it is what actually finds these.
//
// NOT a test: a measurement harness, standalone against the header-only engine
// (tools/audit/build.ps1 spawn_speed_audit).

#include "Board.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "Troop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Tier { const char* name; float perTick; };

// The five tiers as movement actually sees them: the constant, times
// MOVEMENT_SPEED_SCALE, which CardStats::troop() applies on the way in.
const std::vector<Tier>& tiers() {
    static const std::vector<Tier> t = {
        { "VERY_SLOW", SPEED_VERY_SLOW * MOVEMENT_SPEED_SCALE },
        { "SLOW",      SPEED_SLOW      * MOVEMENT_SPEED_SCALE },
        { "MEDIUM",    SPEED_MEDIUM    * MOVEMENT_SPEED_SCALE },
        { "FAST",      SPEED_FAST      * MOVEMENT_SPEED_SCALE },
        { "VERY_FAST", SPEED_VERY_FAST * MOVEMENT_SPEED_SCALE },
    };
    return t;
}

// Nearest tier and the percentage gap to it. A card ON tier reads ~0.0%.
void classify(float perTick, const char*& tierName, float& pctOff) {
    tierName = "?"; pctOff = 1e9f;
    for (const Tier& t : tiers()) {
        float d = std::fabs(perTick - t.perTick) / t.perTick * 100.0f;
        if (d < pctOff) { pctOff = d; tierName = t.name; }
    }
}

struct Row {
    std::string name;
    int cardId;
    float perTick;
    bool spawned;   // true = arrived via a death effect, false = played directly
};

void collect(Board& board, std::vector<Row>& out, bool spawned,
             const std::vector<int>& before) {
    for (const auto& e : board.getEntities()) {
        if (std::find(before.begin(), before.end(), e->id) != before.end()) continue;
        auto* troop = dynamic_cast<Troop*>(e.get());
        if (!troop) continue;
        out.push_back(Row{ e->name, e->cardId, troop->getSpeed(), spawned });
    }
}

} // namespace

int main() {
    std::printf("tier reference (tiles/tick, and tiles/s at 10 ticks/s):\n");
    for (const Tier& t : tiers())
        std::printf("  %-10s %.5f  (%.3f tiles/s)\n", t.name, t.perTick, t.perTick * 10.0f);
    std::printf("\n");

    // Every parent whose death or periodic effect puts a NEW unit on the
    // board. Enumerated by playing each registered card and then killing it.
    std::map<std::string, Row> byName;

    for (const auto& entry : CardRegistry::getInstance().getAllCards()) {
        const CardDefinition* def = &entry.second;

        Board board;
        def->spawnEntity(9.0f, 10.0f, 0, board);
        board.commitPendingEntities();

        std::vector<Row> direct;
        collect(board, direct, /*spawned=*/false, {});
        for (const Row& r : direct) byName[r.name + "#direct"] = r;

        // Ids present before the death effects fire, so anything appearing
        // afterwards is unambiguously a CHILD.
        std::vector<int> before;
        for (const auto& e : board.getEntities()) before.push_back(e->id);

        for (const auto& e : board.getEntities()) e->hp = 0;
        board.cleanDeadEntities();
        board.commitPendingEntities();

        std::vector<Row> children;
        collect(board, children, /*spawned=*/true, before);
        for (const Row& r : children) byName[r.name + "#spawned"] = r;
    }

    std::printf("%-28s %-8s %-9s %-10s %8s   %s\n",
                "unit", "cardId", "tiles/s", "nearest", "off by", "origin");
    std::printf("%s\n", std::string(84, '-').c_str());

    int offTier = 0, belowSlowest = 0, total = 0;
    const float slowest = tiers().front().perTick;

    for (const auto& kv : byName) {
        const Row& r = kv.second;
        const char* tier; float pct;
        classify(r.perTick, tier, pct);
        ++total;
        const bool bad = pct > 1.0f;
        if (bad) ++offTier;
        const bool tooSlow = r.perTick < slowest - 1e-6f;
        if (tooSlow) ++belowSlowest;
        if (!bad && !tooSlow) continue;   // on-tier units are the boring majority
        std::printf("%-28s %-8d %-9.3f %-10s %7.1f%%   %s%s\n",
                    r.name.c_str(), r.cardId, r.perTick * 10.0f, tier, pct,
                    r.spawned ? "SPAWNED" : "played",
                    tooSlow ? "   << BELOW THE SLOWEST REAL TIER" : "");
    }

    std::printf("\n%d distinct units seen, %d off-tier by more than 1%%, "
                "%d slower than SPEED_VERY_SLOW.\n", total, offTier, belowSlowest);
    std::printf("A unit below VERY_SLOW is moving slower than ANY card in the real\n"
                "game, whose published table bottoms out at 30 tiles/min.\n\n");

    // ---- the check that actually matters --------------------------------
    // A spawned unit and a playable card sharing a NAME are the same unit.
    // Where they disagree on speed the registry contradicts itself, and one of
    // the two is on a real tier -- no external source needed to call it.
    //
    // Compound cards are excluded BY NAME: Goblin Machine, Goblinstein, Ram
    // Rider and Rascals each register a secondary unit under the parent's name
    // that is deliberately a DIFFERENT creature (different archetype, range and
    // hp), not a copy of it.
    std::printf("SPAWNED vs PLAYABLE, same name:\n");
    std::printf("%-20s %-11s %-11s %s\n", "unit", "played", "spawned", "spawned is");
    std::printf("%s\n", std::string(66, '-').c_str());

    const std::vector<std::string> compound = {
        "Goblin Machine", "Goblinstein", "Ram Rider", "Rascals"
    };
    int contradictions = 0, compared = 0;
    for (const auto& kv : byName) {
        const std::string& key = kv.first;
        if (key.size() < 9 || key.substr(key.size() - 8) != "#spawned") continue;
        const std::string name = key.substr(0, key.size() - 8);
        if (std::find(compound.begin(), compound.end(), name) != compound.end()) continue;
        auto played = byName.find(name + "#direct");
        if (played == byName.end()) continue;
        ++compared;
        const float a = played->second.perTick, b = kv.second.perTick;
        if (std::fabs(a - b) < 1e-6f) continue;
        ++contradictions;
        std::printf("%-20s %-11.3f %-11.3f %+.1f%%\n",
                    name.c_str(), a * 10.0f, b * 10.0f, (b - a) / a * 100.0f);
    }
    std::printf("\n%d spawned units compared against their own card, "
                "%d CONTRADICT it.\n", compared, contradictions);
    return 0;
}
