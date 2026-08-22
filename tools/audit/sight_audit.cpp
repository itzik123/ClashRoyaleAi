// Per-card sight-range coverage audit.
//
// CardStats::sightRange defaults to 5.5 ("the main standard for most cards")
// and 63 registry entries override it. This prints, for EVERY registered card,
// the sight range the engine actually hands the spawned entity -- read off the
// live CombatEntity, not parsed out of the registry source, so a card whose
// stats look right and whose entity gets something else is visible.
//
// Also checks the invariant test_sight_range.cpp pins: effective sight must be
// >= effective attack range, or a unit can attack what it cannot see.

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "ClashEnv.h"
#include "BuildingTargeter.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

int main() {
    auto& reg = CardRegistry::getInstance();
    std::vector<std::pair<int, CardDefinition>> all;
    for (const auto& [id, def] : reg.getAllCards()) all.emplace_back(id, def);
    std::sort(all.begin(), all.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    std::cout << "id\tevo\tsight\tatk\trad\teffSight\teffAtk\tok\tbt\tname\n";
    int onDefault = 0, explicitSet = 0, spells = 0, violations = 0;

    for (const auto& [id, def] : all) {
        if (def.isSpell) { spells++; continue; }
        Board board;
        def.spawnEntity(9.0f, 5.0f, 0, board);
        board.commitPendingEntities(0);
        for (const auto& e : board.getEntities()) {
            auto* ce = dynamic_cast<CombatEntity*>(e.get());
            if (!ce) continue;
            float rad = ce->getCollisionRadius();
            if (rad <= 0.0f) rad = Entity::IMPLICIT_TROOP_RADIUS;
            // vs a plain troop target of implicit radius
            float tr = Entity::IMPLICIT_TROOP_RADIUS;
            float effSight = ce->sightRange + rad + tr;
            float effAtk = ce->getAttackRange() + rad + tr;
            bool bt = dynamic_cast<BuildingTargeter*>(e.get()) != nullptr;
            bool ok = effSight >= effAtk - 1e-4f;
            if (!ok) violations++;
            if (ce->sightRange == 5.5f) onDefault++; else explicitSet++;
            std::cout << id << "\t" << (def.isEvolution ? "E" : "-") << "\t"
                      << std::fixed << std::setprecision(2)
                      << ce->sightRange << "\t" << ce->getAttackRange() << "\t"
                      << rad << "\t" << effSight << "\t" << effAtk << "\t"
                      << (ok ? "y" : "NO") << "\t" << (bt ? "BT" : "--") << "\t" << def.name << "\n";
            break;   // primary entity only
        }
    }
    std::cout << "\n# reads 5.5 (the default): " << onDefault
              << "\n# reads something else:    " << explicitSet
              << "\n# spells (no sight):       " << spells
              << "\n# sight < attack:          " << violations << "\n";
    return 0;
}
