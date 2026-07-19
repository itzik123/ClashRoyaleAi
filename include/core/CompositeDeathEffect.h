#pragma once
#include "DeathEffect.h"
#include <vector>
#include <memory>

// Runs several death effects in sequence (e.g. Golem: splits into two
// Golemites AND deals a death-explosion) -- CombatEntity::deathEffect is a
// single slot, so a card needing more than one on-death behavior composes
// them here instead of that slot growing into a vector everywhere else.
class CompositeDeathEffect : public IDeathEffect {
    std::vector<std::shared_ptr<IDeathEffect>> effects;

public:
    explicit CompositeDeathEffect(std::vector<std::shared_ptr<IDeathEffect>> effects)
        : effects(std::move(effects)) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        for (const auto& effect : effects) {
            effect->apply(board, position, team);
        }
    }
};
