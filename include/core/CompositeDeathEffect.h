#pragma once
#include "DeathEffect.h"
#include <vector>
#include <memory>

// Runs several death effects in order, since CombatEntity::deathEffect is a
// single slot (e.g. Golem: split and explode).
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
