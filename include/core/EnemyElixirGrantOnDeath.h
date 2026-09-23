#pragma once
#include "DeathEffect.h"
#include "Board.h"

// Grants elixir to the OPPOSING team on death (Elixir Golem and each of its
// split tiers), via Board::pendingElixirGrant.
class EnemyElixirGrantOnDeath : public IDeathEffect {
    float amount;

public:
    explicit EnemyElixirGrantOnDeath(float amount) : amount(amount) {}

    void apply(Board& board, const Vector2D& /*position*/, int team) const override {
        board.pendingElixirGrant[1 - team] += amount;
    }
};
