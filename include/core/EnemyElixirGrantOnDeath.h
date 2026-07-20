#pragma once
#include "DeathEffect.h"
#include "Board.h"

// Grants elixir to the OPPOSING team when this entity dies (Elixir Golem's
// split chain: the Golem and each Golemite/Blob tier hands the attacker's
// side a bit of free elixir on death). Mirrors ElixirGrantEffect's use of
// Board::pendingElixirGrant, just crediting 1-team instead of team.
class EnemyElixirGrantOnDeath : public IDeathEffect {
    float amount;

public:
    explicit EnemyElixirGrantOnDeath(float amount) : amount(amount) {}

    void apply(Board& board, const Vector2D& /*position*/, int team) const override {
        board.pendingElixirGrant[1 - team] += amount;
    }
};
