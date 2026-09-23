#pragma once
#include "PeriodicEffect.h"
#include "Board.h"

// Elixir Collector: credits Board::pendingElixirGrant, which GameManager::step
// drains into the PlayerState each tick.
class ElixirGrantEffect : public IPeriodicEffect {
    float amount;

public:
    explicit ElixirGrantEffect(float amount) : amount(amount) {}

    void apply(Board& board, const Vector2D&, int team) const override {
        board.pendingElixirGrant[team] += amount;
    }
};
