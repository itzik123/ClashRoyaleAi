#pragma once
#include "PeriodicEffect.h"
#include "Board.h"

// Elixir Collector: instead of spawning a child troop, a periodic effect
// that credits its own team's pending elixir grant (see
// Board::pendingElixirGrant) -- GameManager::step() drains this into the
// actual PlayerState each tick, since Board has no notion of PlayerState
// to credit directly.
class ElixirGrantEffect : public IPeriodicEffect {
    float amount;

public:
    explicit ElixirGrantEffect(float amount) : amount(amount) {}

    void apply(Board& board, const Vector2D&, int team) const override {
        board.pendingElixirGrant[team] += amount;
    }
};
