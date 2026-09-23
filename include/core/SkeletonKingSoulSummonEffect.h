#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"
#include <cmath>

// Skeleton King's "Soul Summoning": spawns baseSkeletons plus his collected
// souls in a ring, then consumes the souls. The real card spawns in 0.25 s
// waves; here the whole batch lands in one tick.
class SkeletonKingSoulSummonEffect : public IAbilityEffect {
    CardStats skeletonStats;
    int baseSkeletons;
    float spawnRadius;

public:
    SkeletonKingSoulSummonEffect(CardStats skeletonStats, int baseSkeletons, float spawnRadius)
        : skeletonStats(std::move(skeletonStats)), baseSkeletons(baseSkeletons), spawnRadius(spawnRadius) {}

    void apply(Board& board, CombatEntity& self) const override {
        int count = baseSkeletons + self.soulCount;
        for (int i = 0; i < count; ++i) {
            float angle = (2.0f * 3.14159265f * static_cast<float>(i)) / static_cast<float>(count);
            float x = self.position.x + std::cos(angle) * spawnRadius;
            float y = self.position.y + std::sin(angle) * spawnRadius;
            CardFactories::spawn(skeletonStats, x, y, self.team, board);
        }
        self.soulCount = 0; // consumed
    }
};
