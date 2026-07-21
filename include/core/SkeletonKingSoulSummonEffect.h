#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "Board.h"
#include <cmath>

// Skeleton King's "Soul Summoning": spawns baseSkeletons + however many
// souls he's collected (see CombatEntity::soulCount/soulCollectionRadius),
// scattered in a ring of spawnRadius around him, then consumes the souls.
// The real card spawns them in a sequence of 0.25s waves; this engine
// spawns the whole batch in one tick instead (same "collapse a multi-tick
// unfolding into one instant" simplification already used for Golden
// Knight's dash chain above).
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
