#pragma once
#include "CardStats.h"

// Spirit Empress's dual ground/flying forms -- see GameManager::playCard's
// own SPIRIT_EMPRESS_CARD_ID branch for the dynamic elixir-cost/form
// selection at play time. Both forms share the same placement footprint
// (MeleeSquad/RangedSquad both resolve to Entity::IMPLICIT_TROOP_RADIUS
// via CardFactories::placementRadius, and neither is a spell or deploy-
// anywhere), so unlike Mirror, no placement-rule substitution is needed
// -- only cost and spawn stats vary. Both forms share id 165 (Spirit
// Empress's own registered id) so spawned entities are attributed to
// "Spirit Empress" regardless of which form was actually played.
//
// Confirmed (Liquipedia, level 9): both forms 926hp/249dmg. Ground: melee
// (1.2 range), 1.2s hit speed, fast (speed 90), ground-only, 3 elixir.
// Flying: 5 range, 1.4s hit speed, medium (speed 60), targets air+ground,
// 6 elixir. The exact switching RULE isn't clearly sourced even from
// Liquipedia directly -- documented approximation (see playCard): current
// elixir at the moment of play, re-evaluated fresh each time, not a
// sticky per-match ratchet.
inline CardStats spiritEmpressGroundStats() {
    CardStats stats;
    stats.id = 165;
    stats.name = "Spirit Empress";
    stats.cost = 3.0f;
    stats.archetype = Archetype::MeleeSquad;
    stats.hp = 926;
    // Fast, matching this file's own established Fast convention (e.g. Dart
    // Goblin, Ice Spirit). Scaled here because this file assigns `speed`
    // directly and so BYPASSES CardStats::troop() -- a scale applied only in
    // the factory would silently miss both Spirit Empress forms.
    stats.speed = 0.85f * MOVEMENT_SPEED_SCALE;
    stats.attackRange = 1.2f;
    stats.damage = 249;
    stats.attackCooldown = 12;
    stats.symbol = '<';
    return stats;
}

inline CardStats spiritEmpressFlyingStats() {
    CardStats stats;
    stats.id = 165;
    stats.name = "Spirit Empress";
    stats.cost = 6.0f;
    stats.archetype = Archetype::RangedSquad;
    stats.hp = 926;
    // Medium, matching this file's own established Medium convention. Scaled
    // here for the same reason as the ground form above -- direct assignment
    // bypasses CardStats::troop().
    stats.speed = 0.5f * MOVEMENT_SPEED_SCALE;
    stats.attackRange = 5.0f;
    stats.damage = 249;
    stats.attackCooldown = 14;
    stats.symbol = '<';
    stats.isFlying = true;
    stats.targetsAir = true;
    return stats;
}
