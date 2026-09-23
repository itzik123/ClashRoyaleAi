#pragma once
#include "Entity.h"

class CombatEntity;

// Behaviour fired by a Champion's activated ability
// (GameManager::activateChampionAbility). Concrete effects live in core/. Takes
// the acting entity, since an ability mutates its own state (e.g. Mighty
// Miner's teleport).
class IAbilityEffect {
public:
    virtual ~IAbilityEffect() = default;
    virtual void apply(Board& board, CombatEntity& self) const = 0;
};
