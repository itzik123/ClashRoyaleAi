#pragma once
#include "Entity.h"

class CombatEntity;

// Composable behavior triggered when a Champion's activated ability fires
// (e.g. Mighty Miner's "Explosive Escape"), decoupled from *what* the
// ability actually does -- same shape and reasoning as IDeathEffect/
// IPeriodicEffect, just triggered by an explicit player action
// (GameManager::activateChampionAbility) instead of a death or a repeating
// timer. Concrete effects live in core/ (e.g. MightyMinerEscapeEffect) --
// same "concrete effect implementations live in core/" convention as
// SpawnOnDeath/AreaDamageOnDeath/PeriodicSpawnEffect, regardless of whether
// a specific one individually needs CardFactories.
//
// Takes the acting entity itself, not just position/team like IDeathEffect,
// because an activated ability needs to mutate ITS OWN state (Mighty
// Miner's teleport moves its own position) -- IDeathEffect/IPeriodicEffect
// never need to write back to the entity that fired them.
class IAbilityEffect {
public:
    virtual ~IAbilityEffect() = default;
    virtual void apply(Board& board, CombatEntity& self) const = 0;
};
