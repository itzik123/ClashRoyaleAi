#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Wizard's "Fiery Flight": takes flight, and for the same duration each
// landed attack pulses a damaging, pulling tornado on the target (handled in
// CombatEntity::update).
class HeroWizardFieryFlightEffect : public IAbilityEffect {
    int durationTicks;
    float pulseRadius;
    int pulseDamage;
    float pulsePullDistance;

public:
    HeroWizardFieryFlightEffect(int durationTicks, float pulseRadius, int pulseDamage, float pulsePullDistance)
        : durationTicks(durationTicks), pulseRadius(pulseRadius), pulseDamage(pulseDamage),
          pulsePullDistance(pulsePullDistance) {}

    void apply(Board&, CombatEntity& self) const override {
        self.isFlying = true;
        self.temporaryFlightTicksRemaining = durationTicks;
        self.flightPulseTicksRemaining = durationTicks;
        self.flightPulseRadius = pulseRadius;
        self.flightPulseDamage = pulseDamage;
        self.flightPulsePullDistance = pulsePullDistance;
    }
};
