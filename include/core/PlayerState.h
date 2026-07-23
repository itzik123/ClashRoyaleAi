#pragma once
#include <vector>
#include <deque>
#include <unordered_map>
#include "CardRegistry.h"

class PlayerState {
public:
    // Per-Evolution-slot progress: how many more (un-evolved) plays until
    // this slot unlocks, and how many evolved plays it has left once
    // unlocked. NOT a one-time-per-match thing -- confirmed via research
    // (Wall Breakers Evolution: "2 Cycles" to unlock, "1 in every 3
    // deploys will be evolved", repeating for the whole match, not a fixed
    // number of total charges) -- once evolvedUsesRemaining hits 0, the
    // slot resets cyclesUntilEvolved back to the CardDefinition's own
    // evolutionCycleThreshold and starts counting down again, indefinitely.
    // Keyed by the evolution's own CardDefinition id (see
    // CardRegistry::addEvolution) -- a plain deck id with no evolution
    // equipped never gets an entry here at all.
    struct EvolutionSlotState {
        int cyclesUntilEvolved = 0;
        int evolvedUsesRemaining = 0;
    };

    // What playCard actually did -- cardId (-1 on failure, matching the old
    // plain-int return) plus whether this particular play should use the
    // evolved form (Evolutions) instead of the base one. GameManager reads
    // useEvolvedForm to decide which of CardDefinition's two spawn closures
    // to call; it's always false for a non-evolution card.
    struct PlayCardResult {
        int cardId = -1;
        bool useEvolvedForm = false;
    };

    // Per-Champion-slot ability tracking (deck slot 1 "Heroic" and/or slot
    // 2 "Wild Card" -- see CardRegistry::validateDeckSlots). trackedEntityId
    // is unconditionally overwritten every time GameManager::playCard
    // deploys a fresh instance of that slot's Champion, regardless of
    // whether an older instance is somehow still alive -- this is what
    // gives "the ability belongs to whoever was created last" for free, and
    // (since a Clone-spell duplicate is never created via playCard) is also
    // what makes a clone permanently unable to activate the ability.
    // persistedCooldownRemaining is synced from the tracked entity's own
    // abilityCooldownRemaining every tick it's alive (see
    // GameManager::syncChampionCooldowns) and left untouched once that
    // entity is gone -- so a later redeploy of the same slot's Champion
    // resumes the cooldown instead of starting fresh at 0.
    struct ChampionSlotState {
        int trackedEntityId = -1;
        int persistedCooldownRemaining = 0;
    };

    float elixir;
    std::vector<int> hand;
    std::deque<int> deckQueue;
    std::unordered_map<int, EvolutionSlotState> evolutionState;
    // Keyed by deck slot index (only ever 1 or 2, see ChampionSlotState's
    // own comment) -- absent entirely if that slot isn't a Champion at all,
    // same "no entry unless relevant" idiom as evolutionState above.
    std::unordered_map<int, ChampionSlotState> championSlots;
    // Last cardId this team successfully played via GameManager::playCard,
    // excluding Mirror itself (a second Mirror replays whatever was played
    // before the first Mirror, not the first Mirror) -- see Mirror's
    // handling in GameManager::playCard. -1 means "nothing played yet".
    int lastPlayedCardId = -1;

    PlayerState() : elixir(0.0f) {}

    void initializeDeck(const std::vector<int>& deckList) {
        elixir = 5.0f;
        hand.clear();
        deckQueue.clear();
        evolutionState.clear();
        championSlots.clear();
        lastPlayedCardId = -1;

        for (size_t i = 0; i < deckList.size(); ++i) {
            if (i < 4) hand.push_back(deckList[i]);
            else deckQueue.push_back(deckList[i]);

            const CardDefinition* def = CardRegistry::getInstance().getCard(deckList[i]);
            if (def && def->isEvolution) {
                evolutionState[deckList[i]] = EvolutionSlotState{
                    def->evolutionCycleThreshold, def->evolvedUsesGranted };
            }
            if (def && def->isChampion && (i == 1 || i == 2)) {
                championSlots[static_cast<int>(i)] = ChampionSlotState{};
            }
        }
    }

    // costOverride >= 0 charges that amount instead of the card's own
    // registered cost -- Mirror (mirrored card's cost + 1) and Spirit
    // Empress (dynamic 3 or 6 depending on current elixir) both need this;
    // every other card passes nothing and gets its own CardDefinition::cost
    // as before.
    PlayCardResult playCard(int handIndex, float costOverride = -1.0f) {
        if (handIndex < 0 || handIndex >= static_cast<int>(hand.size())) return {};

        int cardId = hand[handIndex];
        const CardDefinition* cardDef = CardRegistry::getInstance().getCard(cardId);
        if (!cardDef) return {};

        float cost = (costOverride >= 0.0f) ? costOverride : cardDef->cost;
        if (elixir < cost || deckQueue.empty()) return {};

        elixir -= cost;

        int nextCard = deckQueue.front();
        deckQueue.pop_front();
        deckQueue.push_back(cardId);
        hand[handIndex] = nextCard;

        bool useEvolvedForm = false;
        if (cardDef->isEvolution) {
            auto it = evolutionState.find(cardId);
            // Always present (seeded by initializeDeck), but guard anyway
            // rather than assume -- a missing entry just behaves as
            // permanently un-evolved instead of crashing.
            if (it != evolutionState.end()) {
                EvolutionSlotState& slot = it->second;
                if (slot.cyclesUntilEvolved <= 0) {
                    useEvolvedForm = true;
                    slot.evolvedUsesRemaining--;
                    if (slot.evolvedUsesRemaining <= 0) {
                        // Repeats for the rest of the match, not a one-time
                        // charge -- start the cycle countdown over.
                        slot.cyclesUntilEvolved = cardDef->evolutionCycleThreshold;
                        slot.evolvedUsesRemaining = cardDef->evolvedUsesGranted;
                    }
                } else {
                    slot.cyclesUntilEvolved--;
                }
            }
        }

        return { cardId, useEvolvedForm };
    }
};
