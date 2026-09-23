#pragma once
#include <vector>
#include <deque>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <numeric>
#include "CardRegistry.h"

class PlayerState {
public:
    // Per-Evolution progress: un-evolved plays left until the slot unlocks, and
    // evolved plays left once it has. Repeats for the whole match: when
    // evolvedUsesRemaining reaches 0 the countdown restarts from
    // evolutionCycleThreshold. Keyed by the evolution's own card id; a deck
    // without evolutions has no entries.
    struct EvolutionSlotState {
        int cyclesUntilEvolved = 0;
        int evolvedUsesRemaining = 0;
    };

    // cardId is -1 on failure; useEvolvedForm selects the evolved spawn.
    struct PlayCardResult {
        int cardId = -1;
        bool useEvolvedForm = false;
    };

    // Per-Champion-slot ability tracking, for deck slots 1 and 2
    // (CardRegistry::validateDeckSlots). trackedEntityId is overwritten by
    // every fresh deploy of the slot's Champion, so the ability belongs to the
    // newest instance; a Mirror copy goes through playCard and qualifies, a
    // Clone-spell copy never does.
    //
    // persistedCooldownRemaining follows the tracked entity's cooldown while it
    // lives (GameManager::syncChampionCooldowns) and persists after it dies, so
    // a redeploy resumes the cooldown.
    struct ChampionSlotState {
        int trackedEntityId = -1;
        int persistedCooldownRemaining = 0;
        // Post-death squad reactivation (Hero Goblins' "Banner Brigade"): the
        // tick and position at which this slot's squad last died out, or -1.
        // Set in GameManager::syncChampionCooldowns, consumed via
        // CardDefinition::abilityUsableAfterDeathTicks /
        // postDeathAbilityEffect.
        int lastSquadWipeTick = -1;
        Vector2D lastSquadWipePosition{ 0.0f, 0.0f };
    };

    float elixir;
    std::vector<int> hand;
    // Parallel to hand. > 0 means the card just cycled in and is not yet
    // playable (20 ticks); 0 for the opening hand.
    std::vector<int> handCooldownTicks;
    std::deque<int> deckQueue;
    std::unordered_map<int, EvolutionSlotState> evolutionState;
    // Keyed by deck slot index (1 or 2); absent if that slot holds no Champion.
    std::unordered_map<int, ChampionSlotState> championSlots;
    // The last card this team played, excluding Mirror (a second Mirror replays
    // what preceded the first). -1 before any play.
    int lastPlayedCardId = -1;

    PlayerState() : elixir(0.0f) {}

    // Evolution and Champion bookkeeping keys off the original deck (evolution
    // by card id, Champion by deck index 1 or 2), never off hand position, so
    // the hand shuffle cannot affect it.
    void seedSlotState(const std::vector<int>& deckList) {
        for (size_t i = 0; i < deckList.size(); ++i) {
            const CardDefinition* def = CardRegistry::getInstance().getCard(deckList[i]);
            if (def && def->isEvolution) {
                evolutionState[deckList[i]] = EvolutionSlotState{
                    def->evolutionCycleThreshold, def->evolvedUsesGranted };
            }
            // A Hero uses the same per-slot tracking as a Champion.
            if (def && (def->isChampion || def->isHero) && (i == 1 || i == 2)) {
                championSlots[static_cast<int>(i)] = ChampionSlotState{};
            }
        }
    }

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
        }
        handCooldownTicks.assign(hand.size(), 0);
        seedSlotState(deckList);
    }

    // Random opening hand: shuffles the eight deck indices, the first four
    // forming the hand. The deterministic overload above stays the default
    // because many tests assert its hand; GameManager::reset uses this one.
    //
    // Elixir Collector (99) and Mirror (164) may not start in hand; one dealt
    // there is swapped with an eligible card from the queue. A legal deck holds
    // each at most once, so a swap always exists.
    void initializeDeck(const std::vector<int>& deckList, std::mt19937& rng) {
        elixir = 5.0f;
        hand.clear();
        deckQueue.clear();
        evolutionState.clear();
        championSlots.clear();
        lastPlayedCardId = -1;

        std::vector<size_t> order(deckList.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::shuffle(order.begin(), order.end(), rng);

        auto excludedFromOpeningHand = [](int cardId) { return cardId == 99 || cardId == 164; };
        for (size_t i = 0; i < 4 && i < order.size(); ++i) {
            if (!excludedFromOpeningHand(deckList[order[i]])) continue;
            for (size_t j = 4; j < order.size(); ++j) {
                if (!excludedFromOpeningHand(deckList[order[j]])) {
                    std::swap(order[i], order[j]);
                    break;
                }
            }
        }

        for (size_t pos = 0; pos < order.size(); ++pos) {
            int cardId = deckList[order[pos]];
            if (pos < 4) hand.push_back(cardId);
            else deckQueue.push_back(cardId);
        }
        handCooldownTicks.assign(hand.size(), 0);
        seedSlotState(deckList);
    }

    // Once per tick, from GameManager::step: counts down post-cycle delays.
    void tick() {
        for (int& cooldown : handCooldownTicks) {
            if (cooldown > 0) --cooldown;
        }
    }

    // costOverride >= 0 replaces the registered cost (Mirror: the mirrored
    // card's cost + 1; Spirit Empress: 3 or 6).
    PlayCardResult playCard(int handIndex, float costOverride = -1.0f) {
        if (handIndex < 0 || handIndex >= static_cast<int>(hand.size())) return {};
        // Still on its post-cycle delay: fails like insufficient elixir.
        if (handCooldownTicks[handIndex] > 0) return {};

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
        handCooldownTicks[handIndex] = 20;

        bool useEvolvedForm = false;
        if (cardDef->isEvolution) {
            auto it = evolutionState.find(cardId);
            // Always seeded by initializeDeck; a missing entry behaves as never
            // evolved.
            if (it != evolutionState.end()) {
                EvolutionSlotState& slot = it->second;
                if (slot.cyclesUntilEvolved <= 0) {
                    useEvolvedForm = true;
                    slot.evolvedUsesRemaining--;
                    if (slot.evolvedUsesRemaining <= 0) {
                        // Restart the countdown: evolution repeats for the
                        // whole match.
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
