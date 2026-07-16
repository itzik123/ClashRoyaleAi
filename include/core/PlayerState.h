#pragma once
#include <vector>
#include <deque>
#include "CardRegistry.h"

class PlayerState {
public:
    float elixir;
    std::vector<int> hand;
    std::deque<int> deckQueue;

    PlayerState() : elixir(0.0f) {}

    void initializeDeck(const std::vector<int>& deckList) {
        elixir = 5.0f;
        hand.clear();
        deckQueue.clear();

        for (size_t i = 0; i < deckList.size(); ++i) {
            if (i < 4) hand.push_back(deckList[i]);
            else deckQueue.push_back(deckList[i]);
        }
    }

    int playCard(int handIndex) {
        if (handIndex < 0 || handIndex >= static_cast<int>(hand.size())) return -1;

        int cardId = hand[handIndex];
        const CardDefinition* cardDef = CardRegistry::getInstance().getCard(cardId);

        if (cardDef && elixir >= cardDef->cost && !deckQueue.empty()) {
            elixir -= cardDef->cost;

            int nextCard = deckQueue.front();
            deckQueue.pop_front();
            deckQueue.push_back(cardId);

            hand[handIndex] = nextCard;
            return cardId;
        }
        return -1;
    }
};