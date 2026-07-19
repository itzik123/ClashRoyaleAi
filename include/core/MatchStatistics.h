#pragma once
#include "Board.h"
#include "stats/StatsCollectors.h"
#include <memory>
#include <sstream>
#include <string>

// The "convenient to process" query facade: owns the five built-in
// collectors, subscribes them to a Board's StatsEventBus once, and exposes
// a clean read-only API plus a JSON export instead of making every consumer
// hand-roll tick diffing the way python_ai/train.py currently does.
//
// damageDealtByCard/killsByCard/elixirSpentByCard are keyed by the
// ATTACKER's/PLAYER's card, not the victim's -- "how much damage did Knight
// deal", not "how much damage did Knight take". A victim-keyed view
// (deathsByCard, damageTakenByCard) is a plausible later addition once
// there's a concrete need for it; not part of this pass.
class MatchStatistics {
    std::shared_ptr<DamageStatsCollector> damage;
    std::shared_ptr<DamageByTargetTypeCollector> damageByTargetType;
    std::shared_ptr<KillStatsCollector> kill;
    std::shared_ptr<ElixirStatsCollector> elixir;
    std::shared_ptr<CardPlayStatsCollector> cardPlay;
    std::shared_ptr<MatchOutcomeCollector> outcome;

    static std::string floatStr(float v) {
        std::ostringstream ss;
        ss << std::fixed;
        ss.precision(2);
        ss << v;
        return ss.str();
    }

public:
    // Unattached: no collectors, every query safely reports zero/empty.
    // Safe to hold before the first attach() (e.g. as a GameManager member
    // constructed before reset() runs) and between attach() calls.
    MatchStatistics() = default;

    // (Re)constructs fresh collectors and subscribes them to board's event
    // bus. Called once per match -- GameManager::reset() calls this as the
    // first line right after `board = Board();`, since that reassignment
    // destroys the old Board's (and its statsEvents subscriber list's)
    // lifetime entirely.
    void attach(Board& board) {
        damage = std::make_shared<DamageStatsCollector>();
        damageByTargetType = std::make_shared<DamageByTargetTypeCollector>();
        kill = std::make_shared<KillStatsCollector>();
        elixir = std::make_shared<ElixirStatsCollector>();
        cardPlay = std::make_shared<CardPlayStatsCollector>();
        outcome = std::make_shared<MatchOutcomeCollector>();

        board.statsEvents.subscribe(damage);
        board.statsEvents.subscribe(damageByTargetType);
        board.statsEvents.subscribe(kill);
        board.statsEvents.subscribe(elixir);
        board.statsEvents.subscribe(cardPlay);
        board.statsEvents.subscribe(outcome);
    }

    int totalDamageDealt(int team) const { return damage ? damage->total(team) : 0; }
    int damageDealtByCard(int cardId, int team) const { return damage ? damage->byCard(cardId, team) : 0; }

    // Cross-team damage dealt BY `team`, split by whether the target was a
    // troop or a building -- see DamageByTargetTypeCollector. This is the
    // breakdown python_ai/train.py's reward shaping actually consumes.
    int troopDamageDealt(int team) const { return damageByTargetType ? damageByTargetType->troopDamageDealt(team) : 0; }
    int buildingDamageDealt(int team) const { return damageByTargetType ? damageByTargetType->buildingDamageDealt(team) : 0; }

    int kills(int team) const { return kill ? kill->kills(team) : 0; }
    int killsByCard(int cardId, int team) const { return kill ? kill->killsByCard(cardId, team) : 0; }

    float elixirSpent(int team) const { return elixir ? elixir->total(team) : 0.0f; }
    float elixirSpentByCard(int cardId, int team) const { return elixir ? elixir->byCard(cardId, team) : 0.0f; }

    const std::vector<CardPlayedEvent>& cardsPlayed(int team) const {
        static const std::vector<CardPlayedEvent> empty;
        return cardPlay ? cardPlay->timeline(team) : empty;
    }

    int loserTeam() const { return outcome ? outcome->loserTeam() : -1; }
    int matchDurationTicks() const { return outcome ? outcome->matchDurationTicks() : 0; }

    // Hand-rolled, matching GameLogger::save()'s existing inline
    // std::ofstream/std::ostringstream style rather than extracting a
    // shared JSON-writer utility -- the shape here (ints, floats, a nested
    // card-play array) isn't known precisely enough yet to generalize well;
    // revisit extraction once GameLogger and this end up needing genuinely
    // the same shapes.
    std::string toJson() const {
        std::ostringstream out;
        out << "{";
        out << "\"loserTeam\":" << loserTeam() << ",";
        out << "\"matchDurationTicks\":" << matchDurationTicks() << ",";

        out << "\"totalDamageDealt\":{\"team0\":" << totalDamageDealt(0)
            << ",\"team1\":" << totalDamageDealt(1) << "},";
        out << "\"troopDamageDealt\":{\"team0\":" << troopDamageDealt(0)
            << ",\"team1\":" << troopDamageDealt(1) << "},";
        out << "\"buildingDamageDealt\":{\"team0\":" << buildingDamageDealt(0)
            << ",\"team1\":" << buildingDamageDealt(1) << "},";
        out << "\"kills\":{\"team0\":" << kills(0) << ",\"team1\":" << kills(1) << "},";
        out << "\"elixirSpent\":{\"team0\":" << floatStr(elixirSpent(0))
            << ",\"team1\":" << floatStr(elixirSpent(1)) << "},";

        out << "\"damageDealtByCard\":{";
        writeTeamCardMaps(out, damage ? &damage->byCardMap(0) : nullptr, damage ? &damage->byCardMap(1) : nullptr);
        out << "},";

        out << "\"killsByCard\":{";
        writeTeamCardMaps(out, kill ? &kill->killsByCardMap(0) : nullptr, kill ? &kill->killsByCardMap(1) : nullptr);
        out << "},";

        out << "\"cardsPlayed\":{";
        out << "\"team0\":[";
        writeCardPlayTimeline(out, cardsPlayed(0));
        out << "],\"team1\":[";
        writeCardPlayTimeline(out, cardsPlayed(1));
        out << "]}";

        out << "}";
        return out.str();
    }

private:
    template <typename MapT>
    static void writeTeamCardMaps(std::ostringstream& out, const MapT* team0, const MapT* team1) {
        out << "\"team0\":{";
        writeCardMap(out, team0);
        out << "},\"team1\":{";
        writeCardMap(out, team1);
        out << "}";
    }

    template <typename MapT>
    static void writeCardMap(std::ostringstream& out, const MapT* m) {
        if (!m) return;
        bool first = true;
        for (const auto& [cardId, value] : *m) {
            if (!first) out << ",";
            out << "\"" << cardId << "\":" << value;
            first = false;
        }
    }

    static void writeCardPlayTimeline(std::ostringstream& out, const std::vector<CardPlayedEvent>& timeline) {
        for (size_t i = 0; i < timeline.size(); ++i) {
            const auto& p = timeline[i];
            if (i > 0) out << ",";
            out << "{\"cardId\":" << p.cardId
                << ",\"cost\":" << floatStr(p.cost)
                << ",\"x\":" << floatStr(p.x)
                << ",\"y\":" << floatStr(p.y)
                << ",\"tick\":" << p.tick << "}";
        }
    }
};
