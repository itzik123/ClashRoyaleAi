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
    std::shared_ptr<ElixirValueKilledCollector> elixirValueKilled;
    std::shared_ptr<ElixirStatsCollector> elixir;
    std::shared_ptr<CardPlayStatsCollector> cardPlay;
    std::shared_ptr<ChampionAbilityStatsCollector> championAbility;
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
        elixirValueKilled = std::make_shared<ElixirValueKilledCollector>();
        elixir = std::make_shared<ElixirStatsCollector>();
        cardPlay = std::make_shared<CardPlayStatsCollector>();
        championAbility = std::make_shared<ChampionAbilityStatsCollector>();
        outcome = std::make_shared<MatchOutcomeCollector>();

        subscribeAll(board);
    }

    // Independent copy of every collector, subscribed to `board` -- the stats
    // half of GameManager::snapshot(), paired with Board::deepCopy().
    //
    // Note what this is NOT: calling attach() on the copied board would have
    // been one line, and WRONG. attach() builds FRESH ZEROED collectors, so a
    // snapshot would report a match in which nobody had dealt any damage yet.
    // train.py's tower-damage term is potential-based -- Phi(s) is a function
    // of CUMULATIVE damage -- so a search scoring candidates by Phi on a
    // zeroed snapshot would read every rollout as an enormous instant loss of
    // accumulated progress, identically for every candidate. It would look
    // like a working search that simply never preferred anything.
    //
    // Copying the shared_ptrs instead (the implicit copy) is the opposite
    // failure and the one Board::deepCopy already documents: the collectors
    // are stateful, so a rollout's hits would land in the LIVE match's totals.
    // Only a genuine per-collector deep copy is correct, and each is plain
    // data (ints and unordered_maps), so the implicit copy constructor does it.
    MatchStatistics snapshotFor(Board& board) const {
        MatchStatistics copy;
        copy.damage = damage ? std::make_shared<DamageStatsCollector>(*damage) : nullptr;
        copy.damageByTargetType = damageByTargetType
            ? std::make_shared<DamageByTargetTypeCollector>(*damageByTargetType) : nullptr;
        copy.kill = kill ? std::make_shared<KillStatsCollector>(*kill) : nullptr;
        copy.elixirValueKilled = elixirValueKilled
            ? std::make_shared<ElixirValueKilledCollector>(*elixirValueKilled) : nullptr;
        copy.elixir = elixir ? std::make_shared<ElixirStatsCollector>(*elixir) : nullptr;
        copy.cardPlay = cardPlay ? std::make_shared<CardPlayStatsCollector>(*cardPlay) : nullptr;
        copy.championAbility = championAbility
            ? std::make_shared<ChampionAbilityStatsCollector>(*championAbility) : nullptr;
        copy.outcome = outcome ? std::make_shared<MatchOutcomeCollector>(*outcome) : nullptr;
        copy.subscribeAll(board);
        return copy;
    }

    int totalDamageDealt(int team) const { return damage ? damage->total(team) : 0; }
    int damageDealtByCard(int cardId, int team) const { return damage ? damage->byCard(cardId, team) : 0; }

    // Cross-team damage dealt BY `team`, split by whether the target was a
    // troop or a building -- see DamageByTargetTypeCollector. This is the
    // breakdown python_ai/train.py's reward shaping actually consumes.
    int troopDamageDealt(int team) const { return damageByTargetType ? damageByTargetType->troopDamageDealt(team) : 0; }
    int buildingDamageDealt(int team) const { return damageByTargetType ? damageByTargetType->buildingDamageDealt(team) : 0; }
    // Towers only -- buildingDamageDealt() minus this is damage to DEPLOYED
    // buildings (Cannon, Tesla, ...). The shaping potential must use this one:
    // see DamageByTargetTypeCollector's comment for why lumping them together
    // taught the agent to hide its Cannon behind its own King.
    int towerDamageDealt(int team) const { return damageByTargetType ? damageByTargetType->towerDamageDealt(team) : 0; }

    int kills(int team) const { return kill ? kill->kills(team) : 0; }
    // Elixir value of everything `cardId` has killed for `team` -- see
    // ElixirValueKilledCollector for why this is priced in cost, not HP.
    float elixirValueKilledBy(int cardId, int team) const {
        return elixirValueKilled ? elixirValueKilled->byCard(cardId, team) : 0.0f;
    }
    int killsByCard(int cardId, int team) const { return kill ? kill->killsByCard(cardId, team) : 0; }

    float elixirSpent(int team) const { return elixir ? elixir->total(team) : 0.0f; }
    float elixirSpentByCard(int cardId, int team) const { return elixir ? elixir->byCard(cardId, team) : 0.0f; }

    const std::vector<CardPlayedEvent>& cardsPlayed(int team) const {
        static const std::vector<CardPlayedEvent> empty;
        return cardPlay ? cardPlay->timeline(team) : empty;
    }

    float championAbilityElixirSpent(int team) const { return championAbility ? championAbility->elixirSpent(team) : 0.0f; }
    int championAbilityActivations(int team) const { return championAbility ? championAbility->activations(team) : 0; }

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
        out << "\"championAbilityElixirSpent\":{\"team0\":" << floatStr(championAbilityElixirSpent(0))
            << ",\"team1\":" << floatStr(championAbilityElixirSpent(1)) << "},";
        out << "\"championAbilityActivations\":{\"team0\":" << championAbilityActivations(0)
            << ",\"team1\":" << championAbilityActivations(1) << "},";

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
    // The subscribe list in ONE place, shared by attach() and snapshotFor().
    // Written as a helper rather than repeated, because two copies of "every
    // collector" is a list that drifts: a ninth collector added to attach()
    // and forgotten here would go on recording the live match while silently
    // recording nothing across a snapshot -- and the query API would still
    // answer, with stale numbers, rather than fail.
    //
    // The null guards only matter for snapshotFor() on a never-attached
    // MatchStatistics (a GameManager copied before its first reset()), which
    // stays legitimately empty rather than becoming half-subscribed.
    void subscribeAll(Board& board) {
        if (damage) board.statsEvents.subscribe(damage);
        if (damageByTargetType) board.statsEvents.subscribe(damageByTargetType);
        if (kill) board.statsEvents.subscribe(kill);
        if (elixirValueKilled) board.statsEvents.subscribe(elixirValueKilled);
        if (elixir) board.statsEvents.subscribe(elixir);
        if (cardPlay) board.statsEvents.subscribe(cardPlay);
        if (championAbility) board.statsEvents.subscribe(championAbility);
        if (outcome) board.statsEvents.subscribe(outcome);
    }

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
