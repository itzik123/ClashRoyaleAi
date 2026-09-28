#pragma once
// The Reflex Lab's view of the engine: one attacker, one defender, and the
// tower damage the defender prevents. Everything the page knows about the
// board comes from here -- legality, geometry, card facts -- so the page keeps
// no second copy of an engine constant.
//
// Compiled natively (tools/lab/lab_cli.cpp: curation, outcome tables, the dev
// backend) and to WebAssembly (lab_wasm.cpp: the page). Both must answer
// identically, which tools/lab/parity.mjs checks.
//
// The engine headers are included unchanged; nothing here edits include/.

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ArenaLayout.h"
#include "ClashEnv.h"
#include "Projectile.h"
#include "AreaSpell.h"

namespace lab {

// A rollout lasts at most 30 s, the defender may drop 0-5 s after the
// attacker in 0.5 s steps, and a troop attacker is dropped in the first rows
// past the river on the enemy side. A spell attacker that drops troops (the
// Goblin Barrel) is aimed instead at the cells within SPELL_AIM_REACH of one
// of our Princess Towers, where the real card is thrown.
constexpr int WINDOW_TICKS = 300;
constexpr int DELAY_STEPS = 11;
constexpr int DELAY_TICK_STEP = 5;
constexpr int SPAWN_ROWS = 6;
constexpr int SPELL_AIM_REACH = 2;
constexpr int NO_DEFENCE = -1;

// Any legal deck does: both bodies are placed with ClashEnv::inject, so hands
// and elixir never matter.
inline const std::vector<int>& baseDeck() {
    static const std::vector<int> deck = {15, 6, 25, 40, 24, 72, 33, 7};
    return deck;
}

struct XY {
    int x, y;
};

// getTowerHp reads -1 for a destroyed tower; a destroyed tower has 0 hp.
inline int towerHpSum(const ClashEnv& env, int team) {
    int sum = 0;
    for (int slot = 0; slot < 3; ++slot) sum += std::max(0, env.getTowerHp(team, slot));
    return sum;
}

// Can team 1 still damage anything? Only its troops and buildings can, and a
// spell of its own still in the air (a Goblin Barrel before it lands): a tower
// shot in flight at one of our troops cannot hurt our towers. Pending bodies
// (death spawns, a just-injected attacker) count as remaining.
inline bool enemyUnitsRemain(ClashEnv& env) {
    Board& board = env.debugGame().getBoard();
    if (board.pendingEntityCount() > 0) return true;
    for (const auto& e : board.getEntities()) {
        if (!e->isAlive() || e->team != 1 || e->isTower()) continue;
        if (dynamic_cast<const CombatEntity*>(e.get())) return true;
        if (dynamic_cast<const AreaSpell*>(e.get())) return true;
    }
    return false;
}

inline std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += ' ';
                else out += c;
        }
    }
    return out;
}

// Entity flags in a frame.
constexpr int F_FLYING = 1, F_BUILDING = 2, F_TOWER = 4, F_DEPLOYING = 8,
              F_PROJECTILE = 16, F_SPELL = 32;

class LabEngine {
public:
    explicit LabEngine(unsigned seed = 1)
        : base_(std::make_unique<ClashEnv>(baseDeck(), baseDeck(), 3600)) {
        base_->seed(seed);
    }

    // ---- registry and geometry -------------------------------------------

    std::string cardsJson(const std::vector<int>& ids) const {
        std::ostringstream o;
        o << "[";
        bool first = true;
        for (int id : ids) {
            const CardDefinition* d = CardRegistry::getInstance().getCard(id);
            if (!d) continue;
            if (!first) o << ",";
            first = false;
            o << "{\"id\":" << d->id << ",\"name\":\"" << jsonEscape(d->name)
              << "\",\"cost\":" << d->cost << ",\"hp\":" << d->hp
              << ",\"symbol\":\"" << jsonEscape(std::string(1, d->symbol))
              << "\",\"isFlying\":" << (d->isFlying ? "true" : "false")
              << ",\"isBuilding\":" << (d->isBuilding ? "true" : "false")
              << ",\"isSpell\":" << (d->isSpell ? "true" : "false")
              << ",\"placementRadius\":" << d->placementRadius << "}";
        }
        o << "]";
        return o.str();
    }

    std::string arenaJson() const {
        Board& b = base_->debugGame().getBoard();
        std::ostringstream o;
        o << "{\"width\":" << b.getWidth() << ",\"height\":" << b.getHeight()
          << ",\"riverStart\":" << b.getRiverStart() << ",\"riverEnd\":" << b.getRiverEnd()
          << ",\"bridgeColumns\":[";
        bool first = true;
        for (int x = 0; x < b.getWidth(); ++x) {
            if (!b.isOnBridge(static_cast<float>(x))) continue;
            if (!first) o << ",";
            first = false;
            o << x;
        }
        o << "],\"deadCells\":[";
        first = true;
        for (int y = 0; y < b.getHeight(); ++y)
            for (int x = 0; x < b.getWidth(); ++x) {
                if (!b.isBackRowDeadZone(static_cast<float>(x), static_cast<float>(y))) continue;
                if (!first) o << ",";
                first = false;
                o << "[" << x << "," << y << "]";
            }
        o << "],\"windowTicks\":" << WINDOW_TICKS << ",\"delaySteps\":" << DELAY_STEPS
          << ",\"delayTickStep\":" << DELAY_TICK_STEP << ",\"towers\":";
        std::unordered_map<int, int> maxHp;
        o << frameJson(*base_, 0, maxHp) << "}";
        return o.str();
    }

    // ---- matchup ----------------------------------------------------------

    std::string setMatchup(int attackerId, int defenderId) {
        const CardDefinition* a = CardRegistry::getInstance().getCard(attackerId);
        const CardDefinition* d = CardRegistry::getInstance().getCard(defenderId);
        if (!a || !d) throw std::invalid_argument("unknown card id");
        if (a->isBuilding) throw std::invalid_argument("attacker must not be a building");
        if (a->isSpell && !dropsTroops(attackerId))
            throw std::invalid_argument("a spell attacker must drop troops");
        if (d->isSpell) throw std::invalid_argument("defender must not be a spell");
        attacker_ = attackerId;
        defender_ = defenderId;

        Board& b = base_->debugGame().getBoard();
        const int w = b.getWidth(), h = b.getHeight();

        spawns_.clear();
        if (a->isSpell) {
            // Around each of our Princess Towers (ArenaLayout), where team 1
            // may cast it by the engine's own predicate.
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const bool nearTower =
                        std::abs(y - ArenaLayout::princessY(0)) <= SPELL_AIM_REACH &&
                        (std::abs(x - ArenaLayout::LEFT_LANE_X) <= SPELL_AIM_REACH ||
                         std::abs(x - ArenaLayout::RIGHT_LANE_X) <= SPELL_AIM_REACH);
                    if (nearTower && base_->isValidPlacementForCard(attackerId, float(x), float(y), 1))
                        spawns_.push_back({x, y});
                }
        } else {
            // The first SPAWN_ROWS rows past the river where team 1 may drop
            // the attacker, by the engine's own predicate.
            int rows = 0;
            for (int y = static_cast<int>(std::ceil(b.getRiverEnd())); y < h && rows < SPAWN_ROWS; ++y) {
                bool any = false;
                for (int x = 0; x < w; ++x) {
                    if (base_->isValidPlacementForCard(attackerId, float(x), float(y), 1)) {
                        spawns_.push_back({x, y});
                        any = true;
                    }
                }
                if (any) ++rows;
            }
        }

        cells_.clear();
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (base_->isValidPlacementForCard(defenderId, float(x), float(y), 0))
                    cells_.push_back({x, y});

        // No-defence damage per spawn; a spawn that cannot reach a tower in
        // the window would score every try against zero, so it is dropped.
        d0_.clear();
        std::vector<XY> kept;
        for (size_t i = 0; i < spawns_.size(); ++i) {
            const float dmg = simulate(spawns_[i], NO_DEFENCE, 0, nullptr);
            if (dmg > 0.0f) {
                kept.push_back(spawns_[i]);
                d0_.push_back(dmg);
            }
        }
        spawns_ = kept;
        live_.reset();
        return matchupJson();
    }

    std::string matchupJson() const {
        std::ostringstream o;
        o << "{\"attacker\":" << attacker_ << ",\"defender\":" << defender_ << ",\"spawns\":";
        writeXY(o, spawns_);
        o << ",\"d0\":[";
        for (size_t i = 0; i < d0_.size(); ++i) o << (i ? "," : "") << d0_[i];
        o << "],\"cells\":";
        writeXY(o, cells_);
        o << ",\"delaySteps\":" << DELAY_STEPS << ",\"delayTickStep\":" << DELAY_TICK_STEP
          << ",\"windowTicks\":" << WINDOW_TICKS << "}";
        return o.str();
    }

    int spawnCount() const { return static_cast<int>(spawns_.size()); }
    int cellCount() const { return static_cast<int>(cells_.size()); }
    float d0(int spawn) const { return d0_.at(spawn); }
    XY spawnAt(int i) const { return spawns_.at(i); }
    XY cellAt(int i) const { return cells_.at(i); }

    // ---- rollouts -----------------------------------------------------------

    // Tower HP lost with the defender on `cell`, dropped `delay` steps after
    // the attacker (NO_DEFENCE: none).
    float rollout(int spawn, int cell, int delay) const {
        return rolloutAtTick(spawn, cell, delay * DELAY_TICK_STEP);
    }

    float rolloutAtTick(int spawn, int cell, int dropTick) const {
        checkSpawn(spawn);
        return simulate(spawns_[spawn], cell, dropTick, nullptr);
    }

    // `sca` holds n (spawn, cell, delay) triples.
    void rolloutBatch(const int* sca, int n, float* out) const {
        for (int i = 0; i < n; ++i) out[i] = rollout(sca[3 * i], sca[3 * i + 1], sca[3 * i + 2]);
    }

    // Every tick of one rollout, for animation. `dropTick` is in ticks so a
    // visitor's challenge placement (any tick) replays exactly.
    std::string framesJson(int spawn, int cell, int dropTick) const {
        checkSpawn(spawn);
        std::vector<std::string> frames;
        const float dmg = simulate(spawns_[spawn], cell, dropTick, &frames);
        std::ostringstream o;
        o << "{\"damage\":" << dmg << ",\"frames\":[";
        for (size_t i = 0; i < frames.size(); ++i) o << (i ? "," : "") << frames[i];
        o << "]}";
        return o.str();
    }

    // A rollout with a per-tick observer (tools/lab's stall check): called
    // after each step with the env and the number of steps taken.
    float simulateWith(int spawn, int cell, int dropTick,
                       const std::function<void(ClashEnv&, int)>& onTick) const {
        checkSpawn(spawn);
        return simulate(spawns_[spawn], cell, dropTick, nullptr, &onTick);
    }

    // ---- the live challenge ------------------------------------------------

    void liveStart(int spawn) {
        checkSpawn(spawn);
        live_ = std::make_unique<ClashEnv>(base_->snapshot());
        const XY s = spawns_[spawn];
        live_->inject(attacker_, float(s.x), float(s.y), 1);
        liveStartHp_ = towerHpSum(*live_, 0);
        liveTick_ = 0;
        livePlacedTick_ = -1;
        liveDone_ = false;
        liveMaxHp_.clear();
    }

    // Advance up to `ticks` ticks; the frames for each, as a JSON array.
    std::string liveStepJson(int ticks) {
        if (!live_) throw std::logic_error("liveStart first");
        std::ostringstream o;
        o << "[";
        for (int i = 0; i < ticks && !liveDone_; ++i) {
            live_->stepSelfPlayFast(4, 0.0f, 0.0f, 4, 0.0f, 0.0f, 1);
            ++liveTick_;
            if (i) o << ",";
            o << frameJson(*live_, liveTick_, liveMaxHp_);
            if (liveTick_ >= WINDOW_TICKS || !enemyUnitsRemain(*live_)) liveDone_ = true;
        }
        o << "]";
        return o.str();
    }

    // Drop the defender now. Placing at tick T equals rolloutAtTick(.., T).
    bool livePlace(int cell) {
        if (!live_ || liveDone_ || livePlacedTick_ >= 0) return false;
        if (cell < 0 || cell >= cellCount()) return false;
        placeDefender(*live_, cell);
        livePlacedTick_ = liveTick_;
        return true;
    }

    std::string liveStateJson() const {
        std::ostringstream o;
        o << "{\"tick\":" << liveTick_ << ",\"done\":" << (liveDone_ ? "true" : "false")
          << ",\"damage\":" << liveDamage() << ",\"placedTick\":" << livePlacedTick_ << "}";
        return o.str();
    }

    bool liveDone() const { return liveDone_; }
    int liveTick() const { return liveTick_; }
    int livePlacedTick() const { return livePlacedTick_; }
    float liveDamage() const { return live_ ? float(liveStartHp_ - towerHpSum(*live_, 0)) : 0.0f; }

private:
    std::unique_ptr<ClashEnv> base_;
    int attacker_ = -1, defender_ = -1;
    std::vector<XY> spawns_, cells_;
    std::vector<float> d0_;

    std::unique_ptr<ClashEnv> live_;
    int liveStartHp_ = 0, liveTick_ = 0, livePlacedTick_ = -1;
    bool liveDone_ = false;
    std::unordered_map<int, int> liveMaxHp_;

    // Does this spell put team-1 troops on the board? Measured, not read off a
    // card list: cast it on an empty board and look for bodies.
    bool dropsTroops(int cardId) const {
        ClashEnv env = base_->snapshot();
        env.inject(cardId, ArenaLayout::LEFT_LANE_X, ArenaLayout::princessY(0), 1);
        for (int t = 0; t < 60; ++t) {
            env.stepSelfPlayFast(4, 0.0f, 0.0f, 4, 0.0f, 0.0f, 1);
            for (const auto& e : env.debugGame().getBoard().getEntities())
                if (e->isAlive() && e->team == 1 && !e->isTower() &&
                    dynamic_cast<const CombatEntity*>(e.get()))
                    return true;
        }
        return false;
    }

    void checkSpawn(int spawn) const {
        if (attacker_ < 0) throw std::logic_error("setMatchup first");
        if (spawn < 0 || spawn >= spawnCount()) throw std::out_of_range("spawn index");
    }

    void placeDefender(ClashEnv& env, int cell) const {
        const XY c = cells_.at(cell);
        env.inject(defender_, float(c.x), float(c.y), 0);
    }

    // The one rollout loop: rollout, frames and simulateWith all run it, and
    // the live stepper takes the same steps one call at a time.
    float simulate(XY s, int cell, int dropTick, std::vector<std::string>* frames,
                   const std::function<void(ClashEnv&, int)>* onTick = nullptr) const {
        ClashEnv env = base_->snapshot();
        env.inject(attacker_, float(s.x), float(s.y), 1);
        const int start = towerHpSum(env, 0);
        std::unordered_map<int, int> maxHp;
        if (frames) frames->push_back(frameJson(env, 0, maxHp));
        for (int t = 0; t < WINDOW_TICKS; ++t) {
            if (cell != NO_DEFENCE && t == dropTick) placeDefender(env, cell);
            env.stepSelfPlayFast(4, 0.0f, 0.0f, 4, 0.0f, 0.0f, 1);
            if (frames) frames->push_back(frameJson(env, t + 1, maxHp));
            if (onTick && *onTick) (*onTick)(env, t + 1);
            if (!enemyUnitsRemain(env)) break;
        }
        return float(start - towerHpSum(env, 0));
    }

    static void writeXY(std::ostringstream& o, const std::vector<XY>& v) {
        o << "[";
        for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << "[" << v[i].x << "," << v[i].y << "]";
        o << "]";
    }

    // {t, towers:[[team,slot,hp,max]...], e:[[id,cardId,team,x,y,hp,maxHp,flags,
    // radius,"symbol","name"]...]}. A troop's max HP is not stored on it, so it
    // is the HP it was first seen with (every body appears at full health).
    static std::string frameJson(ClashEnv& env, int t, std::unordered_map<int, int>& maxHp) {
        std::ostringstream o;
        o.setf(std::ios::fixed);
        o.precision(3);
        o << "{\"t\":" << t << ",\"towers\":[";
        bool first = true;
        for (int team = 0; team < 2; ++team)
            for (int slot = 0; slot < 3; ++slot) {
                if (!first) o << ",";
                first = false;
                o << "[" << team << "," << slot << "," << std::max(0, env.getTowerHp(team, slot))
                  << "," << env.getTowerMaxHp(team, slot) << "]";
            }
        o << "],\"e\":[";
        first = true;
        for (const auto& e : env.debugGame().getBoard().getEntities()) {
            if (!e->isAlive()) continue;
            int flags = 0, mhp = 0;
            if (e->isFlying) flags |= F_FLYING;
            if (e->isBuilding()) flags |= F_BUILDING;
            if (e->isTower()) flags |= F_TOWER;
            if (auto* b = dynamic_cast<const Building*>(e.get())) mhp = b->getMaxHp();
            if (auto* c = dynamic_cast<const CombatEntity*>(e.get()))
                if (c->deployTicksRemaining > 0) flags |= F_DEPLOYING;
            if (dynamic_cast<const Projectile*>(e.get())) flags |= F_PROJECTILE;
            if (dynamic_cast<const AreaSpell*>(e.get())) flags |= F_SPELL;
            if (mhp <= 0) {
                auto it = maxHp.find(e->id);
                if (it == maxHp.end()) it = maxHp.emplace(e->id, e->hp).first;
                mhp = std::max(it->second, e->hp);
            }
            if (!first) o << ",";
            first = false;
            o << "[" << e->id << "," << e->cardId << "," << e->team << "," << e->position.x << ","
              << e->position.y << "," << e->hp << "," << mhp << "," << flags << ","
              << e->getCollisionRadius() << ",\"" << jsonEscape(std::string(1, e->symbol))
              << "\",\"" << jsonEscape(e->name) << "\"]";
        }
        o << "]}";
        return o.str();
    }
};

}  // namespace lab
