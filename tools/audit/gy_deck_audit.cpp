// Behavioural audit of one real deck, measured off the engine rather than read
// off the registry:
//   Evo Furnace (138)  Hero Barbarian Barrel (174)  Evo Bats (126)  Poison (32)
//   Giant Skeleton (39)  Graveyard (110)  Berserker (51)  Goblin Hut (95)
//
// Every section prints what the engine DID (ticks, damage events, positions),
// next to the real game's value as of 2026-09-25, so the comparison is made on
// behaviour. 10 ticks = 1 s. Findings and sources: perception/UPSTREAM_REQUESTS.md
// item 32.
//
// Two harness rules, both learned the hard way while writing it:
//  * Towers are silenced with Tower::sleep(), not applyFreeze(): before item 32m
//    a stunned unit whose cooldown was already 0 still acquired and fired once
//    (section 13 checks that it no longer does), and sleep() holds either way.
//  * Look children up with stepUntilCombat(): Hero Barbarian Barrel's rolling
//    spell and its Barbarian share one name, and the spell is found first.
//
// Build and run (from the repo root, PowerShell):
//   powershell -File tools/audit/build.ps1 gy_deck_audit
//   tools/audit/bin/gy_deck_audit.exe

#include "GameManager.h"
#include "CardRegistry.h"
#include "CombatEntity.h"
#include "Troop.h"
#include "Building.h"
#include "Tower.h"
#include "MeleeTroop.h"
#include "AreaSpell.h"
#include "StatsEventBus.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

const std::vector<int> USER_DECK = { 138, 174, 126, 32, 39, 110, 51, 95 };

struct Recorder : IStatsObserver {
    std::vector<DamageDealtEvent> dmg;
    std::vector<EntitySpawnedEvent> spawned;
    std::vector<EntityDiedEvent> died;
    std::vector<ChampionAbilityActivatedEvent> abilities;
    void onDamageDealt(const DamageDealtEvent& e) override { dmg.push_back(e); }
    void onEntitySpawned(const EntitySpawnedEvent& e) override { spawned.push_back(e); }
    void onEntityDied(const EntityDiedEvent& e) override { died.push_back(e); }
    void onChampionAbilityActivated(const ChampionAbilityActivatedEvent& e) override { abilities.push_back(e); }
};

struct Sim {
    GameManager g;
    std::shared_ptr<Recorder> rec;

    explicit Sim(bool freezeAllTowers = true) : g(USER_DECK, USER_DECK) {
        g.seed(7);
        g.reset();
        rec = std::make_shared<Recorder>();
        g.getBoard().statsEvents.subscribe(rec);
        if (freezeAllTowers) freezeTowers(-1);
    }
    Board& b() { return g.getBoard(); }
    int tick() const { return g.getCurrentTick(); }
    void step(int n = 1) { for (int i = 0; i < n; ++i) g.step(); }

    // Tower::sleep(): a sleeping tower acquires nothing (Tower::findTarget) until
    // it takes damage. NOT applyFreeze -- a stunned unit whose cooldown is
    // already 0 still acquires and fires once (see section 13).
    void freezeTowers(int team) {
        for (auto& e : b().getEntities())
            if (e->isTower() && (team < 0 || e->team == team))
                if (auto t = std::dynamic_pointer_cast<Tower>(e)) t->sleep();
    }
    std::vector<std::shared_ptr<Entity>> spawn(int cardId, float x, float y, int team, bool evolved = false) {
        std::set<int> before;
        for (auto& e : b().getEntities()) before.insert(e->id);
        const CardDefinition* c = CardRegistry::getInstance().getCard(cardId);
        if (evolved) c->spawnEvolvedEntity(x, y, team, b()); else c->spawnEntity(x, y, team, b());
        b().commitPendingEntities(tick());
        std::vector<std::shared_ptr<Entity>> out;
        for (auto& e : b().getEntities()) if (!before.count(e->id)) out.push_back(e);
        return out;
    }
    // Stationary, harmless punching bag: speed 0, damage 0, no deploy time.
    std::shared_ptr<MeleeTroop> dummy(float x, float y, int team, int hp, bool flying = false) {
        auto d = std::make_shared<MeleeTroop>(b().allocateId(), x, y, hp, team, 0.0f, 0.5f, 0, 1000, 'D');
        d->name = flying ? "AirDummy" : "Dummy";
        d->cardId = -99;
        d->isFlying = flying;
        d->deployTicksRemaining = 0;
        b().addEntity(d);
        b().commitPendingEntities(tick());
        return d;
    }
    std::shared_ptr<Entity> byId(int id) {
        for (auto& e : b().getEntities()) if (e->id == id) return e;
        return nullptr;
    }
    std::vector<std::shared_ptr<Entity>> named(const std::string& name, int team) {
        std::vector<std::shared_ptr<Entity>> out;
        for (auto& e : b().getEntities())
            if (e->isAlive() && e->team == team && e->name == name) out.push_back(e);
        return out;
    }
    std::shared_ptr<Entity> firstNamed(const std::string& name, int team) {
        auto v = named(name, team);
        return v.empty() ? nullptr : v.front();
    }
    // Steps until an entity with this name exists on `team`; nullptr on timeout.
    std::shared_ptr<Entity> stepUntil(const std::string& name, int team, int maxTicks) {
        for (int i = 0; i < maxTicks; ++i) {
            if (auto e = firstNamed(name, team)) return e;
            step();
        }
        return firstNamed(name, team);
    }
    std::shared_ptr<CombatEntity> stepUntilCombat(const std::string& name, int team, int maxTicks) {
        for (int i = 0; i <= maxTicks; ++i) {
            for (auto& e : b().getEntities())
                if (e->isAlive() && e->team == team && e->name == name)
                    if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) return c;
            step();
        }
        return nullptr;
    }
    std::shared_ptr<Entity> princessTower(int team, float x) {
        for (auto& e : b().getEntities())
            if (e->isTower() && e->team == team && e->symbol != 'R' && std::fabs(e->position.x - x) < 0.1f) return e;
        return nullptr;
    }
    std::vector<DamageDealtEvent> hitsBy(int attackerId) const {
        std::vector<DamageDealtEvent> out;
        for (auto& d : rec->dmg) if (d.attackerId == attackerId) out.push_back(d);
        return out;
    }
};

void header(const char* title) {
    std::printf("\n==================================================================\n%s\n"
                "==================================================================\n", title);
}

std::string intervals(const std::vector<int>& ticks) {
    std::string s;
    for (size_t i = 1; i < ticks.size(); ++i) {
        s += std::to_string(ticks[i] - ticks[i - 1]);
        if (i + 1 < ticks.size()) s += ",";
    }
    return s.empty() ? "-" : s;
}

void identity(const char* label, const std::shared_ptr<Entity>& e) {
    if (!e) { std::printf("%-28s (not produced)\n", label); return; }
    auto c = std::dynamic_pointer_cast<CombatEntity>(e);
    auto t = std::dynamic_pointer_cast<Troop>(e);
    std::printf("%-28s hp=%5d", label, e->hp);
    if (c) {
        int cd = c->getAttackCooldown();
        int dmg = static_cast<int>(std::lround(c->getDamagePerTick() * (cd > 0 ? cd : 1)));
        std::printf(" dmg=%4d hitspeed=%.1fs range=%.2f sight=%.1f radius=%.2f air=%s fly=%s splash=%.1f kamikaze=%s deploy=%.1fs",
            dmg, cd / 10.0, c->getAttackRange(), c->sightRange, CombatEntity::effectiveRadiusOf(*e),
            c->targetsAir ? "Y" : "n", c->isFlying ? "Y" : "n", c->splashRadius,
            c->dieAfterFirstHit ? "Y" : "n", c->deployTicksRemaining / 10.0);
        if (c->periodicIntervalTicks) std::printf(" spawnEvery=%.1fs", c->periodicIntervalTicks / 10.0);
        if (c->deathEffect) std::printf(" [deathEffect]");
        if (c->periodicHotIntervalTicks) std::printf(" hotSpawnEvery=%.1fs", c->periodicHotIntervalTicks / 10.0);
        if (c->periodicSideOffset > 0.0f) std::printf(" spawnSide=+-%.1f", c->periodicSideOffset);
        if (c->healOnHitAmount) std::printf(" [healOnHit=%d%s cap=%d]", c->healOnHitAmount,
            c->healOnHitSecondPulseDelayTicks ? "x2" : "", c->healOnHitMaxHp);
        if (c->abilitySlotCardId) std::printf(" [slotCard=%d]", c->abilitySlotCardId);
        if (c->isHero) std::printf(" [HERO cost=%.1f uses=%d cd=%d]", c->abilityElixirCost, c->abilityUsesRemaining, c->abilityCooldownTicks);
    }
    if (t) std::printf(" speed=%.3f tiles/s", t->getSpeed() * 10.0f);
    std::printf("\n");
}

// Median per-tick displacement over a window, times 10: tiles per second.
double measureSpeed(Sim& s, const std::shared_ptr<Entity>& e, int ticks) {
    std::vector<double> steps;
    Vector2D prev = e->position;
    for (int i = 0; i < ticks && e->isAlive(); ++i) {
        s.step();
        double d = std::hypot(e->position.x - prev.x, e->position.y - prev.y);
        steps.push_back(d);
        prev = e->position;
    }
    if (steps.empty()) return 0.0;
    std::sort(steps.begin(), steps.end());
    return steps[steps.size() / 2] * 10.0;
}

// ---------------------------------------------------------------- sections

void sectionIdentity() {
    header("1. IDENTITY -- stats read off spawned entities (base, evolved, hero, children)");
    { Sim s; identity("Furnace (70)", s.spawn(70, 8.5f, 6.0f, 0).front()); }
    { Sim s; identity("Evo Furnace (138, evolved)", s.spawn(138, 8.5f, 6.0f, 0, true).front()); }
    { Sim s; auto v = s.spawn(78, 8.5f, 8.0f, 0); std::printf("  Bats bodies: %zu\n", v.size()); identity("Bats (78)", v.front()); }
    { Sim s; auto v = s.spawn(126, 8.5f, 8.0f, 0, true); std::printf("  Evo Bats bodies: %zu\n", v.size()); identity("Evo Bats (126, evolved)", v.front()); }
    { Sim s; identity("Giant Skeleton (39)", s.spawn(39, 8.5f, 8.0f, 0).front()); }
    { Sim s; identity("Berserker (51)", s.spawn(51, 8.5f, 8.0f, 0).front()); }
    { Sim s; identity("Goblin Hut (95)", s.spawn(95, 8.5f, 8.0f, 0).front()); }
    { Sim s; s.spawn(70, 8.5f, 3.5f, 0); identity("  child: Furnace Fire Spirit", s.stepUntil("Fire Spirit", 0, 200)); }
    { Sim s; s.spawn(95, 3.0f, 10.0f, 0); s.dummy(3.0f, 14.0f, 1, 100000);
      identity("  child: Goblin Hut Spear Gob", s.stepUntil("Spear Goblins", 0, 100)); }
    { Sim s; s.spawn(110, 8.5f, 22.0f, 0); identity("  child: Graveyard Skeleton", s.stepUntil("Skeletons", 0, 100)); }
    { Sim s; s.spawn(101, 8.5f, 8.0f, 0); identity("  child: BB Barbarian", s.stepUntil("Barbarians", 0, 60)); }
    { Sim s; s.spawn(174, 8.5f, 8.0f, 0); identity("  child: Hero BB Barbarian", s.stepUntilCombat("Hero Barbarian Barrel", 0, 60)); }
    std::printf("  -- playable counterparts, for comparison --\n");
    { Sim s; identity("Fire Spirit card (73)", s.spawn(73, 8.5f, 8.0f, 0).front()); }
    { Sim s; identity("Spear Goblins card (23)", s.spawn(23, 8.5f, 8.0f, 0).front()); }
    { Sim s; identity("Barbarians card (8)", s.spawn(8, 8.5f, 8.0f, 0).front()); }
}

void sectionSpeed() {
    header("2. MOVEMENT SPEED -- measured displacement, no enemies in sight (tiles/s)");
    std::printf("tiers: SLOW %.3f  MEDIUM %.3f  FAST %.3f  VERY_FAST %.3f\n",
        SPEED_SLOW * MOVEMENT_SPEED_SCALE * 10, SPEED_MEDIUM * MOVEMENT_SPEED_SCALE * 10,
        SPEED_FAST * MOVEMENT_SPEED_SCALE * 10, SPEED_VERY_FAST * MOVEMENT_SPEED_SCALE * 10);
    auto row = [](const char* label, double v, const char* realTier) {
        std::printf("  %-26s %.3f tiles/s   real: %s\n", label, v, realTier);
    };
    { Sim s; auto e = s.spawn(70, 8.5f, 4.0f, 0).front(); s.step(12); row("Furnace", measureSpeed(s, e, 30), "Medium (since 2025-10-06)"); }
    { Sim s; auto e = s.spawn(138, 8.5f, 4.0f, 0, true).front(); s.step(12); row("Evo Furnace", measureSpeed(s, e, 30), "Medium"); }
    { Sim s; auto e = s.spawn(51, 8.5f, 4.0f, 0).front(); s.step(12); row("Berserker", measureSpeed(s, e, 30), "Fast"); }
    { Sim s; auto e = s.spawn(39, 8.5f, 4.0f, 0).front(); s.step(12); row("Giant Skeleton", measureSpeed(s, e, 30), "Medium"); }
    { Sim s; auto e = s.spawn(78, 8.5f, 4.0f, 0).front(); s.step(12); row("Bats (one bat)", measureSpeed(s, e, 30), "Very Fast"); }
    { Sim s; s.spawn(101, 8.5f, 4.0f, 0); auto e = s.stepUntil("Barbarians", 0, 60); s.step(12); row("BB Barbarian", measureSpeed(s, e, 30), "Medium"); }
    { Sim s; s.spawn(174, 8.5f, 4.0f, 0); std::shared_ptr<Entity> e = s.stepUntilCombat("Hero Barbarian Barrel", 0, 60); s.step(12); row("Hero BB Barbarian", measureSpeed(s, e, 30), "Medium"); }
    { Sim s; s.spawn(110, 8.5f, 4.0f, 0); auto e = s.stepUntil("Skeletons", 0, 100); s.step(12); row("Graveyard Skeleton", measureSpeed(s, e, 30), "Fast"); }
    { Sim s; auto f = s.spawn(70, 8.5f, 2.0f, 0).front(); auto e = s.stepUntil("Fire Spirit", 0, 200); f->hp = 0; s.step(12); row("Furnace Fire Spirit", measureSpeed(s, e, 30), "Very Fast"); }
    { Sim s; auto h = s.spawn(95, 8.5f, 4.0f, 0).front(); s.step(12); h->hp = 0; auto e = s.stepUntil("Spear Goblins", 0, 20); s.step(12); row("Hut Spear Goblin (death)", measureSpeed(s, e, 30), "Very Fast"); }
}

// Attacker vs a stationary dummy; returns hit ticks (relative to spawn) and amounts.
struct Cadence { std::vector<int> ticks; std::vector<int> amounts; };
Cadence cadence(Sim& s, int attackerId, int spawnTick) {
    Cadence c;
    for (auto& d : s.rec->dmg) if (d.attackerId == attackerId) { c.ticks.push_back(d.tick - spawnTick); c.amounts.push_back(d.amount); }
    return c;
}
void printCadence(const char* label, const Cadence& c, const char* real) {
    std::printf("  %-24s first hit @%s  intervals(t)=[%s]  dmg/hit=%s   real: %s\n", label,
        c.ticks.empty() ? "never" : (std::to_string(c.ticks.front()) + "t").c_str(),
        intervals(c.ticks).c_str(), c.amounts.empty() ? "-" : std::to_string(c.amounts.front()).c_str(), real);
}

void sectionCadence() {
    header("3. ATTACK CADENCE -- vs a stationary 0-damage dummy (ticks after spawn; deploy is 10t)");
    {   Sim s; auto e = s.spawn(51, 8.5f, 10.0f, 0).front(); s.dummy(8.5f, 11.6f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Berserker @100% hp", cadence(s, e->id, t0), "102 dmg every 0.6s, first hit 0.2s"); }
    {   Sim s; auto e = s.spawn(51, 8.5f, 10.0f, 0).front(); e->hp = 448; s.dummy(8.5f, 11.6f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Berserker @50% hp", cadence(s, e->id, t0), "unchanged: 0.6s"); }
    {   Sim s; auto e = s.spawn(51, 8.5f, 10.0f, 0).front(); e->hp = 90; s.dummy(8.5f, 11.6f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Berserker @10% hp", cadence(s, e->id, t0), "unchanged: 0.6s"); }
    {   Sim s; auto e = s.spawn(39, 8.5f, 10.0f, 0).front(); s.dummy(8.5f, 11.8f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Giant Skeleton", cadence(s, e->id, t0), "276 every 1.3s, first hit 0.3s"); }
    {   Sim s; auto v = s.spawn(78, 8.5f, 10.0f, 0); s.dummy(8.5f, 11.5f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Bats (bat #1)", cadence(s, v.front()->id, t0), "81 every 1.2s, first hit 0.6s"); }
    {   Sim s; s.spawn(101, 8.5f, 5.0f, 0); auto e = s.stepUntil("Barbarians", 0, 60); int t0 = s.tick();
        s.dummy(e->position.x, e->position.y + 1.4f, 1, 1000000); s.step(80);
        printCadence("BB Barbarian", cadence(s, e->id, t0), "192 every 1.4s, first hit 0.4s"); }
    {   Sim s; s.spawn(110, 8.5f, 8.0f, 0); auto e = s.stepUntil("Skeletons", 0, 100); int t0 = s.tick();
        s.dummy(e->position.x, e->position.y + 1.2f, 1, 1000000); s.step(80);
        printCadence("Graveyard Skeleton", cadence(s, e->id, t0), "81 every 1.1s, first hit 0.5s"); }
    {   Sim s; auto e = s.spawn(70, 8.5f, 8.0f, 0).front(); s.dummy(8.5f, 12.5f, 1, 1000000);
        int t0 = s.tick(); s.step(80); printCadence("Furnace (own attack)", cadence(s, e->id, t0), "179 (DeckShop: 135) every 1.7s"); }
    {   Sim s; auto h = s.spawn(95, 3.0f, 10.0f, 0).front(); (void)h; s.dummy(3.0f, 14.0f, 1, 1000000);
        auto e = s.stepUntil("Spear Goblins", 0, 100); int t0 = s.tick(); s.step(80);
        printCadence("Hut Spear Goblin", cadence(s, e->id, t0), "81 every 1.6s (Aug 2026), first hit 0.5s"); }
}

void sectionFurnace() {
    header("4. FURNACE -- Fire Spirit schedule, and what a spirit does on contact");
    auto schedule = [](const char* label, bool evolved, bool withTarget, const char* real) {
        Sim s;
        // (8.5, 6.0): clear of the King Tower's 2.0 footprint, so no collision
        // nudges the Furnace or its spawns.
        auto f = s.spawn(evolved ? 138 : 70, 8.5f, 6.0f, 0, evolved).front();
        if (withTarget) s.dummy(8.5f, 10.5f, 1, 100000000);  // in the Furnace's 5.5 range from the start
        else if (auto fc = std::dynamic_pointer_cast<CombatEntity>(f)) fc->applyFreeze(100000000, 0.0f);  // held in place, nothing in sight
        int t0 = s.tick();
        std::vector<int> ticks; std::vector<float> xs;
        std::set<int> seen;
        bool attacked = false;
        for (int i = 0; i < 160; ++i) {
            s.step();
            for (auto& e : s.b().getEntities())
                if (e->name == "Fire Spirit" && e->team == 0 && !seen.count(e->id)) {
                    seen.insert(e->id); ticks.push_back(s.tick() - t0); xs.push_back(e->position.x - f->position.x);
                }
        }
        for (auto& d : s.rec->dmg) if (d.attackerId == f->id) attacked = true;
        std::printf("  %-34s furnace attacked=%s  spirit spawns @[", label, attacked ? "Y" : "n");
        for (size_t i = 0; i < ticks.size(); ++i) std::printf("%s%d", i ? "," : "", ticks[i]);
        std::printf("]t  lateral offsets=[");
        for (size_t i = 0; i < xs.size(); ++i) std::printf("%s%+.1f", i ? "," : "", xs[i]);
        std::printf("]\n      real: %s\n", real);
    };
    schedule("base, nothing to attack (16 s)", false, false, "one spirit every 7 s (5 s per wiki since 2026-08-04), in front");
    schedule("base, attacking the whole time", false, true, "same as above");
    schedule("EVO, nothing to attack (16 s)", true, false, "hot spawn ONLY while attacking -> base interval here");
    schedule("EVO, attacking the whole time", true, true, "every 2.4 s, alternating left/right");

    // What a Furnace spirit does against a tight clump of three ground units.
    auto clump = [](const char* label, bool fromFurnace, const char* real) {
        Sim s;
        std::shared_ptr<Entity> spirit;
        if (fromFurnace) {
            auto f = s.spawn(70, 8.5f, 2.0f, 0).front();
            spirit = s.stepUntil("Fire Spirit", 0, 200);
            f->hp = 0;  // keep only the spirit
            s.step();
        } else {
            spirit = s.spawn(73, 8.5f, 2.0f, 0).front();
        }
        int sx = static_cast<int>(spirit->position.x * 10) ;
        (void)sx;
        auto a = s.dummy(8.0f, 6.5f, 1, 1000);
        auto b = s.dummy(9.0f, 6.5f, 1, 1000);
        auto c = s.dummy(8.5f, 7.3f, 1, 1000);
        s.step(80);
        std::set<int> victims; int hits = 0;
        for (auto& d : s.rec->dmg) if (d.attackerId == spirit->id) { hits++; victims.insert(d.targetId); }
        std::printf("  %-34s hits=%d distinct victims=%zu/3  spirit alive after=%s  dummy hp: %d %d %d\n      real: %s\n",
            label, hits, victims.size(), spirit->isAlive() ? "YES" : "no", a->hp, b->hp, c->hp, real);
    };
    clump("Furnace-spawned spirit vs 3-clump", true, "one kamikaze hit, 215 area dmg (radius 2.3) to all 3, then dies");
    clump("Fire Spirit card (73) vs 3-clump", false, "same");
}

void sectionEvoBats() {
    header("5. EVO BATS -- self-heal on attack");
    Sim s;
    auto v = s.spawn(126, 8.5f, 10.0f, 0, true);
    auto bat = std::dynamic_pointer_cast<CombatEntity>(v.front());
    s.dummy(8.5f, 11.5f, 1, 1000000);
    s.step(10);                // deploy
    bat->hp = 40;              // wounded, so every heal is visible
    std::printf("  start hp=%d (max per engine %d)\n  hp after each of its hits:", bat->hp, bat->healOnHitMaxHp);
    size_t seen = 0; int lastHp = bat->hp;
    for (int i = 0; i < 160; ++i) {
        s.step();
        auto hits = s.hitsBy(bat->id);
        if (hits.size() > seen) { seen = hits.size(); std::printf(" %d(+%d)", bat->hp, bat->hp - lastHp); lastHp = bat->hp; }
    }
    std::printf("\n  real: spawn hp 122, each attack heals 2 pulses x 38 = 76 (0.5 s apart), overheal cap 244\n");
}

void sectionPoison() {
    header("6. POISON -- pulses, radius edge, Crown Tower damage, slow, friendly fire");
    {   // Pulse timing and the radius edge. Dummies are troops (radius 0.4).
        Sim s;
        std::vector<float> offs = { 0.0f, 3.0f, 3.4f, 3.6f, 3.8f };
        std::vector<std::shared_ptr<MeleeTroop>> ds;
        for (size_t i = 0; i < offs.size(); ++i) {
            float a = 1.2566f * static_cast<float>(i);   // 72-degree spokes
            ds.push_back(s.dummy(8.5f + offs[i] * std::cos(a), 20.0f + offs[i] * std::sin(a), 1, 100000));
        }
        auto sp = s.spawn(32, 8.5f, 20.0f, 0).front();
        int t0 = s.tick(); s.step(100);
        std::vector<int> pulseTicks; std::map<int, int> perDummy;
        for (auto& d : s.rec->dmg) if (d.attackerId == sp->id) {
            if (pulseTicks.empty() || pulseTicks.back() != d.tick - t0) pulseTicks.push_back(d.tick - t0);
            perDummy[d.targetId] += d.amount;
        }
        std::printf("  pulses @[");
        for (size_t i = 0; i < pulseTicks.size(); ++i) std::printf("%s%d", i ? "," : "", pulseTicks[i]);
        std::printf("]t  (%zu pulses)\n", pulseTicks.size());
        for (size_t i = 0; i < ds.size(); ++i)
            std::printf("  dummy centre %.1f from Poison centre (edge %.1f): total %d\n", offs[i], offs[i] - 0.4f, perDummy[ds[i]->id]);
        std::printf("  real: 8 pulses x 92 = 736, 1/s; a unit whose hitbox overlaps the 3.5 circle is hit\n");
    }
    {   // Crown Tower damage.
        Sim s;
        auto tw = s.princessTower(1, 3.0f);
        int before = tw->hp;
        auto sp = s.spawn(32, 3.0f, 27.0f, 0).front();
        s.step(100);
        int perPulse = 0;
        for (auto& d : s.rec->dmg) if (d.attackerId == sp->id && d.targetIsTower) { perPulse = d.amount; break; }
        std::printf("  Princess Tower: %d per pulse, %d total   real: 21 per pulse, 168 total (June 2026 -9%%, fixed 26 Aug)\n",
            perPulse, before - tw->hp);
    }
    {   // Slow: an enemy walking through the cloud.
        Sim s;
        auto knight = s.spawn(0, 8.5f, 24.0f, 1).front();  // enemy Knight walking toward our side
        s.step(12);
        double outside = measureSpeed(s, knight, 10);
        s.spawn(32, knight->position.x, knight->position.y - 2.0f, 0);
        double inside = measureSpeed(s, knight, 15);
        std::printf("  enemy Knight speed: outside %.3f, inside Poison %.3f tiles/s   real: 15%% slower inside\n", outside, inside);
    }
    {   // Friendly fire: our own Graveyard skeleton inside our own Poison.
        Sim s;
        s.spawn(110, 8.5f, 22.0f, 0);
        auto sk = s.stepUntil("Skeletons", 0, 100);
        auto sp = s.spawn(32, sk->position.x, sk->position.y, 0).front();
        int hp0 = sk->hp; s.step(30);
        int ff = 0; for (auto& d : s.rec->dmg) if (d.attackerId == sp->id && d.targetTeam == 0) ff += d.amount;
        std::printf("  own skeleton in own Poison: damage taken from it = %d (hp %d -> %d)   real: 0\n", ff, hp0, sk->hp);
    }
    {   // Enemy Bats (81 hp) die to one pulse; Evo Bats (121) survive one.
        Sim s;
        auto bats = s.spawn(78, 3.0f, 21.0f, 1); auto evo = s.spawn(126, 14.0f, 21.0f, 1, true);
        s.spawn(32, 3.0f, 21.0f, 0); s.spawn(32, 14.0f, 21.0f, 0);   // 11 tiles apart: no overlap
        s.step(1);
        int batsAlive = 0, evoAlive = 0;
        for (auto& e : bats) batsAlive += e->isAlive(); for (auto& e : evo) evoAlive += e->isAlive();
        std::printf("  after the first pulse: Bats alive %d/5, Evo Bats alive %d/5   real: 0/5 and 5/5\n", batsAlive, evoAlive);
    }
}

void sectionGiantSkeleton() {
    header("7. GIANT SKELETON -- death bomb: fuse, radius, damage, and against a Crown Tower");
    {
        Sim s;
        auto gs = s.spawn(39, 8.5f, 12.0f, 0).front();
        s.step(10);
        std::vector<float> offs = { 1.0f, 1.9f, 2.4f, 2.9f, 3.3f };
        std::vector<std::shared_ptr<MeleeTroop>> ds;
        for (float o : offs) ds.push_back(s.dummy(gs->position.x + o, gs->position.y, 1, 5000));
        std::vector<int> hpAtDeath; for (auto& d : ds) hpAtDeath.push_back(d->hp);
        gs->hp = 0;
        int t0 = s.tick();
        std::vector<int> firstDropTick(ds.size(), -1);
        for (int i = 0; i < 50; ++i) {
            s.step();
            for (size_t k = 0; k < ds.size(); ++k)
                if (firstDropTick[k] < 0 && ds[k]->hp < 5000) firstDropTick[k] = s.tick() - t0;
        }
        for (size_t k = 0; k < ds.size(); ++k)
            std::printf("  dummy %.1f from GS centre: took %4d, at t+%d   \n", offs[k], 5000 - ds[k]->hp, firstDropTick[k]);
        std::printf("  real (since 2026-05-04): bomb lands on death, explodes 3.0 s later, radius 3, dmg 688*1.29 ~= 887\n"
                    "        (official notes' own scale: 209 -> 269, +29%%), knockback\n");
    }
    {
        // The chip play: Giant Skeleton reaches the Princess Tower, dies there.
        Sim s(false);
        s.freezeTowers(0);                           // our towers idle
        for (auto& e : s.b().getEntities())          // enemy King idle too, Princess active
            if (e->isTower() && e->team == 1 && e->symbol == 'R')
                if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) c->applyFreeze(100000000, 0.0f);
        auto gs = s.spawn(39, 3.0f, 15.0f, 0).front();
        auto tw = s.princessTower(1, 3.0f);
        int t;
        for (t = 0; t < 400; ++t) {                  // walk until it has hit the tower once
            s.step();
            bool hit = false;
            for (auto& d : s.rec->dmg) if (d.attackerId == gs->id && d.targetIsTower) hit = true;
            if (hit) break;
        }
        double gap = gs->position.distanceTo(tw->position);
        int beforeBomb = tw->hp;
        gs->hp = 0;                                  // dies in place
        s.step(1);
        int instant = beforeBomb - tw->hp;
        s.step(40);
        // Subtract anything else that hit the tower (nothing else is alive on our side).
        std::printf("  GS died %.2f tiles (centre-centre) from the Princess Tower: bomb tower damage instantly=%d, within 4 s=%d\n",
            gap, instant, beforeBomb - tw->hp);
        std::printf("  real: the bomb explodes on the tower for ~887 (no longer x2 since 2026-05-04)\n");
    }
}

void sectionGraveyard() {
    header("8. GRAVEYARD -- spawn schedule, positions, and on a Princess Tower");
    {
        Sim s;
        auto sp = s.spawn(110, 8.5f, 22.0f, 0).front();
        (void)sp;
        int t0 = s.tick();
        std::vector<std::pair<int, Vector2D>> out; std::set<int> seen;
        for (int i = 0; i < 120; ++i) {
            s.step();
            for (auto& e : s.b().getEntities())
                if (e->name == "Skeletons" && e->team == 0 && !seen.count(e->id)) {
                    seen.insert(e->id);
                    out.push_back({ s.tick() - t0, e->position });
                }
        }
        std::printf("  %zu skeletons. spawn ticks: ", out.size());
        for (auto& p : out) std::printf("%d ", p.first);
        std::printf("\n  distance from the cast point at spawn: ");
        for (auto& p : out) std::printf("%.2f ", std::hypot(p.second.x - 8.5f, p.second.y - 22.0f));
        std::printf("\n  real: 12 skeletons, first at 2.2 s then every 0.5 s, on a fixed ring (~3.3 tiles), one on the tower\n");
    }
    {
        Sim s(false);
        s.freezeTowers(0);
        auto tw = s.princessTower(1, 3.0f);
        int before = tw->hp;
        s.spawn(110, 3.0f, 27.0f, 0);
        s.step(150);
        // Split by target: skeletons that miss this tower can walk on to another.
        // The Graveyard spell itself deals 0, and each of its pulses still
        // emits a 0-amount event on every enemy in its disc: not hits.
        int onHits = 0, onDmg = 0, offHits = 0, offDmg = 0, emptyPulses = 0;
        for (auto& d : s.rec->dmg) {
            if (d.attackerTeam != 0 || !d.targetIsTower) continue;
            if (d.amount == 0) { emptyPulses++; continue; }
            if (d.targetId == tw->id) { onHits++; onDmg += d.amount; } else { offHits++; offDmg += d.amount; }
        }
        std::printf("  cast on an active Princess Tower, 15 s: %d skeleton hits for %d on it (hp lost %d), %d for %d on other towers"
            " [+%d zero-damage spell pulses]\n", onHits, onDmg, before - tw->hp, offHits, offDmg, emptyPulses);
    }
}

void sectionBarbarianBarrel() {
    header("9. BARBARIAN BARREL -- corridor, reach, ground-only, the Barbarian");
    Sim s;
    struct Probe { float dx, dy; bool air; std::shared_ptr<MeleeTroop> d; };
    std::vector<Probe> ps = {
        { 0.0f, 1.0f, false }, { 1.2f, 2.0f, false }, { 1.6f, 2.0f, false }, { 1.9f, 2.0f, false },
        { 0.0f, 4.8f, false }, { 0.0f, 5.2f, false }, { 0.0f, 2.0f, true } };
    for (auto& p : ps) p.d = s.dummy(8.5f + p.dx, 6.0f + p.dy, 1, 10000, p.air);
    auto sp = s.spawn(101, 8.5f, 6.0f, 0).front();
    int t0 = s.tick(); s.step(30);
    for (auto& p : ps) {
        int got = 0; for (auto& d : s.rec->dmg) if (d.attackerId == sp->id && d.targetId == p.d->id) got += d.amount;
        std::printf("  %s dummy at lateral %.1f, forward %.1f: %d\n", p.air ? "AIR   " : "ground", p.dx, p.dy, got);
    }
    auto barb = s.firstNamed("Barbarians", 0);
    int spawnT = -1; for (auto& e : s.rec->spawned) if (barb && e.entityId == barb->id) spawnT = e.tick - t0;
    std::printf("  Barbarian appeared at t+%d, %.2f tiles forward of the cast point, hp %d\n",
        spawnT, barb ? barb->position.y - 6.0f : -1.0f, barb ? barb->hp : -1);
    // Through the card's own definition, as playCard and the Python mask ask.
    const CardDefinition* bb = CardRegistry::getInstance().getCard(101);
    for (float y : { 14.9f, 15.4f, 16.0f, 17.0f, 17.4f, 17.6f })
        std::printf("  cast at y=%.1f (river %.1f-%.1f): %s\n", y, s.b().getRiverStart(), s.b().getRiverEnd(),
            s.g.isValidPlacement(0, 8.5f, y, bb->isSpell, bb->placementRadius, bb->deployAnywhere,
                                 bb->rollRange > 0.0f, bb->castOwnSideOnly) ? "ALLOWED" : "refused");
    std::printf("  real: 232 dmg, width 2.6 (hits units overlapping +-1.3), range 4.5, ground only, no knockback;\n"
                "        Barbarian 716 hp / 192 dmg / 1.4 s / Medium, 1 s deploy; castable on own side only\n");
}

void sectionHeroBarrel() {
    header("10. HERO BARBARIAN BARREL -- 'Rowdy Reroll' through the real play path");
    {
        Sim s;
        GameManager& g = s.g;
        bool handOk = g.setHand(0, { 174, 32, 39, 110 });
        g.setElixir(0, 10.0f);
        bool played = g.playCard(0, 174, 8.5f, 8.0f);
        s.step(25);
        std::shared_ptr<CombatEntity> hero;
        for (auto& e : s.b().getEntities())
            if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) if (c->isHero && c->team == 0) hero = c;
        float elixirBefore = g.getElixir(0);
        bool ready = g.isChampionAbilityReady(0, 1);
        bool activated = g.activateChampionAbility(0, 1);
        std::printf("  setHand=%s playCard=%s  hero Barbarian on board=%s (cardId %d)\n",
            handOk ? "ok" : "REFUSED", played ? "ok" : "REFUSED", hero ? "yes" : "no", hero ? hero->cardId : 0);
        std::printf("  isChampionAbilityReady(slot 1)=%s  activateChampionAbility(slot 1)=%s  elixir %.2f -> %.2f\n",
            ready ? "true" : "FALSE", activated ? "true" : "FALSE", elixirBefore, g.getElixir(0));
        std::printf("  real: the ability button is live once the Barbarian lands (1 elixir, single use)\n");
    }
    {
        // The effect itself, invoked directly on the Barbarian.
        Sim s;
        s.spawn(174, 8.5f, 4.0f, 0);
        auto hero = s.stepUntilCombat("Hero Barbarian Barrel", 0, 60);
        s.step(10);
        hero->hp = 300;
        Vector2D p = hero->position;
        auto d0 = s.dummy(p.x, p.y + 2.0f, 1, 10000);
        auto d1 = s.dummy(p.x + 1.0f, p.y + 2.0f, 1, 10000);
        auto d2 = s.dummy(p.x + 1.6f, p.y + 2.0f, 1, 10000);
        auto da = s.dummy(p.x - 0.3f, p.y + 1.0f, 1, 10000, true);
        int hpBefore = hero->hp;
        bool fired = hero->activateAbility(s.b());
        std::printf("  direct activateAbility=%s  barbarian moved %.2f tiles forward\n", fired ? "true" : "false", hero->position.y - p.y);
        std::printf("  damage: lateral 0.0 -> %d, lateral 1.0 -> %d, lateral 1.6 -> %d, AIR unit -> %d\n",
            10000 - d0->hp, 10000 - d1->hp, 10000 - d2->hp, 10000 - da->hp);
        std::printf("  barbarian hp %d -> %d (heal %d)   uses left %d\n", hpBefore, hero->hp, hero->hp - hpBefore, hero->abilityUsesRemaining);
        std::printf("  real: 3-tile roll, width 2.6, ground only, heals the Barbarian 50%% of damage dealt,\n"
                    "        Crown Tower damage 116, single use (Aug 2026), 1 elixir\n");
    }
}

void sectionGoblinHut() {
    header("11. GOBLIN HUT -- proximity gate, schedule, lifetime, death spawn, air targeting");
    {
        Sim s;
        auto hut = s.spawn(95, 3.0f, 10.0f, 0).front();
        int t0 = s.tick();
        s.step(50);
        int idle = static_cast<int>(s.named("Spear Goblins", 0).size());
        auto enemy = s.dummy(3.0f, 15.5f, 1, 100000000);   // 5.5 from the hut
        int tEnemy = s.tick() - t0;
        std::vector<int> spawnT; std::set<int> seen;
        int hutDiedAt = -1;
        for (int i = 0; i < 300; ++i) {
            s.step();
            for (auto& e : s.b().getEntities())
                if (e->name == "Spear Goblins" && e->team == 0 && !seen.count(e->id)) { seen.insert(e->id); spawnT.push_back(s.tick() - t0); }
            if (hutDiedAt < 0 && !hut->isAlive()) hutDiedAt = s.tick() - t0;
        }
        std::printf("  first 5 s with no enemy in range: %d spawns\n", idle);
        std::printf("  enemy appears at t+%d. spawns at: ", tEnemy);
        for (int t : spawnT) std::printf("%d ", t);
        std::printf("\n  intervals: [%s]  hut expired at t+%d\n", intervals(spawnT).c_str(), hutDiedAt);
        std::printf("  real: nothing while idle; a Spear Goblin 0.5 s after an enemy enters 6 tiles, then every 2.2 s;\n"
                    "        30 s lifetime; +1 Spear Goblin on death; hp 1180 (1228 before 2025-10-06 -4%%)\n");
    }
    {
        Sim s;
        s.spawn(95, 3.0f, 10.0f, 0);
        auto air = s.dummy(3.0f, 14.0f, 1, 100000, true);
        s.step(100);
        int got = 0; for (auto& d : s.rec->dmg) if (d.targetId == air->id) got += d.amount;
        std::printf("  vs a FLYING enemy 4 tiles away for 10 s: spear goblins spawned=%zu, damage to it=%d   real: they shoot air\n",
            s.named("Spear Goblins", 0).size(), got);
    }
}

void sectionInteractions() {
    header("12. KEY INTERACTIONS");
    {   // Giant Skeleton cannot hit Bats (it targets ground only).
        Sim s;
        auto gs = s.spawn(39, 8.5f, 10.0f, 1).front();
        auto bats = s.spawn(78, 8.5f, 11.0f, 0);
        s.step(60);
        std::set<int> batIds; for (auto& b : bats) batIds.insert(b->id);
        int gsOnBats = 0; for (auto& d : s.rec->dmg) if (d.attackerId == gs->id && batIds.count(d.targetId)) gsOnBats += d.amount;
        int alive = 0; for (auto& b : bats) alive += b->isAlive();
        std::printf("  enemy Giant Skeleton vs our Bats, 6 s: GS damage to bats=%d, bats alive=%d/5   real: 0, 5/5\n", gsOnBats, alive);
    }
    {   // Barbarian Barrel over a Graveyard: every skeleton in the corridor dies.
        Sim s;
        s.spawn(110, 8.5f, 12.0f, 1);   // enemy graveyard on our side
        s.step(70);
        int before = static_cast<int>(s.named("Skeletons", 1).size());
        s.spawn(101, 8.5f, 8.5f, 0);
        s.step(20);
        int after = static_cast<int>(s.named("Skeletons", 1).size());
        std::printf("  enemy Graveyard skeletons: %d alive, %d after one Barbarian Barrel   real: every skeleton inside the corridor dies\n", before, after);
    }
    {   // Evo Bats on an idle Princess Tower: they can take a big chunk.
        Sim s(false);
        s.freezeTowers(0);
        auto tw = s.princessTower(1, 3.0f);
        auto v = s.spawn(126, 3.0f, 22.5f, 0, true);
        int before = tw->hp;
        s.step(200);
        int alive = 0; for (auto& b : v) alive += b->isAlive();
        std::printf("  lone Evo Bats vs an active Princess Tower, 20 s: tower lost %d, bats alive %d/5\n", before - tw->hp, alive);
    }
}


void sectionStunQuirk() {
    header("13. STUN AND SLOW (item 32m) -- a stun blocks a ready attack; a slow after a stun stays a slow");
    Sim s;
    auto knight = std::dynamic_pointer_cast<CombatEntity>(s.spawn(0, 8.5f, 20.0f, 1).front());
    s.step(15);                                   // deployed, idle, cooldown 0
    knight->applyFreeze(50, 0.0f);                // a 5 s stun (Freeze is 4 s at level 11)
    auto target = s.dummy(8.5f, 18.6f, 0, 100000); // walks into its reach while it is stunned
    s.step(40);
    int hits = 0, dealt = 0;
    for (auto& d : s.rec->dmg) if (d.attackerId == knight->id) { hits++; dealt += d.amount; }
    std::printf("  enemy Knight stunned for 5 s, target placed in reach: %d hit(s) for %d during the stun   real: 0\n", hits, dealt);

    // freezeSlow is only ever min()'d, never reset when a freeze ends.
    auto slowedSpeed = [](bool stunFirst) {
        Sim s2;
        auto k = std::dynamic_pointer_cast<CombatEntity>(s2.spawn(0, 8.5f, 5.0f, 0).front());
        s2.step(12);
        if (stunFirst) { k->applyFreeze(5, 0.0f); s2.step(30); }   // a 0.5 s Zap-style stun, long over
        k->applyFreeze(30, 0.65f);                                  // then an Ice Wizard-style 35% slow
        return measureSpeed(s2, k, 20);
    };
    double clean = slowedSpeed(false), afterStun = slowedSpeed(true);
    std::printf("  Knight under a 0.65 slow: %.3f tiles/s fresh, %.3f tiles/s if it was stunned 3 s earlier   expected: %.3f both\n",
        clean, afterStun, SPEED_MEDIUM * MOVEMENT_SPEED_SCALE * 10 * 0.65);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    sectionIdentity();
    sectionSpeed();
    sectionCadence();
    sectionFurnace();
    sectionEvoBats();
    sectionPoison();
    sectionGiantSkeleton();
    sectionGraveyard();
    sectionBarbarianBarrel();
    sectionHeroBarrel();
    sectionGoblinHut();
    sectionInteractions();
    sectionStunQuirk();
    return 0;
}
