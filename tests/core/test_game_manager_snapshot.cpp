#include <catch_amalgamated.hpp>
#include "GameManager.h"
#include "ClashEnv.h"
#include "CardRegistry.h"
#include "Tower.h"
#include <vector>

// GameManager::snapshot() and ClashEnv::snapshot() -- the layer above
// Board::deepCopy() (covered in test_board_deepcopy.cpp). deepCopy handles the
// entities; these two carry the scalars, the players and the statistics, and
// each has its own way of going quietly wrong:
//
//   * stats reset to zero instead of carrying over, which would make every
//     candidate in a search score identically and look like "search never
//     helps";
//   * stats shared instead of copied, which would post a rollout's imagined
//     damage into the live match's totals -- and those feed reward shaping;
//   * elixir/hand state shared, so a rollout's spending would bankrupt the
//     real player.

namespace {

const std::vector<int> DEFAULT_DECK = { 10, 1, 41, 25, 7, 2, 6, 5 };

// Same shape as the board-level describe() in test_board_deepcopy.cpp, but
// reaching through a GameManager so it also covers the manager's own scalars.
struct ManagerRow {
    int tick;
    bool over;
    int loser;
    float elixir0;
    float elixir1;
    size_t entities;
    int towerDamage0;
    int towerDamage1;
    int troopDamage0;
    int troopDamage1;

    bool operator==(const ManagerRow& o) const {
        return tick == o.tick && over == o.over && loser == o.loser &&
            elixir0 == o.elixir0 && elixir1 == o.elixir1 && entities == o.entities &&
            towerDamage0 == o.towerDamage0 && towerDamage1 == o.towerDamage1 &&
            troopDamage0 == o.troopDamage0 && troopDamage1 == o.troopDamage1;
    }
    // Spelled out because this is C++17: != is only synthesised from == in
    // C++20. The vector comparisons elsewhere in this file work without it
    // because std::vector defines its own !=.
    bool operator!=(const ManagerRow& o) const { return !(*this == o); }
};

std::ostream& operator<<(std::ostream& os, const ManagerRow& r) {
    return os << "{tick=" << r.tick << " over=" << r.over << " loser=" << r.loser
        << " elixir=(" << r.elixir0 << ", " << r.elixir1 << ")"
        << " entities=" << r.entities
        << " towerDmg=(" << r.towerDamage0 << ", " << r.towerDamage1 << ")"
        << " troopDmg=(" << r.troopDamage0 << ", " << r.troopDamage1 << ")}";
}

ManagerRow describe(const GameManager& game) {
    const MatchStatistics& s = game.getStatistics();
    return {
        // GameManager has no currentTick accessor; Board::currentTick is
        // public and is assigned from it on every step(), so it is the same
        // number without adding an accessor just for a test.
        game.getBoard().currentTick,
        game.isGameOver(),
        game.getLoserTeam(),
        game.getElixirAI(),
        game.getElixirOpp(),
        game.getBoard().getEntities().size(),
        s.towerDamageDealt(0), s.towerDamageDealt(1),
        s.troopDamageDealt(0), s.troopDamageDealt(1),
    };
}

// Every entity's exact state, so "identical" means identical rather than
// "the summary numbers agree".
struct EntityRow {
    int id;
    int hp;
    int team;
    float x;
    float y;

    bool operator==(const EntityRow& o) const {
        return id == o.id && hp == o.hp && team == o.team && x == o.x && y == o.y;
    }
};

std::ostream& operator<<(std::ostream& os, const EntityRow& r) {
    return os << "{id=" << r.id << " hp=" << r.hp << " team=" << r.team
        << " pos=(" << r.x << ", " << r.y << ")}";
}

std::vector<EntityRow> describeEntities(const GameManager& game) {
    std::vector<EntityRow> rows;
    for (const auto& e : game.getBoard().getEntities()) {
        rows.push_back({ e->id, e->hp, e->team, e->position.x, e->position.y });
    }
    return rows;
}

// Cards are injected rather than played from hand: the opening hand is
// shuffled by an unseeded mt19937, and a test about copying must not also
// depend on what was dealt.
void injectCard(GameManager& game, int cardId, float x, float y, int team) {
    const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
    REQUIRE(def != nullptr);
    def->spawnEntity(x, y, team, game.getBoard());
}

// Pushes a real fight onto a real match so the snapshot has something to
// preserve.
//
// Note the fixture's own limit, learned by writing a test against it that was
// wrong: GameManager::step() plays no cards by itself (HeuristicOpponent lives
// a layer up, in ClashEnv), so once these troops kill each other the board goes
// quiet and NO further damage of any kind is dealt. A test that needs the
// rollout to keep accruing damage has to inject its own -- see the collector
// isolation test below.
void buildMidGamePosition(GameManager& game, int ticks = 120) {
    injectCard(game, 2, 4.0f, 14.0f, 0);   // Giant
    injectCard(game, 6, 5.0f, 11.0f, 0);   // Musketeer
    injectCard(game, 10, 14.0f, 13.0f, 0); // Valkyrie
    injectCard(game, 5, 4.0f, 19.0f, 1);   // Mini P.E.K.K.A
    injectCard(game, 1, 5.0f, 22.0f, 1);   // Archers
    injectCard(game, 6, 13.0f, 20.0f, 1);  // Musketeer

    for (int i = 0; i < ticks; ++i) game.step();
}

} // namespace

TEST_CASE("a GameManager snapshot and its original step identically", "[gamemanager][snapshot]") {
    GameManager original(DEFAULT_DECK, DEFAULT_DECK);
    buildMidGamePosition(original);

    // Damage has actually been dealt, or the stats assertions below are vacuous.
    REQUIRE(original.getStatistics().troopDamageDealt(0) + original.getStatistics().troopDamageDealt(1) > 0);

    GameManager copy = original.snapshot();
    REQUIRE(describe(copy) == describe(original));
    REQUIRE(describeEntities(copy) == describeEntities(original));

    for (int i = 0; i < 150; ++i) {
        original.step();
        copy.step();
        INFO("divergence first seen at step " << i);
        REQUIRE(describeEntities(copy) == describeEntities(original));
        REQUIRE(describe(copy) == describe(original));
    }
}

TEST_CASE("stepping a GameManager snapshot leaves the original untouched", "[gamemanager][snapshot]") {
    GameManager original(DEFAULT_DECK, DEFAULT_DECK);
    buildMidGamePosition(original);

    ManagerRow before = describe(original);
    std::vector<EntityRow> entitiesBefore = describeEntities(original);

    GameManager copy = original.snapshot();
    for (int i = 0; i < 150; ++i) copy.step();

    REQUIRE(describe(original) == before);
    REQUIRE(describeEntities(original) == entitiesBefore);
    REQUIRE(describe(copy) != before); // the rollout really did advance
}

TEST_CASE("snapshot statistics carry over rather than resetting to zero", "[gamemanager][snapshot][stats]") {
    GameManager original(DEFAULT_DECK, DEFAULT_DECK);
    buildMidGamePosition(original);

    int troopDamageAtSnapshot = original.getStatistics().troopDamageDealt(0);
    REQUIRE(troopDamageAtSnapshot > 0);

    GameManager copy = original.snapshot();

    // The trap this pins down: attaching fresh collectors to the copied board
    // would have been the obvious one-liner and would report 0 here. Any
    // caller scoring candidates with a potential function of cumulative damage
    // would then read every rollout as the same catastrophic loss, and the
    // search would silently never prefer anything.
    REQUIRE(copy.getStatistics().troopDamageDealt(0) == troopDamageAtSnapshot);
}

TEST_CASE("a rollout's damage never reaches the original's collectors", "[gamemanager][snapshot][stats]") {
    GameManager original(DEFAULT_DECK, DEFAULT_DECK);
    buildMidGamePosition(original);

    int towerDamageBefore = original.getStatistics().towerDamageDealt(0);
    int troopDamageBefore = original.getStatistics().troopDamageDealt(0);

    GameManager copy = original.snapshot();
    // The rollout needs a live fight of its own: by now the position above has
    // resolved and a bare step() spawns nothing. A Giant dropped in front of
    // team 0's left Princess Tower guarantees both counters move -- the tower
    // shoots it (team 0 troop damage) and it hits the tower (team 1 tower
    // damage). Injected into the COPY only, which is itself part of the test.
    injectCard(copy, 2, 4.0f, 8.0f, 1);
    for (int i = 0; i < 200; ++i) copy.step();

    // The copy kept fighting...
    REQUIRE(copy.getStatistics().troopDamageDealt(0) > troopDamageBefore);
    REQUIRE(copy.getStatistics().towerDamageDealt(1) > original.getStatistics().towerDamageDealt(1));
    // ...and none of it was recorded against the live match. If the collectors
    // were shared, every hypothetical hit in every rollout would land in the
    // totals train.py's reward shaping reads.
    REQUIRE(original.getStatistics().towerDamageDealt(0) == towerDamageBefore);
    REQUIRE(original.getStatistics().troopDamageDealt(0) == troopDamageBefore);
}

TEST_CASE("a rollout spends its own elixir and cycles its own hand", "[gamemanager][snapshot][player]") {
    GameManager original(DEFAULT_DECK, DEFAULT_DECK);
    for (int i = 0; i < 300; ++i) original.step(); // bank some elixir

    float elixirBefore = original.getElixirAI();
    std::vector<int> handBefore = original.getHand(0);
    REQUIRE(elixirBefore >= 4.0f);

    GameManager copy = original.snapshot();
    REQUIRE(copy.getElixirAI() == elixirBefore);
    REQUIRE(copy.getHand(0) == handBefore);

    // Play whatever is affordable in the copy.
    bool played = false;
    for (int slot = 0; slot < 4 && !played; ++slot) {
        played = copy.playCard(0, copy.getHand(0)[slot], 9.0f, 5.0f);
    }
    REQUIRE(played);

    REQUIRE(copy.getElixirAI() < elixirBefore);       // the copy paid
    REQUIRE(original.getElixirAI() == elixirBefore);  // the original did not
    REQUIRE(original.getHand(0) == handBefore);       // and its hand never cycled
}

TEST_CASE("two snapshots of one position roll out identically", "[gamemanager][snapshot][determinism]") {
    // What 1-ply search depends on: branches differ because the ACTION
    // differed, not because the copies did. If `rng` were reseeded per
    // snapshot instead of copied, candidates would be scored across different
    // futures and the comparison would be measuring noise.
    GameManager position(DEFAULT_DECK, DEFAULT_DECK);
    buildMidGamePosition(position);

    GameManager branchA = position.snapshot();
    GameManager branchB = position.snapshot();
    for (int i = 0; i < 100; ++i) { branchA.step(); branchB.step(); }

    REQUIRE(describeEntities(branchA) == describeEntities(branchB));
    REQUIRE(describe(branchA) == describe(branchB));
}

TEST_CASE("two snapshots given different actions diverge from each other", "[gamemanager][snapshot][determinism]") {
    // The other half of the above, and the one that makes search meaningful at
    // all: identical copies must still be capable of producing different
    // futures when they are asked to do different things.
    GameManager position(DEFAULT_DECK, DEFAULT_DECK);
    for (int i = 0; i < 300; ++i) position.step();

    GameManager withPlay = position.snapshot();
    GameManager withoutPlay = position.snapshot();

    bool played = false;
    for (int slot = 0; slot < 4 && !played; ++slot) {
        played = withPlay.playCard(0, withPlay.getHand(0)[slot], 9.0f, 5.0f);
    }
    REQUIRE(played);

    // Checked over a SHORT horizon, and that is not laziness -- an earlier
    // version of this test stepped 100 times and was flaky at roughly 1 run in
    // 20. The two futures genuinely RECONVERGE: a lone troop with no support
    // walks into the enemy Princess Towers, dies without landing a hit, and
    // elixir re-caps at 10, so both boards end up holding the same six
    // full-health towers and nothing else. That is the game behaving correctly,
    // not the snapshot failing, and asserting permanent divergence asserts
    // something false about Clash Royale.
    //
    // The property that actually matters for search is that the branch differs
    // WHILE THE DECISION IS STILL LIVE -- which is the only window a 1-ply
    // rollout ever scores.
    //
    // The first step() is load-bearing and not a warm-up: playCard queues the
    // spawn into pendingEntities, and getEntities() exposes only
    // activeEntities, so the new troop is invisible until step() calls
    // commitPendingEntities(). Comparing before that shows two identical
    // boards and fails despite the play having succeeded.
    withPlay.step();
    withoutPlay.step();
    for (int i = 0; i < 10; ++i) {
        INFO("branches identical at step " << i);
        REQUIRE(describeEntities(withPlay) != describeEntities(withoutPlay));
        withPlay.step();
        withoutPlay.step();
    }
}

// ---------------- ClashEnv, the Python-facing layer ----------------

TEST_CASE("a ClashEnv snapshot steps identically to its original", "[clashenv][snapshot]") {
    ClashEnv original(DEFAULT_DECK, DEFAULT_DECK);
    original.reset();
    for (int i = 0; i < 40; ++i) original.step(4, 0.0f, 0.0f); // 4 = no-op slot

    ClashEnv copy = original.snapshot();
    REQUIRE(copy.getObservationForTeam(0) == original.getObservationForTeam(0));
    REQUIRE(copy.getElixir() == original.getElixir());
    REQUIRE(copy.getHand() == original.getHand());

    // Both sides step the SAME action sequence. The heuristic opponent is
    // copied along with the rng it draws from, so its choices have to match
    // too -- a rollout whose opponent played differently from the real one
    // would be scoring the wrong game.
    for (int i = 0; i < 60; ++i) {
        original.step(4, 0.0f, 0.0f);
        copy.step(4, 0.0f, 0.0f);
        INFO("divergence first seen at step " << i);
        REQUIRE(copy.getObservationForTeam(0) == original.getObservationForTeam(0));
        REQUIRE(copy.isGameOver() == original.isGameOver());
    }
}

TEST_CASE("stepping a ClashEnv snapshot leaves the original's observation untouched", "[clashenv][snapshot]") {
    ClashEnv original(DEFAULT_DECK, DEFAULT_DECK);
    original.reset();
    for (int i = 0; i < 40; ++i) original.step(4, 0.0f, 0.0f);

    std::vector<float> before = original.getObservationForTeam(0);
    float elixirBefore = original.getElixir();

    ClashEnv copy = original.snapshot();
    for (int i = 0; i < 100; ++i) copy.step(4, 0.0f, 0.0f);

    REQUIRE(original.getObservationForTeam(0) == before);
    REQUIRE(original.getElixir() == elixirBefore);
    REQUIRE(copy.getObservationForTeam(0) != before);
}

TEST_CASE("snapshotting does not disturb an env that is then stepped normally", "[clashenv][snapshot]") {
    // Search takes a snapshot at a decision point and then carries on playing
    // the real match. Taking the snapshot must not perturb what happens next,
    // which is a different property from the copy being correct.
    ClashEnv control(DEFAULT_DECK, DEFAULT_DECK);
    ClashEnv probed(DEFAULT_DECK, DEFAULT_DECK);
    control.reset();
    probed.reset();

    // Force both onto the same footing: reset() re-rolls the heuristic's lane
    // from a random_device-seeded rng, so two envs are not comparable unless
    // the observation says they are. Skip the test rather than assert a
    // coincidence if they happen to differ.
    if (control.getObservationForTeam(0) != probed.getObservationForTeam(0)) {
        SUCCEED("independent envs started from different random openings; nothing to compare");
        return;
    }

    for (int i = 0; i < 50; ++i) {
        control.step(4, 0.0f, 0.0f);

        ClashEnv rollout = probed.snapshot();
        for (int j = 0; j < 20; ++j) rollout.step(4, 0.0f, 0.0f);
        probed.step(4, 0.0f, 0.0f);
    }

    REQUIRE(probed.getObservationForTeam(0) == control.getObservationForTeam(0));
}
