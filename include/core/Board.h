#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include "ArenaLayout.h"
#include "Entity.h"
#include "StatsEventBus.h"

class Board {
private:
    int width, height;
    std::vector<std::shared_ptr<Entity>> activeEntities;
    std::vector<std::shared_ptr<Entity>> pendingEntities;
    int idCounter = 1000;

    // Centered on 16.5 to match the tower layout's own symmetry (King
    // 2.5<->30.5, Princess 6.0<->27.0, mirrored by y -> 33-y exactly as
    // ClashEnv::extractObservationForTeam does it) -- NOT 17.0, which was
    // half a tile off-centre and gave team 0 one more placeable row than
    // team 1 in each side's own mirrored frame (team 0 could reach row 15,
    // team 1 could not, confirmed empirically: 15/20 vs 0/20 placements).
    // Both teams' GameManager::isValidPlacement bound now sits at the same
    // mirrored row (15.0), and python_ai/model.py's single shared
    // own_half_rows mask (already computed once and applied to both sides)
    // becomes correct for both instead of encoding the old asymmetry.
    float riverY_start = 15.5f;
    float riverY_end = 17.5f;
    // x on an 18-wide board is a CELL INDEX clamped to [0, width-1], so the
    // board's own centre is (18-1)/2 = 8.5 and the mirror of a column is
    // 17 - x -- exactly the convention isBackRowDeadZone() below already uses
    // for its centre, and the x-analogue of extractObservationForTeam's
    // y -> 33 - y.
    //
    // A real bridge is TWO tiles wide, and the real arena's river row reads
    //
    //        column  012345678901234567
    //                WWBBWWWWWWWWWWBBWW      (W water, B bridge)
    //
    // i.e. columns 2-3 and 14-15. So each centre sits on the SEAM between its
    // two tiles, not on a tile: 2.5 and 14.5, which mirror onto each other
    // under 17 - x. Putting the centre on a tile (3.0 / 14.0) is what made
    // clampToBoard's +/-1.0 corridor three tiles wide instead of two; with the
    // centre on the seam that same +/-1.0 spans exactly cells 2 and 3, since
    // cell i covers [i - 0.5, i + 0.5].
    //
    // These were 4.0 / 14.0 until 2026-08-20 -- symmetric about 9.0 rather
    // than 8.5, the same half-tile convention error the river itself carried
    // before it was re-centred on 16.5.
    Vector2D leftBridge{ ArenaLayout::LEFT_BRIDGE_X, ArenaLayout::BRIDGE_Y };
    Vector2D rightBridge{ ArenaLayout::RIGHT_BRIDGE_X, ArenaLayout::BRIDGE_Y };

    // Scratch for resolveCollisions' per-entity property hoist and
    // colliders()' index cache. Members rather than function locals so the
    // heap allocation is paid once per Board rather than once per tick;
    // mutable because colliders() is const. Neither holds meaning between
    // calls -- bodyScratch is fully rebuilt on entry, and colliderCache is
    // guarded by collidersDirty.
    struct Body {
        Entity* entity;
        float radius;
        bool building;
        bool troop;
        bool flying;
        bool alive;
    };
    struct Collider { size_t index; float radius; };
    mutable std::vector<Body> bodyScratch;
    mutable std::vector<Collider> colliderCache;
    mutable bool collidersDirty = true;

    // Real-map sync: the arena is 18x34, not 18x32 -- there's one extra row
    // behind each King Tower that this engine used to just not have. Most of
    // that row is dead space (matches the decorative rock/wall texture
    // flanking the real King Tower), except a BACK_ROW_OPENING_WIDTH-wide gap
    // centered on the board, which is real, placeable ground. See
    // isBackRowDeadZone() below.
    static constexpr float BACK_ROW_OPENING_HALF_WIDTH = 3.0f;

public:
    // Half of a bridge's two-tile width. Named rather than repeated as a bare
    // 1.0f at each clampToBoard comparison, because it is only correct in
    // company with a seam-centred leftBridge/rightBridge -- see those.
    static constexpr float BRIDGE_HALF_WIDTH = 1.0f;

    // Cell i covers [i - 0.5, i + 0.5] (the convention isOnBridge's comment
    // states and clampToBoard's +/-BRIDGE_HALF_WIDTH depends on), so the
    // board's PHYSICAL extent runs half a cell beyond the outermost cell
    // INDEX on every side: x in [-0.5, width - 0.5], y in [-0.5, height-0.5].
    //
    // Named because GameManager::isValidPlacement's footprint check is the
    // one place that needs the physical edge rather than the index edge, and
    // getting the two confused is not a small error: checking a footprint
    // against the INDEX range instead would reject a 0.4-radius troop at
    // x = 0 and x = width-1, silently deleting two of eighteen columns from
    // the action space while looking like a bounds fix.
    static constexpr float CELL_HALF_EXTENT = 0.5f;

private:

public:
    // Public, not a getter-wrapped private member: every fire site
    // (entities, GameManager) already holds a Board& and just calls
    // board.statsEvents.notifyX(...) directly, the same way they already
    // read board.getEntities()/call board.addEntity(...).
    StatsEventBus statsEvents;

    // Set once per tick by GameManager::step() (board.currentTick =
    // currentTick;), read by combat code deep in the update()/performAttack()
    // call chain (Projectile::applyHit, AreaSpell::update, Building/
    // MeleeTroop/BuildingTargeter::performAttack) to stamp DamageDealtEvent.
    // Unlike commitPendingEntities()/cleanDeadEntities() below -- which take
    // an explicit tick parameter because GameManager calls them directly --
    // there's no parameter-passing route all the way down through every
    // update() override without threading a tick argument through the whole
    // entity hierarchy, which would be a far bigger, more invasive change
    // than this one field. Board doesn't act on this value itself, purely a
    // read-only convenience for whoever's stamping an event.
    int currentTick = 0;

    // Elixir Collector: accumulated here by ElixirGrantEffect (a periodic
    // effect, like Board::currentTick above there's no clean route from
    // deep inside CombatEntity::update() to PlayerState's elixir, which
    // Board doesn't even know exists -- so this is a small drop-box
    // instead. GameManager::step() drains both entries into
    // playerAI/playerOpponent.elixir once per tick and resets them to 0;
    // Board doesn't act on this value itself.
    float pendingElixirGrant[2] = { 0.0f, 0.0f };

    Board(int w = 18, int h = 34) : width(w), height(h) {}

    int allocateId() { return idCounter++; }

    // Fully independent copy of this board: every entity is duplicated via
    // Entity::snapshot() instead of shared, so the copy can be stepped without
    // touching the original. This is what makes decision-time search possible
    // (CLAUDE.md open problem #2) -- roll candidate actions forward on a copy,
    // score them, keep the best.
    //
    // NOT the same as copy-constructing a Board, which is still the implicit
    // shallow copy: activeEntities is a vector of shared_ptr, so a plain copy
    // shares every entity and stepping one board mutates the other. The
    // implicit copy is left alone deliberately (GameManager holds a Board by
    // value and deleting it would make GameManager non-copyable), so this is
    // additive -- but anything doing lookahead wants THIS, not `Board b = a;`.
    //
    // Two things are deliberately NOT carried across, both because carrying
    // them would corrupt the live match rather than the copy:
    //
    //   * `statsEvents` starts EMPTY. StatsEventBus holds shared_ptr to
    //     stateful collectors (DamageStatsCollector's running totals,
    //     KillStatsCollector's attribution map, MatchOutcomeCollector). Copying
    //     the subscriber list would post every hypothetical hit in every
    //     rollout into the REAL match's statistics -- and those feed the reward
    //     shaping, so a search would silently rewrite the returns it is being
    //     scored against. Same failure shape as the Projectile alias below, one
    //     level up. A caller that genuinely wants stats off a rollout
    //     subscribes its own collectors to the copy.
    //   * Nothing outside Board. currentTick is mirrored here for damage
    //     stamping, but GameManager's own currentTick/gameOver/loserTeam/elixir
    //     are its members, not this one's -- a GameManager-level snapshot needs
    //     those too and is a separate piece of work.
    Board deepCopy() const {
        Board copy(width, height);

        // Geometry copied explicitly rather than trusting the fresh Board's
        // in-class initialisers to still agree. They do today -- nothing
        // mutates them -- but the river has already moved twice in this
        // project's history, and if it ever becomes configurable the silent
        // failure is every rollout quietly reverting to the default arena.
        copy.riverY_start = riverY_start;
        copy.riverY_end = riverY_end;
        copy.leftBridge = leftBridge;
        copy.rightBridge = rightBridge;

        // idCounter must carry across or the copy re-issues ids the original
        // already handed out: a projectile spawned during a rollout would
        // collide with an existing entity's id, and every id-keyed lookup
        // (resolveCurrentTarget, the remap below) would join on the wrong row.
        copy.idCounter = idCounter;
        copy.currentTick = currentTick;
        copy.pendingElixirGrant[0] = pendingElixirGrant[0];
        copy.pendingElixirGrant[1] = pendingElixirGrant[1];

        std::unordered_map<int, std::shared_ptr<Entity>> byOldId;
        byOldId.reserve(activeEntities.size() + pendingEntities.size());

        // Pending entities are copied too, not dropped: a card played this
        // tick lives there until the next commitPendingEntities(), so skipping
        // them would make a snapshot taken mid-tick lose the very placement a
        // search is trying to evaluate.
        copy.activeEntities.reserve(activeEntities.size());
        for (const auto& e : activeEntities) {
            auto c = e->snapshot();
            byOldId[e->id] = c;
            copy.activeEntities.push_back(std::move(c));
        }
        copy.pendingEntities.reserve(pendingEntities.size());
        for (const auto& e : pendingEntities) {
            auto c = e->snapshot();
            byOldId[e->id] = c;
            copy.pendingEntities.push_back(std::move(c));
        }

        // Second pass, and it must be second: a projectile can be homing on an
        // entity that had not been copied yet when the projectile itself was,
        // so the map has to be complete before anything joins on it. A no-op
        // for every type except Projectile -- see Entity::remapSnapshotReferences.
        for (const auto& e : copy.activeEntities) e->remapSnapshotReferences(byOldId);
        for (const auto& e : copy.pendingEntities) e->remapSnapshotReferences(byOldId);

        return copy;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Exposed so callers that need to reason about the river band (e.g.
    // GameManager's own-half placement check) derive it from here instead
    // of re-guessing the same two numbers as a second, driftable copy.
    float getRiverStart() const { return riverY_start; }
    float getRiverEnd() const { return riverY_end; }
    // Read-only, same rationale as getRiverStart/getRiverEnd above: the bridge
    // columns are board geometry, and anything that needs them (lane pathing,
    // HeuristicOpponent's own LEFT_BRIDGE_X/RIGHT_BRIDGE_X, the audit tools)
    // should read them here rather than keep a second copy that goes stale --
    // which is exactly what happened to HeuristicOpponent's 3.5/13.5.
    // Is this column a bridge? THE single definition of that question.
    //
    // Both the physics and the observation encoder need it, and until
    // 2026-08-21 they each answered it their own way: clampToBoard compared
    // against leftBridge/rightBridge +/- BRIDGE_HALF_WIDTH, while
    // ClashEnv::extractObservationForTeam painted channel 8 from a hardcoded
    // `(x >= 3 && x <= 4) || (x >= 13 && x <= 14)`. The arena correction moved
    // the bridges and only the physics followed, so the network was being told
    // two of the four real bridge columns were water (2 and 15) and two water
    // columns were bridge (4 and 13) -- the agent's map of where it can cross
    // disagreed with where it actually could.
    //
    // A shared FORMULA would not have prevented that; a shared FUNCTION does.
    //
    // Takes a float so it serves both callers unchanged: continuous positions
    // from clampToBoard, and integer cell centres from the encoder. Cell i
    // covers [i-0.5, i+0.5], so passing the integer i asks "is the centre of
    // cell i on a bridge", which for a seam-centred 2.5 with half-width 1.0
    // selects exactly cells 2 and 3 -- the real river row's `WWBB...`.
    bool isOnBridge(float x) const {
        return (x >= leftBridge.x - BRIDGE_HALF_WIDTH && x <= leftBridge.x + BRIDGE_HALF_WIDTH)
            || (x >= rightBridge.x - BRIDGE_HALF_WIDTH && x <= rightBridge.x + BRIDGE_HALF_WIDTH);
    }

    const Vector2D& getLeftBridge() const { return leftBridge; }
    const Vector2D& getRightBridge() const { return rightBridge; }

    // True for the unplaceable corners of the two new back rows (y in
    // [0,1) or (height-2, height-1], outside the BACK_ROW_OPENING_HALF_WIDTH
    // gap centered on the board) -- everywhere else returns false, including
    // every row that existed before this back-row expansion. Exposed so
    // GameManager::isValidPlacement checks it alongside the plain width/
    // height bounds check, the same way it already reads getRiverStart()/
    // getRiverEnd() instead of re-deriving river geometry itself.
    bool isBackRowDeadZone(float x, float y) const {
        bool inBackRow = (y < 1.0f) || (y > static_cast<float>(height) - 2.0f);
        if (!inBackRow) return false;
        float centerX = (static_cast<float>(width) - 1.0f) / 2.0f;
        return (x < centerX - BACK_ROW_OPENING_HALF_WIDTH) || (x > centerX + BACK_ROW_OPENING_HALF_WIDTH);
    }

    void addEntity(std::shared_ptr<Entity> entity) {
        pendingEntities.push_back(entity);
    }

    // Lets a caller (GameManager::playCard, to find the Champion entity a
    // deploy just spawned) inspect what's about to be committed, without
    // waiting for the next commitPendingEntities() -- CardDefinition::
    // spawnEntity is void-returning (fire and forget), so this is the only
    // handle back to a just-spawned entity this tick.
    size_t pendingEntityCount() const { return pendingEntities.size(); }
    const std::shared_ptr<Entity>& getPendingEntity(size_t index) const { return pendingEntities[index]; }

    // tick defaults to 0 (rather than Board storing its own tick mirror,
    // which would be a second, driftable copy of GameManager's currentTick
    // -- exactly the kind of duplication this codebase avoids elsewhere,
    // e.g. getRiverStart()/getWidth() existing so GameManager doesn't
    // re-guess them). GameManager::step() passes the real tick; the several
    // existing tickless test call sites keep compiling unchanged.
    void commitPendingEntities(int tick = 0) {
        if (!pendingEntities.empty()) {
            for (const auto& e : pendingEntities) {
                statsEvents.notifyEntitySpawned({ e->id, e->cardId, e->team, tick });
            }
            activeEntities.insert(activeEntities.end(), pendingEntities.begin(), pendingEntities.end());
            pendingEntities.clear();
            collidersDirty = true;   // membership changed -- see colliders()
        }
    }

    const std::vector<std::shared_ptr<Entity>>& getEntities() const {
        return activeEntities;
    }

    void cleanDeadEntities(int tick = 0) {
        // Nothing died: bail before three full passes over the entity list.
        // The overwhelming majority of ticks take this branch (a match of a
        // few thousand ticks contains a few dozen deaths), and the three
        // passes below are each provably no-ops in that case -- neither loop
        // has a live body and remove_if erases nothing -- so this is an exact
        // short-circuit, not a behaviour change.
        bool anyDead = false;
        for (const auto& e : activeEntities) {
            if (!e->isAlive()) { anyDead = true; break; }
        }
        if (!anyDead) return;
        collidersDirty = true;   // membership is about to change -- see colliders()

        // Fire death effects (e.g. Golem spawning two Golemites) before the
        // erase below, purely via Entity's own virtual onDeath() -- Board
        // never needs to know which entities are CombatEntity-shaped enough
        // to actually have one, same as clampPosition().
        for (const auto& e : activeEntities) {
            if (!e->isAlive()) {
                e->onDeath(*this);
                statsEvents.notifyEntityDied({ e->id, e->cardId, e->team, tick });
            }
        }

        // Notify every still-alive entity of every death that just
        // happened (Skeleton King's soul collection: "a troop dies in his
        // presence") -- a second pass, not folded into the loop above,
        // since a death effect firing (e.g. a spawn) could itself add
        // pending entities, and this only needs to reach entities already
        // on the board this tick, not anything a death effect just created.
        for (const auto& dying : activeEntities) {
            if (!dying->isAlive()) {
                for (const auto& other : activeEntities) {
                    if (other->isAlive()) other->onNearbyDeath(*this, dying->position, dying->team);
                }
            }
        }

        activeEntities.erase(
            std::remove_if(activeEntities.begin(), activeEntities.end(),
                [](const std::shared_ptr<Entity>& e) { return !e->isAlive(); }),
            activeEntities.end()
        );
    }

    // Single source of truth for "push a point out of a circular obstacle,
    // with a small perpendicular slide so it can travel around it instead of
    // sticking". Used for normal movement (resolvePositionAgainstBuildings
    // below) and, identically, by GameManager's post-move collision pass --
    // previously copy-pasted between the two call sites (and a third time,
    // as a near-duplicate, for the mirrored troop2-vs-building1 case).
    static Vector2D pushAwayFrom(Vector2D pos, const Vector2D& obstacleCenter, float minDist) {
        float dx = pos.x - obstacleCenter.x;
        float dy = pos.y - obstacleCenter.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < minDist) {
            // DIRECTION and DISTANCE are separate questions, and the
            // degenerate branch used to conflate them: it set dist = 1.0f
            // purely to make dx/dist a unit vector, and that same 1.0f then
            // flowed into `push = minDist - dist` -- so a point sitting
            // exactly on an obstacle's centre was pushed out by minDist - 1.0
            // instead of by minDist. Measured: a troop landing dead-centre on
            // a Building (minDist 1.4) moved 0.403 tiles and was still inside
            // the footprint; on a King Tower (minDist 2.4) it moved 1.4 of the
            // 2.4 it needed. It escaped over two ticks instead of one, and the
            // shortfall was exactly 1.0 tile every time -- the fake distance.
            //
            // Splitting them leaves the ordinary case bit-identical
            // (ux == dx / dist, same operands in the same order) and makes the
            // coincident case push the full minDist along an arbitrary but
            // fixed +x direction.
            float ux, uy;
            if (dist < 0.001f) {
                ux = 1.0f; uy = 0.0f;   // coincident: any direction will do
            } else {
                ux = dx / dist; uy = dy / dist;
            }
            float push = minDist - dist;
            pos.x += ux * push + uy * 0.05f;
            pos.y += uy * push - ux * 0.05f;
        }
        return pos;
    }

    // Indices into activeEntities of everything with a real collision radius
    // -- Buildings and Towers, which on a typical board is 6 entities out of
    // 30-40. resolvePositionAgainstBuildings runs once per MOVING TROOP per
    // tick and used to walk the whole entity list making a virtual
    // getCollisionRadius() call per candidate, so the cost was quadratic in
    // board population to answer a question about a handful of entities.
    //
    // A radius never changes during an entity's life (Building's is a
    // constant, Tower's keys off `symbol`, everything else inherits Entity's
    // 0), so the only thing that can invalidate this is MEMBERSHIP -- which
    // changes in exactly two places, commitPendingEntities and
    // cleanDeadEntities, both of which set the flag. deepCopy builds a fresh
    // Board, whose flag starts dirty; the implicit shallow copy carries a
    // vector of indices that stay valid against its identical entity vector.
    //
    // Liveness is deliberately NOT cached: hp changes constantly, so the
    // isAlive() check stays in the loop below exactly where it was.
    const std::vector<Collider>& colliders() const {
        if (collidersDirty) {
            colliderCache.clear();
            for (size_t i = 0; i < activeEntities.size(); ++i) {
                const float r = activeEntities[i]->getCollisionRadius();
                if (r > 0.0f) colliderCache.push_back(Collider{ i, r });
            }
            collidersDirty = false;
        }
        return colliderCache;
    }

    Vector2D resolvePositionAgainstBuildings(const Vector2D& pos, int entityId) const {
        Vector2D resolved = pos;
        // Ascending index order, i.e. the same order as the original walk over
        // activeEntities -- the pushes compose, so the order is part of the
        // answer and not an implementation detail.
        for (const Collider& c : colliders()) {
            const auto& entity = activeEntities[c.index];
            if (entity->id == entityId || !entity->isAlive()) continue;

            resolved = pushAwayFrom(resolved, entity->position, c.radius + Entity::IMPLICIT_TROOP_RADIUS);
        }
        return resolved;
    }

    // Physically separates every overlapping pair of entities for one tick,
    // then re-clamps everyone to the board/river. This is "how entities on
    // this board physically interact" -- a Board concern, not a match-rules
    // one, so it lives here rather than in GameManager::step() alongside
    // elixir economy and win-condition checks.
    void resolveCollisions() {
        // Everything the pair loop needs to know about an entity OTHER than
        // its position, read once per entity instead of twice per PAIR.
        //
        // This loop is O(n^2) and was making four virtual calls per pair
        // (getCollisionRadius and isTargetable, on both sides) for facts that
        // are constant across the whole call: a radius never changes, nor does
        // targetability within a tick, and nothing here can kill anything, so
        // isAlive() is fixed too. At 30 entities that is ~1,700 virtual
        // dispatches replaced by 60. Positions are NOT hoisted -- they are the
        // one thing this function mutates, and every read below stays live.
        const size_t count = activeEntities.size();
        bodyScratch.clear();
        bodyScratch.reserve(count);
        for (const auto& e : activeEntities) {
            const float r = e->getCollisionRadius();
            bodyScratch.push_back(Body{
                e.get(), r, r > 0.0f, (r <= 0.0f && e->isTargetable()), e->isFlying, e->isAlive() });
        }

        for (size_t i = 0; i < count; ++i) {
            const Body& b1 = bodyScratch[i];
            if (!b1.alive) continue;
            for (size_t j = i + 1; j < count; ++j) {
                const Body& b2 = bodyScratch[j];

                if (!b2.alive) continue;

                Entity* e1 = b1.entity;
                Entity* e2 = b2.entity;

                const float r1 = b1.radius;
                const float r2 = b2.radius;

                const bool isBuilding1 = b1.building;
                const bool isBuilding2 = b2.building;
                const bool isTroop1 = b1.troop;
                const bool isTroop2 = b2.troop;

                // Flying units pass through everything -- ground and other
                // fliers alike -- so only entities sharing a plane collide.
                if (isTroop1 && isTroop2 && b1.flying == b2.flying) {
                    float dx = e1->position.x - e2->position.x;
                    float dy = e1->position.y - e2->position.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float minRadius = 2.0f * Entity::IMPLICIT_TROOP_RADIUS;

                    if (dist < minRadius) {
                        if (dist < 0.001f) { dx = 1.0f; dy = 0.0f; dist = 1.0f; }
                        float overlap = minRadius - dist;
                        float pushX = (dx / dist) * overlap * 0.5f;
                        float pushY = (dy / dist) * overlap * 0.5f;

                        // Add tiny orthogonal noise to prevent jitter locking
                        float noise = 0.01f;
                        pushX += (dy / dist) * noise;
                        pushY -= (dx / dist) * noise;

                        e1->position.x += pushX;
                        e1->position.y += pushY;
                        e2->position.x -= pushX;
                        e2->position.y -= pushY;
                    }
                }

                if (isTroop1 && isBuilding2 && !b1.flying) {
                    e1->position = pushAwayFrom(e1->position, e2->position, r2 + Entity::IMPLICIT_TROOP_RADIUS);
                }
                if (isTroop2 && isBuilding1 && !b2.flying) {
                    e2->position = pushAwayFrom(e2->position, e1->position, r1 + Entity::IMPLICIT_TROOP_RADIUS);
                }
            }
        }

        // Re-clamp after collision resolution, which can push a troop back
        // into the river or off the board edge. Delegates to each entity's
        // own clampPosition() (a no-op for anything that isn't a Troop) so
        // this respects riverIgnores instead of re-deriving the rule here.
        for (auto& entity : activeEntities) {
            if (entity->isAlive()) {
                entity->clampPosition(*this);
            }
        }
    }

    // Single source of truth for "keep a position on the board and out of the
    // river unless explicitly allowed in it" -- used both right after a troop
    // moves and again after collision resolution potentially nudges it.
    Vector2D clampToBoard(Vector2D pos, bool ignoresRiver) const {
        pos.x = std::max(0.0f, std::min(pos.x, static_cast<float>(width - 1)));
        pos.y = std::max(0.0f, std::min(pos.y, static_cast<float>(height - 1)));

        if (!ignoresRiver && pos.y > riverY_start && pos.y < riverY_end) {
            // isOnBridge(), not an inline comparison: the observation encoder
            // asks the same question and must get the same answer. See that
            // function for the divergence this is preventing.
            if (!isOnBridge(pos.x)) {
                float riverMid = (riverY_start + riverY_end) / 2.0f;
                pos.y = (pos.y < riverMid) ? riverY_start : riverY_end;
            }
        }
        return pos;
    }

    // One definition of "standing on the waypoint", shared with
    // Troop::moveTowards, which refuses to move when it is closer than this to
    // its waypoint. The two MUST agree: if getNextWaypoint can hand back a
    // point the mover is already within this distance of while it still has
    // further to go, that state is absorbing -- the position never changes, so
    // the waypoint never changes, so the unit is stuck for the rest of the
    // match. See the bridge-mouth regression tests in tests/core/test_board.cpp
    // for the measured case this constant exists to prevent.
    static constexpr float WAYPOINT_ARRIVAL_EPS = 0.01f;

    Vector2D getNextWaypoint(const Vector2D& currentPos, const Vector2D& targetPos) const {
        bool isCurrentBelow = currentPos.y <= riverY_start;
        bool isTargetBelow = targetPos.y <= riverY_start;
        bool isCurrentAbove = currentPos.y >= riverY_end;
        bool isTargetAbove = targetPos.y >= riverY_end;

        if ((isCurrentBelow && isTargetBelow) || (isCurrentAbove && isTargetAbove) ||
            (!isCurrentBelow && !isCurrentAbove && !isTargetBelow && !isTargetAbove)) {
            return targetPos;
        }

        float distToLeft = currentPos.distanceTo(leftBridge);
        float distToRight = currentPos.distanceTo(rightBridge);
        float bridgeX = (distToLeft < distToRight) ? leftBridge.x : rightBridge.x;

        if (isCurrentBelow) {
            // The bank classifications above are INCLUSIVE (`y <= riverY_start`),
            // so a unit that has already arrived at the near bank is still
            // "below" and would be routed to the point it is already standing
            // on. Hand it the FAR bank instead -- it has arrived at this leg and
            // the next leg is the crossing itself.
            Vector2D nearBank{bridgeX, riverY_start};
            if (currentPos.distanceTo(nearBank) <= WAYPOINT_ARRIVAL_EPS) {
                return Vector2D{bridgeX, riverY_end};
            }
            return nearBank;
        } else if (isCurrentAbove) {
            Vector2D nearBank{bridgeX, riverY_end};
            if (currentPos.distanceTo(nearBank) <= WAYPOINT_ARRIVAL_EPS) {
                return Vector2D{bridgeX, riverY_start};
            }
            return nearBank;
        } else {
            // The unit is INSIDE the river band, walking toward the bank it
            // will EXIT on. Exactly the same arrival problem as the two
            // branches above, mirrored: once it is within
            // WAYPOINT_ARRIVAL_EPS of that exit edge this leg is finished,
            // and handing back the edge itself is an absorbing state --
            // moveTowards refuses to move, so the position never changes, so
            // the waypoint never changes, and the unit is stuck for the rest
            // of the match a few thousandths of a tile short of dry land.
            //
            // Handing back targetPos instead is not a special case, it is the
            // removal of a discontinuity: the moment the unit's y actually
            // reaches the bank, the first `if` in this function returns
            // targetPos anyway. This just makes the epsilon-neighbourhood
            // agree with the point it surrounds.
            //
            // Measured 2026-08-20: without this, 6-8 of every 34 lone ground
            // units never crossed at all (Giant 27/34, Musketeer 26/34),
            // freezing for 750-850 ticks. See tests/core/test_board.cpp's
            // bridge-EXIT regression block.
            if (isTargetAbove) {
                Vector2D exitBank{bridgeX, riverY_end};
                if (currentPos.distanceTo(exitBank) <= WAYPOINT_ARRIVAL_EPS) {
                    return targetPos;
                }
                return exitBank;
            } else if (isTargetBelow) {
                Vector2D exitBank{bridgeX, riverY_start};
                if (currentPos.distanceTo(exitBank) <= WAYPOINT_ARRIVAL_EPS) {
                    return targetPos;
                }
                return exitBank;
            } else {
                return targetPos;
            }
        }
    }
};