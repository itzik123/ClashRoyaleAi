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
        }
    }

    const std::vector<std::shared_ptr<Entity>>& getEntities() const {
        return activeEntities;
    }

    void cleanDeadEntities(int tick = 0) {
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
            if (dist < 0.001f) {
                dx = 1.0f; dy = 0.0f; dist = 1.0f;
            }
            float push = minDist - dist;
            pos.x += (dx / dist) * push + (dy / dist) * 0.05f;
            pos.y += (dy / dist) * push - (dx / dist) * 0.05f;
        }
        return pos;
    }

    Vector2D resolvePositionAgainstBuildings(const Vector2D& pos, int entityId) const {
        Vector2D resolved = pos;
        for (const auto& entity : activeEntities) {
            float radius = entity->getCollisionRadius();
            if (radius <= 0.0f || entity->id == entityId || !entity->isAlive()) continue;

            resolved = pushAwayFrom(resolved, entity->position, radius + Entity::IMPLICIT_TROOP_RADIUS);
        }
        return resolved;
    }

    // Physically separates every overlapping pair of entities for one tick,
    // then re-clamps everyone to the board/river. This is "how entities on
    // this board physically interact" -- a Board concern, not a match-rules
    // one, so it lives here rather than in GameManager::step() alongside
    // elixir economy and win-condition checks.
    void resolveCollisions() {
        for (size_t i = 0; i < activeEntities.size(); ++i) {
            for (size_t j = i + 1; j < activeEntities.size(); ++j) {
                auto& e1 = activeEntities[i];
                auto& e2 = activeEntities[j];

                if (!e1->isAlive() || !e2->isAlive()) continue;

                float r1 = e1->getCollisionRadius();
                float r2 = e2->getCollisionRadius();

                bool isBuilding1 = r1 > 0.0f;
                bool isBuilding2 = r2 > 0.0f;
                bool isTroop1 = !isBuilding1 && e1->isTargetable();
                bool isTroop2 = !isBuilding2 && e2->isTargetable();

                // Flying units pass through everything -- ground and other
                // fliers alike -- so only entities sharing a plane collide.
                if (isTroop1 && isTroop2 && e1->isFlying == e2->isFlying) {
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

                if (isTroop1 && isBuilding2 && !e1->isFlying) {
                    e1->position = pushAwayFrom(e1->position, e2->position, r2 + Entity::IMPLICIT_TROOP_RADIUS);
                }
                if (isTroop2 && isBuilding1 && !e2->isFlying) {
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
            bool onLeftBridge = (pos.x >= leftBridge.x - BRIDGE_HALF_WIDTH &&
                                 pos.x <= leftBridge.x + BRIDGE_HALF_WIDTH);
            bool onRightBridge = (pos.x >= rightBridge.x - BRIDGE_HALF_WIDTH &&
                                  pos.x <= rightBridge.x + BRIDGE_HALF_WIDTH);
            if (!onLeftBridge && !onRightBridge) {
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