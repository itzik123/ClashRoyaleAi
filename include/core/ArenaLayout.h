#pragma once

// The arena's fixed geometry, in ONE place.
//
// Before this header the same numbers lived in three: Board's leftBridge/
// rightBridge members, GameManager::setupTowers' six addTower literals, and
// HeuristicOpponent's own LEFT_BRIDGE_X/RIGHT_BRIDGE_X -- which had already gone
// stale (3.5/13.5) against Board's own values before anyone noticed. LanePath.h
// needs the Princess COLUMNS, which existed only inside GameManager, and copying
// them a fourth time is what this header exists to prevent.
//
// COORDINATE CONVENTION -- the thing that was actually wrong.
//
// x is a CELL INDEX, clamped by Board::clampToBoard to [0, WIDTH-1]. So the
// board's centre -- and the fixed point of the mirror mirrorX -- is
// (WIDTH-1)/2 = 8.5, NOT 9.0. This is the x-analogue of
// ClashEnv::extractObservationForTeam's y -> (HEIGHT-1) - y, and it is the
// convention Board::isBackRowDeadZone already used for its own centre.
//
// Corrected 2026-08-21 from a professional player's audit. The King sat at 9.0
// and the left Princess at 4.0 -- symmetric about 9.0 rather than 8.5, the same
// half-tile convention error the river carried before it was re-centred on 16.5.
// The 2026-07-30 change that moved them there was fit against real recordings
// and was anchored on this convention error; the corrected layout is
// self-consistent and mirror-symmetric, which the old one was only about the
// wrong centre.
namespace ArenaLayout {

inline constexpr int WIDTH = 18;
inline constexpr int HEIGHT = 34;

// The mirror that maps one team's frame onto the other, and one lane onto the
// other. Its fixed point is the board centre.
inline constexpr float CENTER_X = (WIDTH - 1) / 2.0f;   // 8.5
constexpr float mirrorX(float x) { return (WIDTH - 1) - x; }
constexpr float mirrorY(float y) { return (HEIGHT - 1) - y; }

// Princess Tower columns. 3 tiles wide (Tower::getCollisionRadius 1.5), so the
// left one occupies columns 2-4 and the right 13-15.
inline constexpr float LEFT_LANE_X = 3.0f;
inline constexpr float RIGHT_LANE_X = mirrorX(LEFT_LANE_X);   // 14.0

// Bridge centres. A bridge is TWO tiles wide -- the real river row reads
//
//     column  012345678901234567
//             WWBBWWWWWWWWWWBBWW      (W water, B bridge)
//
// so columns 2-3 and 14-15, and each centre lands on the SEAM between its two
// tiles rather than on a tile. Cell i covers [i-0.5, i+0.5], so a centre of 2.5
// with Board::BRIDGE_HALF_WIDTH = 1.0 spans exactly cells 2 and 3. Centring on a
// tile is what made that corridor three columns wide.
inline constexpr float LEFT_BRIDGE_X = 2.5f;
inline constexpr float RIGHT_BRIDGE_X = mirrorX(LEFT_BRIDGE_X);   // 14.5
inline constexpr float BRIDGE_Y = 16.5f;                          // the river band's centre

// Tower rows, per team. Team 0 defends the low-y end.
inline constexpr float KING_Y_TEAM0 = 2.5f;
inline constexpr float PRINCESS_Y_TEAM0 = 6.0f;
constexpr float kingY(int team) { return team == 0 ? KING_Y_TEAM0 : mirrorY(KING_Y_TEAM0); }
constexpr float princessY(int team) { return team == 0 ? PRINCESS_Y_TEAM0 : mirrorY(PRINCESS_Y_TEAM0); }

// Which lane a position belongs to.
//
// Nearest-bridge, matching the rule Board::getNextWaypoint already uses to pick
// a crossing -- so a unit's lane objective and the bridge it is actually routed
// over agree by construction, and it can never be sent to one bridge while
// aiming at the other lane's tower. Re-evaluated per call rather than latched at
// spawn, so nothing new has to be carried through Board::deepCopy or snapshot().
constexpr bool isLeftLane(float x) { return x < CENTER_X; }
constexpr float bridgeXFor(float x) { return isLeftLane(x) ? LEFT_BRIDGE_X : RIGHT_BRIDGE_X; }
constexpr float laneXFor(float x) { return isLeftLane(x) ? LEFT_LANE_X : RIGHT_LANE_X; }

}  // namespace ArenaLayout
