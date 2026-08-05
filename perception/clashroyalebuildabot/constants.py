import os

from clashroyalebuildabot.namespaces import Units

# Directories
SRC_DIR = os.path.dirname(__file__)
DEBUG_DIR = os.path.join(SRC_DIR, "debug")
MODELS_DIR = os.path.join(SRC_DIR, "models")
IMAGES_DIR = os.path.join(SRC_DIR, "images")
EMULATOR_DIR = os.path.join(SRC_DIR, "emulator")
ADB_DIR = os.path.join(EMULATOR_DIR, "platform-tools")
ADB_PATH = os.path.normpath(os.path.join(ADB_DIR, "adb"))
SCREENSHOTS_DIR = os.path.join(DEBUG_DIR, "screenshots")
LABELS_DIR = os.path.join(DEBUG_DIR, "labels")

# Display dimensions
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280

# Screenshot dimensions
SCREENSHOT_WIDTH = 368
SCREENSHOT_HEIGHT = 652

# Playable tiles
#
# REFITTED 2026-08-05 against landmarks in this emulator, in DISPLAY space.
# Upstream's values are off by roughly 10% in x and 4% in y, which is a SCALE
# error, not an offset: taps were correct near the river and drifted to more
# than half a tile wrong by the time they reached our own King. That is why it
# read as an erratic placement bug rather than a clean off-by-one, and why
# adjusting adapter.TILE_Y_OFFSET -- a constant -- could never have fixed it.
#
#                   upstream   fitted
#     TILE_WIDTH          34   37.400
#     TILE_HEIGHT       27.6   28.762
#     TILE_INIT_X         52   23.400
#     TILE_INIT_Y        296  276.190
#
# Fitted from one opening frame of a Training Camp match, 720x1280 via
# `adb exec-out screencap`, using landmarks whose ENGINE coordinates are known
# exactly (perception/geometry.py):
#
#   x   the two bridges, engine x = 4.0 and 14.0, found as the interior gaps
#       in the river's water mask. They came out at 173.0 and 547.0 -- centre
#       360.0, which is the exact centre of a 720-wide display and therefore
#       where engine x = 9.0 (the Kings) has to be. That symmetry is an
#       independent check the fit did not get to choose.
#
#   y   scale from the two princess HP bars, which are 27.0 - 6.0 = 21.0 tiles
#       apart. A bar sits at some unknown offset above its tower, but the SAME
#       offset on both sides, so the separation is exact without ever locating
#       a tower centre. Cross-check: both bars then land 2.26 tiles above their
#       towers, agreeing to within a rounding of each other.
#       Absolute anchor from the river centre, engine y = 16.5.
#
# These are DISPLAY-space (720x1280) and specific to this emulator. A different
# device or a resolution change invalidates all four -- refit with
# tools/fit_tile_grid.py rather than scaling them.
#
# Deliberately edited here, in the vendored copy, rather than overridden
# downstream: `unit_detector._get_tile_xy` and `live/actuator.tile_centre` are
# inverses of each other and both read these. Overriding one would silently
# split perception from actuation.
TILE_HEIGHT = 28.762
TILE_WIDTH = 37.400
N_HEIGHT_TILES = 15
N_WIDE_TILES = 18
TILE_INIT_X = 23.400
TILE_INIT_Y = 276.190
ALLY_TILES = [[x, 0] for x in range(N_WIDE_TILES // 3, 2 * N_WIDE_TILES // 3)]
ALLY_TILES += [
    [x, y] for x in range(N_WIDE_TILES) for y in range(1, N_HEIGHT_TILES)
]
ENEMY_TILES = [[x, 31 - y] for x, y in ALLY_TILES]
ALL_TILES = ALLY_TILES + ENEMY_TILES
LEFT_PRINCESS_TILES = [[3, N_HEIGHT_TILES], [3, N_HEIGHT_TILES + 1]]
LEFT_PRINCESS_TILES += [
    [x, y]
    for x in range(N_WIDE_TILES // 2)
    for y in range(N_HEIGHT_TILES + 2, N_HEIGHT_TILES + 6)
]
RIGHT_PRINCESS_TILES = [[14, N_HEIGHT_TILES], [14, N_HEIGHT_TILES + 1]]
RIGHT_PRINCESS_TILES += [
    [x, y]
    for x in range(N_WIDE_TILES // 2, N_WIDE_TILES)
    for y in range(N_HEIGHT_TILES + 2, N_HEIGHT_TILES + 6)
]
DISPLAY_CARD_Y = 1067
DISPLAY_CARD_INIT_X = 164
DISPLAY_CARD_WIDTH = 117
DISPLAY_CARD_HEIGHT = 147
DISPLAY_CARD_DELTA_X = 136

# Cards
CARD_Y = 543
CARD_INIT_X = 84
CARD_WIDTH = 61
CARD_HEIGHT = 73
CARD_DELTA_X = 69
CARD_CONFIG = [
    (21, 609, 47, 642),
    (CARD_INIT_X, CARD_Y, CARD_INIT_X + CARD_WIDTH, CARD_Y + CARD_HEIGHT),
    (
        CARD_INIT_X + CARD_DELTA_X,
        CARD_Y,
        CARD_INIT_X + CARD_WIDTH + CARD_DELTA_X,
        CARD_Y + CARD_HEIGHT,
    ),
    (
        CARD_INIT_X + 2 * CARD_DELTA_X,
        CARD_Y,
        CARD_INIT_X + CARD_WIDTH + 2 * CARD_DELTA_X,
        CARD_Y + CARD_HEIGHT,
    ),
    (
        CARD_INIT_X + 3 * CARD_DELTA_X,
        CARD_Y,
        CARD_INIT_X + CARD_WIDTH + 3 * CARD_DELTA_X,
        CARD_Y + CARD_HEIGHT,
    ),
]

# Numbers
HP_WIDTH = 40
HP_HEIGHT = 10
LEFT_PRINCESS_HP_X = 74
RIGHT_PRINCESS_HP_X = 266
ALLY_PRINCESS_HP_Y = 404
ENEMY_PRINCESS_HP_Y = 95
ELIXIR_BOUNDING_BOX = (100, 628, 350, 643)
ALLY_HP_LHS_COLOUR = (111, 208, 252)
ALLY_HP_RHS_COLOUR = (63, 79, 112)
ENEMY_HP_LHS_COLOUR = (224, 35, 93)
ENEMY_HP_RHS_COLOUR = (90, 49, 68)
HPConfig = tuple[int, int, tuple[int, int, int], tuple[int, int, int]]
NUMBER_CONFIG: dict[str, HPConfig] = {
    "right_ally_princess_hp": (
        RIGHT_PRINCESS_HP_X,
        ALLY_PRINCESS_HP_Y,
        ALLY_HP_LHS_COLOUR,
        ALLY_HP_RHS_COLOUR,
    ),
    "left_ally_princess_hp": (
        LEFT_PRINCESS_HP_X,
        ALLY_PRINCESS_HP_Y,
        ALLY_HP_LHS_COLOUR,
        ALLY_HP_RHS_COLOUR,
    ),
    "right_enemy_princess_hp": (
        RIGHT_PRINCESS_HP_X,
        ENEMY_PRINCESS_HP_Y,
        ENEMY_HP_LHS_COLOUR,
        ENEMY_HP_RHS_COLOUR,
    ),
    "left_enemy_princess_hp": (
        LEFT_PRINCESS_HP_X,
        ENEMY_PRINCESS_HP_Y,
        ENEMY_HP_LHS_COLOUR,
        ENEMY_HP_RHS_COLOUR,
    ),
}

# Units
DETECTOR_UNITS = [
    Units.ARCHER,
    Units.ARCHER_QUEEN,
    Units.BALLOON,
    Units.BANDIT,
    Units.BARBARIAN,
    Units.BARBARIAN_HUT,
    Units.BAT,
    Units.BATTLE_HEALER,
    Units.BATTLE_RAM,
    Units.BOMB_TOWER,
    Units.BOMBER,
    Units.BOWLER,
    Units.BRAWLER,
    Units.CANNON,
    Units.CANNON_CART,
    Units.DARK_PRINCE,
    Units.DART_GOBLIN,
    Units.ELECTRO_DRAGON,
    Units.ELECTRO_GIANT,
    Units.ELECTRO_SPIRIT,
    Units.ELECTRO_WIZARD,
    Units.ELITE_BARBARIAN,
    Units.ELIXIR_COLLECTOR,
    Units.ELIXIR_GOLEM_LARGE,
    Units.ELIXIR_GOLEM_MEDIUM,
    Units.ELIXIR_GOLEM_SMALL,
    Units.EXECUTIONER,
    Units.FIRE_SPIRIT,
    Units.FIRE_CRACKER,
    Units.FISHERMAN,
    Units.FLYING_MACHINE,
    Units.FURNACE,
    Units.GIANT,
    Units.GIANT_SKELETON,
    Units.GIANT_SNOWBALL,
    Units.GOBLIN,
    Units.GOBLIN_CAGE,
    Units.GOBLIN_DRILL,
    Units.GOBLIN_HUT,
    Units.GOLDEN_KNIGHT,
    Units.GOLEM,
    Units.GOLEMITE,
    Units.GUARD,
    Units.HEAL_SPIRIT,
    Units.HOG,
    Units.HOG_RIDER,
    Units.BABY_DRAGON,
    Units.HUNTER,
    Units.ICE_GOLEM,
    Units.ICE_SPIRIT,
    Units.ICE_WIZARD,
    Units.INFERNO_DRAGON,
    Units.INFERNO_TOWER,
    Units.KNIGHT,
    Units.LAVA_HOUND,
    Units.LAVA_PUP,
    Units.LITTLE_PRINCE,
    Units.LUMBERJACK,
    Units.MAGIC_ARCHER,
    Units.MEGA_KNIGHT,
    Units.MEGA_MINION,
    Units.MIGHTY_MINER,
    Units.MINER,
    Units.MINION,
    Units.MINIPEKKA,
    Units.MONK,
    Units.MORTAR,
    Units.MOTHER_WITCH,
    Units.MUSKETEER,
    Units.NIGHT_WITCH,
    Units.PEKKA,
    Units.PHOENIX_EGG,
    Units.PHOENIX_LARGE,
    Units.PHOENIX_SMALL,
    Units.PRINCE,
    Units.PRINCESS,
    Units.RAM_RIDER,
    Units.RASCAL_BOY,
    Units.RASCAL_GIRL,
    Units.ROYAL_GHOST,
    Units.ROYAL_GIANT,
    Units.ROYAL_GUARDIAN,
    Units.ROYAL_HOG,
    Units.ROYAL_RECRUIT,
    Units.SKELETON,
    Units.SKELETON_DRAGON,
    Units.SKELETON_KING,
    Units.SPARKY,
    Units.SPEAR_GOBLIN,
    Units.TESLA,
    Units.TOMBSTONE,
    Units.VALKYRIE,
    Units.WALL_BREAKER,
    Units.WITCH,
    Units.WIZARD,
    Units.X_BOW,
    Units.ZAPPY,
]
