"""Engine constants, derived ONCE from the compiled bindings.

CLAUDE.md's standing rule: never keep a second copy of an engine constant in
Python. This module is where that rule is enforced -- everything here is read
off `clash_royale_env` at import time, so a board-geometry or channel-layout
change on the C++ side propagates instead of silently drifting.

The history that makes the rule concrete: the river moved on 2026-07-29 and the
towers on 2026-07-30, and BOTH times a stale Python copy survived the edit
(model.py's "18*16=288" comment, and calibrate.py scoring bridges against
y=17.0). Where a value genuinely is not exposed by any binding, it belongs in
the module that needs it with a comment naming the header it came from --
`advisors/tactics.py` shows that pattern -- never here.
"""
import numpy as np

import clash_royale_env

# Spatial layout of the observation (must match ClashEnv.h):
# channels 0-2 ally troops (melee/ranged/tank), 3 ally buildings,
# channels 4-6 enemy troops, 7 enemy buildings, 8 river mask.
# Pulled live from the compiled engine's own exposed constants instead of a
# hardcoded copy -- a board-geometry or channel-layout change on the C++ side
# now propagates here automatically instead of silently drifting out of sync
# (confirmed painful in practice: this exact kind of drift crashed training
# more than once this project's history before these were queryable).
N_CHANNELS = clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS

BOARD_H = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT

BOARD_W = clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH

SPATIAL_SIZE = N_CHANNELS * BOARD_H * BOARD_W

# Same blanket HP normalizers ClashEnv::extractObservation() divides by when
# building the observation (MAX_TROOP_HP/MAX_BUILDING_HP in ClashEnv.h) --
# reused here purely to keep the shaping magnitude identical to the old
# HP-channel-diffing version below, now that the raw damage numbers come from
# the engine's MatchStatistics (via gym_wrapper's info dict) instead. Pulled
# live for the same drift-safety reason as N_CHANNELS/BOARD_H/BOARD_W above.
MAX_TROOP_HP = clash_royale_env.ClashRoyaleEnv.MAX_TROOP_HP

MAX_BUILDING_HP = clash_royale_env.ClashRoyaleEnv.MAX_BUILDING_HP

#: Cards in hand, and the card head's extra "play nothing this step" column.
#: Named because the placement entropy term has to be able to exclude the no-op
#: arm -- see `rl.ppo.PPOUpdater`'s mb_placed.
HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
NOOP_ACTION = HAND_SIZE

#: Scalars appended after the one-hot spatial block: elapsed time, both sides'
#: cumulative elixir spend, and the six tower HPs.
NUM_EXTRA_SCALARS = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS

def _own_tower_hp_total():
    """Total starting HP across our three towers, READ FROM THE ENGINE.

    Denominator for the flawless-defense bonus (W_FLAWLESS_DEFENSE). Taken
    from a fresh board's own appended tower scalars rather than written as
    2*2534 + 4008: CLAUDE.md's rule about second copies of engine constants,
    and the tower HPs are exactly the kind of number a balance pass moves.
    The scalars are hp / MAX_BUILDING_HP, so multiplying back recovers points.
    """
    E = clash_royale_env.ClashRoyaleEnv
    probe = E(list(range(8)), list(range(8)), 100)
    obs = np.asarray(probe.reset(), dtype=np.float32)
    tail = probe.observation_size() - E.NUM_EXTRA_SCALARS
    # tail layout: 0 time | 1-2 elixir spent | 3-5 OWN king/left/right | 6-8 enemy
    return float(obs[tail + 3:tail + 6].sum()) * E.MAX_BUILDING_HP

OWN_TOWER_HP_TOTAL = _own_tower_hp_total()

_CARD_NAME_CACHE = {}


def card_name(card_id):
    """Registry name for a card id, for diagnostic labels only.

    Evolutions share their base card's name verbatim (id 1 and id 128 are both
    "Archers"), so the id is appended -- a TensorBoard series that silently
    merged two cards would be worse than no series at all.
    """
    if card_id not in _CARD_NAME_CACHE:
        try:
            name = clash_royale_env.get_card_info(card_id)["name"]
        except Exception:  # noqa: BLE001 -- unknown id is a label problem, not fatal
            name = "card"
        _CARD_NAME_CACHE[card_id] = f"{name.replace(' ', '_')}_{card_id}"
    return _CARD_NAME_CACHE[card_id]
