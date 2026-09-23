"""Scenario injection: reshaping the start-state distribution, not the reward.

A win condition dropped on the bridge is a "defend now or lose the tower"
moment, but in a full game its link to the tower loss ~40 ticks later is buried
in a long GAE trace, and the moment is rare. Starting a fraction of episodes in
such states gives the reflex gradient (as with curated start states in
robotics, AlphaGo, "Backplay").

It stays the same game: the opponent keeps playing normally, the engine and
board are real, and the ordinary reward scores the defence. The optional
truncation window (`max_steps`) bootstraps a value (`rl/gae.py`), never a
terminal.

A scenario dict: name, spawns [(card_id, x, y)], max_steps (None runs to the
natural end), defensive, and optionally require_own_card / warmup_ticks.
"""
import numpy as np

from python_ai import engine_constants as EC

SCENARIO_INJECTION_PROB = 0.30

# Building-targeting win conditions that beeline for a tower. Excluded: Miner
# (registered here as a plain melee squad), Goblin Barrel and Graveyard
# (spells; inject_enemy is unverified for spell-shaped cards).
_WIN_CONDITION_IDS = [
    15,  # Hog Rider
    18,  # Royal Giant
    45,  # Balloon
    2,   # Giant
    19,  # Golem
    81,  # Battle Ram
    82,  # Royal Hogs
    83,  # Wall Breakers
    84,  # Electro Giant
    87,  # Ram Rider
    88,  # Goblin Giant
    89,  # Skeleton Barrel
    91,  # Lava Hound
]

# Ranged units that escort a push (the supported scenario's second spawn).
_SUPPORT_IDS = [
    6,   # Musketeer
    1,   # Archers
    11,  # Wizard
    44,  # Baby Dragon
    63,  # Magic Archer
    36,  # Executioner
    20,  # Dart Goblin
]

# Board coordinates for inject_enemy, which skips isValidPlacement, so a spawn
# inside the river band is allowed. y = 17.0 is inside the band, and
# Board::getNextWaypoint routes it to the bridge exit: an on-the-bridge spawn.
_RIVER_Y = 17.0

# Derived from ArenaLayout.
_BRIDGE_LANES = [EC.LEFT_BRIDGE_X, EC.RIGHT_BRIDGE_X]

def _scenario_bridge_push(rng):
    """One enemy win condition on a random bridge. A short window: the defence
    resolves in ~2-4 steps, the rest lets a counter-push start.
    """
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push",
        "defensive": True,
        "spawns": [(int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y)],
        "max_steps": 15,
    }

def _scenario_bridge_push_supported(rng):
    """A win condition plus a ranged support behind it in the same lane, which one
    cheap defender cannot fully answer. A longer window.
    """
    lane = rng.choice(_BRIDGE_LANES)
    return {
        "name": "bridge_push_supported",
        "defensive": True,
        "spawns": [
            (int(rng.choice(_WIN_CONDITION_IDS)), lane, _RIVER_Y),
            (int(rng.choice(_SUPPORT_IDS)), lane, _RIVER_Y + 3.0),
        ],
        "max_steps": 25,
    }

# Spell target practice. The policy rarely casts Fireball because its aim is
# poor (it missed a cluster by ~3 tiles on its own half and 8-11 at an enemy
# tower, against a 2.5 radius), so declining is an accurate valuation; practice
# fixes the aim without biasing the reward.
#
# Low-HP bodies only, so one good cast kills them and a whiff is punished
# (Barbarians at 691 HP would survive a perfect 689 cast).
_FIREBALL_SWARM_IDS = [
    41,  # Minions (3 bodies)
    1,   # Archers (2 bodies)
    64,  # Firecracker (304 hp)
    6,   # Musketeer
]

def _scenario_fireball_swarm(rng):
    """A cheap swarm inside our half: the defensive spell cast.

    Past the river and short of the Princess Towers, so it must be answered
    now; several cards, so one spell is a large positive trade.
    """
    lane = rng.choice(_BRIDGE_LANES)
    n = int(rng.integers(3, 6))
    cx, cy = lane, 12.0
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-1.1, 1.1)),
                       float(cy + rng.uniform(-1.1, 1.1))))
    # Defensive: the swarm is in our half.
    return {"name": "fireball_swarm", "spawns": spawns, "max_steps": 12,
            "defensive": True}

def _scenario_fireball_tower_value(rng):
    """Enemy troops hugging their own Princess Tower: the two-for-one cast.
    Spawned just in front of the tower so troops and tower fit in one 2.5
    radius.
    """
    # Derived from ArenaLayout.
    tower_x = EC.LEFT_LANE_X if rng.random() < 0.5 else EC.RIGHT_LANE_X
    n = int(rng.integers(2, 5))
    cx, cy = tower_x, 25.6
    spawns = []
    for _ in range(n):
        spawns.append((int(rng.choice(_FIREBALL_SWARM_IDS)),
                       float(cx + rng.uniform(-0.9, 0.9)),
                       float(cy + rng.uniform(-0.9, 0.9))))
    # Not defensive: nothing threatens us, so "did not take a big hit" holds
    # whatever the agent does.
    return {"name": "fireball_tower_value", "spawns": spawns, "max_steps": 15,
            "defensive": False}

# (builder_fn, weight). Set a scenario's "max_steps" to None to run it to the
# natural end of the game. Spell scenarios take 60% of the weight: the bridge
# pushes had saturated (ScenDef ~0.99).
SCENARIOS = [
    (_scenario_bridge_push, 1.0),
    (_scenario_bridge_push_supported, 1.0),
    (_scenario_fireball_swarm, 1.5),
    (_scenario_fireball_tower_value, 1.5),
]

#: Scenarios whose answer is an area-damage spell; drawn only when the deck
#: holds one.
_SPELL_SCENARIOS = (_scenario_fireball_swarm, _scenario_fireball_tower_value)


def _deck_weights(deck):
    """SCENARIOS' weights, with the spell scenarios zeroed for a deck that cannot
    answer them.

    Otherwise a spell-less deck spends 18% of all injected episodes practising
    a reflex it cannot express. The shipped deck's weights are unchanged.
    """
    from python_ai.advisors import card_probes
    has_spell = any(card_probes.spell_effect(int(c)) is not None for c in deck)
    return [0.0 if (b in _SPELL_SCENARIOS and not has_spell) else w
            for b, w in SCENARIOS]


def spell_scenario_share(deck):
    """Fraction of injected scenarios that are spell scenarios, for this deck.
    """
    w = _deck_weights(deck)
    total = sum(w)
    return sum(wi for (b, _), wi in zip(SCENARIOS, w) if b in _SPELL_SCENARIOS) / total


def sample_scenario(rng, deck=None):
    """One scenario dict. `deck` defaults to the trainee's (python_ai.deck)."""
    if deck is None:
        from python_ai.deck import DEFAULT_DECK as deck
    builders = [b for b, _ in SCENARIOS]
    weights = np.array(_deck_weights(tuple(deck)), dtype=np.float64)
    weights /= weights.sum()
    return builders[rng.choice(len(builders), p=weights)](rng)
