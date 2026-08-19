"""Turn the vector env's batched `infos` into the stats dict the shaping reads.

Both pipelines built this same 17-key dictionary inline, with the same defaults
and the same comments. It is a real piece of logic, not plumbing: EVERY default
here is chosen so that a MISSING key contributes exactly zero to the reward,
never a spurious spike.

WHY A KEY CAN BE MISSING AT ALL. gymnasium's info-batching only creates a key if
at least one env actually reported it this step (`AsyncVectorEnv._add_info`),
and the env wrappers return `{}` from `reset()`. So on the rare-but-real step
where EVERY env auto-resets at once -- several timing out on the same tick early
in training -- the whole key is absent rather than present with defaults.

The defaults are therefore not cosmetic:

  cumulative counters -> 0, because `compute_shaping`'s `delta()` clamp turns a
      counter that appears to go backwards into a delta of exactly 0.
  towers_alive        -> 3 (a full set), so a missing key yields a zero crown
      delta rather than a phantom three-crown swing.
  enemy_tower_hp / fireball_in_hand -> the "no opportunity" state, so a missing
      key can only ever zero the lethal-spell potential, never fabricate one.
"""
import numpy as np

#: Cumulative or instantaneous counters that default to a zero of their own
#: dtype. Split by dtype because elixir_spent is a float (card costs) while the
#: damage counters are ints.
_INT_KEYS = (
    "team0_troop_damage", "team1_troop_damage",
    "team0_building_damage", "team1_building_damage",
    "team0_tower_damage", "team1_tower_damage",
    #: Cumulative damage by the deck's win condition -- input to
    #: compute_shaping's win-condition term. Contributes exactly zero for a
    #: deck with no building-targeter, where the key is absent.
    "team0_wincon_damage",
)
_FLOAT_KEYS = ("team0_elixir_spent", "team1_elixir_spent")
#: Heuristic-1 inputs (`spell_value_shaping`). BOTH are needed: value-destroyed
#: alone makes a whiffed spell free, which is the guaranteed-zero trap that
#: parked the Cannon in a back corner.
_SPELL_KEYS = ("fireball_in_hand", "fireball_value_killed",
               "fireball_elixir_spent")


def extract_engine_stats(infos, num_envs):
    """{key: (num_envs,) array} for `rewards.shaping.compute_shaping`."""
    zeros = np.zeros(num_envs, dtype=np.int64)
    zeros_f = np.zeros(num_envs, dtype=np.float32)

    stats = {k: infos.get(k, zeros) for k in _INT_KEYS}
    stats.update({k: infos.get(k, zeros_f) for k in _FLOAT_KEYS})
    stats.update({k: np.asarray(infos.get(k, zeros), dtype=np.float32)
                  for k in _SPELL_KEYS})

    stats["enemy_tower_hp"] = np.asarray(
        infos.get("enemy_tower_hp", np.zeros((num_envs, 3), dtype=np.float32)),
        dtype=np.float32).reshape(num_envs, 3)
    # Instantaneous reading (not cumulative) -- feeds the overflow penalty.
    stats["team0_elixir_current"] = infos.get("elixir", zeros_f)
    stats["team0_towers_alive"] = infos.get(
        "team0_towers_alive", np.full(num_envs, 3, dtype=np.int64))
    stats["team1_towers_alive"] = infos.get(
        "team1_towers_alive", np.full(num_envs, 3, dtype=np.int64))
    return stats


def opponent_elixir_target(infos, num_envs):
    """Ground truth for the auxiliary head. Hidden information: it travels
    through `info` and never through the observation."""
    return np.asarray(
        infos.get("opp_elixir", np.zeros(num_envs, dtype=np.float32)),
        dtype=np.float32)
