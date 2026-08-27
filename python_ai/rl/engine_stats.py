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
import torch

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


def opponent_played_card(infos, num_envs):
    """Which card the opponent played during THIS step, or -1 for none.

    Raw per-step stream, not yet the training label -- `next_card_labels`
    turns it into one. Hidden information: it travels through `info` and never
    through the observation, because at the moment the agent acted this play
    had not happened yet.

    The default is -1 ("nothing played"), which is the value that contributes
    exactly zero to the loss, matching this module's rule for every other
    default: on the all-envs-auto-reset step the key is absent entirely.
    """
    return np.asarray(
        infos.get("opp_played_card", np.full(num_envs, -1, dtype=np.int64)),
        dtype=np.int64)


def next_card_labels(played, masks, valid):
    """(T, N) per-step plays -> (T, N) NEXT-card labels + their loss mask.

    The label for step t is the first card the opponent plays at or after t,
    within the same episode. Computed by ONE backward scan carrying the next
    known play, which is why this is done once per update rather than per
    minibatch.

    `masks[t] == 0` marks the step an episode ENDED on. Scanning backward, the
    carry has to be cleared there BEFORE step t reads it: everything after t
    belongs to a different episode and using it would teach the net to predict
    the next match's opening play from this match's final state. Step t's own
    `played[t]` is still valid -- the episode ended after that play, not
    before it.

    Steps with no future play (the tail of every episode, where the opponent
    simply never plays again) get label -1 and mask 0. They are DROPPED, not
    given a "no card" class: "they played nothing for the rest of the match"
    is an artifact of where the episode stopped, not a fact about the
    opponent, and giving it a class would make it the majority label.

    Returns (labels, has_label) with labels clamped to 0 where has_label is 0,
    so the tensor is always a legal index for cross_entropy even on the rows
    the mask discards.
    """
    T, N = played.shape
    labels = torch.full_like(played, -1)
    carry = torch.full((N,), -1, dtype=played.dtype, device=played.device)
    for t in range(T - 1, -1, -1):
        carry = torch.where(masks[t] == 0,
                            torch.full_like(carry, -1), carry)
        step = played[t]
        labels[t] = torch.where(step >= 0, step, carry)
        carry = labels[t]
    has_label = ((labels >= 0).float() * valid)
    return labels.clamp_min(0), has_label
