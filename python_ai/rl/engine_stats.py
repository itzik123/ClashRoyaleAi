"""Turn the vector env's batched `infos` into the stats dict the shaping reads.

Every default makes a missing key contribute exactly zero to the reward. Keys
go missing because gymnasium only batches a key some env reported this step,
and reset() returns `{}`, so when every env auto-resets at once the key is
absent:

  cumulative counters -> 0 (compute_shaping's delta clamp turns it into 0)
  towers_alive        -> 3, so no phantom crown swing
  enemy_tower_hp, spell_* -> "no opportunity", which can only zero the
      lethal-spell potential; spell_damage/spell_cost 0.0 means "no damage
      spell", under which both spell terms are zero
"""
import numpy as np
import torch

#: Default to a zero of their own dtype: damage counters are ints, elixir is a
#: float.
_INT_KEYS = (
    "team0_troop_damage", "team1_troop_damage",
    "team0_building_damage", "team1_building_damage",
    "team0_tower_damage", "team1_tower_damage",
    #: Cumulative damage by the deck's win condition; absent for a deck without
    #: a building-targeter.
    "team0_wincon_damage",
)
_FLOAT_KEYS = ("team0_elixir_spent", "team1_elixir_spent")
#: Inputs to the two spell terms, for the deck's damage spell. Both value
#: killed and elixir spent are needed, or a whiffed spell is free. spell_damage
#: / spell_cost are per-deck constants published every step.
_SPELL_KEYS = ("spell_in_hand", "spell_value_killed", "spell_elixir_spent",
               "spell_damage", "spell_cost")


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
    # Instantaneous, not cumulative.
    stats["team0_elixir_current"] = infos.get("elixir", zeros_f)
    stats["team0_towers_alive"] = infos.get(
        "team0_towers_alive", np.full(num_envs, 3, dtype=np.int64))
    stats["team1_towers_alive"] = infos.get(
        "team1_towers_alive", np.full(num_envs, 3, dtype=np.int64))
    return stats


def reseat_prev_stats(stats, prev_stats, first_real):
    """`prev_stats` with the rows of `first_real` envs replaced by `stats`' own.

    The phantom post-autoreset step reports `{}`, so its stats are all
    defaults, and they become the "previous state" of the next episode's first
    real step. That is harmless for the counters and the tower potential but
    not for the solvency potential (Phi(elixir=0) = -0.1), which paid +0.084 on
    every episode's first step. Substituting the current row makes that
    transition's shaping ~0, correct since the true s0 was never observed.

    first_real: (num_envs,) bool -- envs whose previous step was the phantom.
    """
    first_real = np.asarray(first_real, dtype=bool)
    if prev_stats is None or not first_real.any():
        return prev_stats
    out = {}
    for key, prev in prev_stats.items():
        prev = np.asarray(prev)
        cur = np.asarray(stats[key])
        cond = first_real.reshape((-1,) + (1,) * (prev.ndim - 1))
        out[key] = np.where(cond, cur, prev).astype(prev.dtype, copy=False)
    return out


def opponent_played_card(infos, num_envs):
    """The card the opponent played during this step, or -1 for none.

    The raw per-step stream; `next_card_labels` turns it into training labels.
    Carried in `info`, never the observation. -1 (also the default for a
    missing key) contributes nothing to the loss.
    """
    return np.asarray(
        infos.get("opp_played_card", np.full(num_envs, -1, dtype=np.int64)),
        dtype=np.int64)


def next_card_labels(played, masks, valid):
    """(T, N) per-step plays -> (T, N) next-card labels and their loss mask.

    The label for step t is the first card the opponent plays at or after t,
    within the same episode, found by one backward scan. The carry is cleared
    at `masks[t] == 0` (an episode ended at t) before step t reads it, so
    labels never cross into the next match; step t's own play still counts.

    Steps with no future play get label -1 and mask 0 rather than a "no card"
    class, which would become the majority label for an artifact of where the
    episode stopped. Labels are clamped to 0 on masked rows so they stay legal
    indices.
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
